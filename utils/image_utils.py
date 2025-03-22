import cv2
import numpy as np
import random
import logging
from collections.abc import Sequence


def _load_image(image_path: str) -> np.ndarray | None:
    """Load an image from the given path.

    :param image_path: Path to the image file.
    :return: Numpy ndarray of the image if successful, otherwise None.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image (file not found or unreadable): {image_path}")
            return None
        return image
    except Exception as e:
        logging.error(f"Unexpected error while loading image from {image_path}: {e}")
        return None


def _load_and_convert_mask(mask_path: str) -> np.ndarray | None:
    """
    Load an image from the given path and convert it to grayscale.

    :param mask_path: Path to the mask image.
    :return: Grayscale image or None if loading or conversion fails.
    """
    try:
        mask = _load_image(mask_path)
        if mask is None:
            logging.error(f"Failed to load mask image: {mask_path}")
            return None
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logging.error(f"Error in load_and_convert_mask: {e}")
        return None


def _preprocess_mask(mask: np.ndarray) -> np.ndarray | None:
    """
    Preprocess the grayscale mask image by applying Gaussian blur and Otsu thresholding.

    :param mask: Grayscale mask image.
    :return: Binary mask after thresholding or None if an error occurs.
    """
    try:
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, binary_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary_mask
    except Exception as e:
        logging.error(f"Error in preprocess_mask: {e}")
        return None


def _extract_contours(binary_mask: np.ndarray) -> Sequence[np.ndarray]:
    """
    Extract external contours from the binary mask.

    :param binary_mask: Processed binary mask.
    :return: List of contours or an empty list if an error occurs.
    """
    try:
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    except Exception as e:
        logging.error(f"Error in extract_contours: {e}")
        return []


def _resize_image_with_padding(image: np.ndarray, resize_shape: int) -> tuple[np.ndarray | None, dict]:
    """
    Resize the image to a square (resize_shape x resize_shape) while maintaining aspect ratio by adding black padding.

    Args:
        image: Input image as a NumPy array of shape (h, w, c).
        resize_shape: Target size for the square output image.

    Returns:
        A tuple containing:
            - Padded image as a NumPy array of shape (resize_shape, resize_shape, c) or None on error.
            - A dictionary with parameters for annotation adjustment.
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        if not isinstance(resize_shape, int) or resize_shape <= 0:
            raise ValueError("resize_shape must be a positive integer.")

        h, w, c = image.shape
        s = resize_shape
        scale = min(s / w, s / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.zeros((s, s, c), dtype=image.dtype)
        offset_x = (s - new_w) // 2
        offset_y = (s - new_h) // 2
        padded_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w, :] = resized_image

        params = {
            'new_w': new_w,
            'new_h': new_h,
            'offset_x': offset_x,
            'offset_y': offset_y,
            's': s
        }
        return padded_image, params

    except Exception as e:
        logging.error(f"Error in resize_image_with_padding: {str(e)}", exc_info=True)
        return None, {}


def _resize_image_and_annotations(
    image: np.ndarray,
    annotations: list[tuple[float, float, float, float]],
    resize_shape: int
) -> tuple[np.ndarray | None, list[tuple[float, float, float, float]]]:
    """
    Resize an image and its YOLO annotations while maintaining aspect ratio by adding black padding.
    """
    from utils.annotation_utils import _adjust_annotations_for_resize

    try:
        padded_image, params = _resize_image_with_padding(image, resize_shape)
        if padded_image is None or not params:
            return None, []
        if annotations:
            annotations = _adjust_annotations_for_resize(annotations, params)
        return padded_image, annotations

    except Exception as e:
        logging.error(f"Error in _resize_image_and_annotations: {str(e)}", exc_info=True)
        return None, []


def _flip_image(image: np.ndarray, flip_code: int) -> np.ndarray | None:
    """
    Flip the image based on the provided flip code.
    """
    try:
        return cv2.flip(image, flip_code)
    except Exception as e:
        logging.error(f"Error flipping image: {e}")
        return None


def _apply_flip_augmentation(
    image: np.ndarray,
    annotations: list[tuple[float, float, float, float]]
) -> tuple[np.ndarray | None, list[tuple[float, float, float, float]]]:
    """
    Apply random flip augmentation to the image and its annotations.
    """
    from utils.annotation_utils import _flip_annotations

    try:
        if image is None:
            logging.error("Input image is None.")
            return None, []

        if not annotations:
            logging.warning("No annotations to flip.")

        flip_code = random.choice([-1, 0, 1])
        flipped_image = _flip_image(image, flip_code)
        processed_annotations = _flip_annotations(annotations, flip_code)

        return flipped_image, processed_annotations
    except Exception as e:
        logging.error(f"Unexpected error in apply_flip_augmentation: {e}")
        return None, []


def _rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    """
    Rotate the image by a given angle while maintaining its original dimensions.
    """
    try:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        return rotated_image, rotation_matrix, width, height
    except Exception as e:
        logging.error(f"Error in rotate_image: {e}")
        return None, None, 0, 0


def _apply_rotation_augmentation(
        image: np.ndarray,
        annotations: list[tuple[float, float, float, float]]
) -> tuple[np.ndarray | None, list[tuple[float, float, float, float]]]:
    """
    Apply random rotation augmentation to the image and its annotations.
    """
    from utils.annotation_utils import _adjust_annotations_with_rotation

    try:
        if image is None:
            logging.error("Input image is None.")
            return None, []
        if not annotations:
            logging.warning("No annotations to rotate.")

        angle = random.uniform(-45, 45)
        rotated_image, rotation_matrix, width, height = _rotate_image(image, angle)
        if rotated_image is None or rotation_matrix is None:
            return None, []

        rotated_annotations = _adjust_annotations_with_rotation(annotations, rotation_matrix, width, height)
        return rotated_image, rotated_annotations

    except Exception as e:
        logging.error(f"Unexpected error in _apply_rotation_augmentation: {e}")
        return None, []


def _crop_and_resize_image(
        image: np.ndarray,
        annotations: list[tuple[float, float, float, float]]
) -> tuple[np.ndarray | None, dict | None]:
    """
    Crop a random region from the image that includes at least one object, then resize to a standard size.
    """
    try:
        if image is None:
            logging.error("Input image is None.")
            return None, None

        if not annotations:
            logging.warning("No annotations provided. Returning original image.")
            return image, None

        orig_height, orig_width = image.shape[:2]
        selected = random.choice(annotations)
        x_center, y_center, box_width, box_height = selected
        abs_x_center = x_center * orig_width
        abs_y_center = y_center * orig_height
        abs_box_width = box_width * orig_width
        abs_box_height = box_height * orig_height
        x_min = abs_x_center - abs_box_width / 2
        y_min = abs_y_center - abs_box_height / 2
        x_max = abs_x_center + abs_box_width / 2
        y_max = abs_y_center + abs_box_height / 2
        box_width = x_max - x_min
        box_height = y_max - y_min
        min_s = max(box_width, box_height)
        max_s = min(orig_width, orig_height)
        s = random.uniform(min_s, max_s)
        x_min_range_start = max(0, x_max - s)
        x_min_range_end = min(x_min, orig_width - s)
        y_min_range_start = max(0, y_max - s)
        y_min_range_end = min(y_min, orig_height - s)
        if x_min_range_start > x_min_range_end or y_min_range_start > y_min_range_end:
            logging.error("Invalid crop ranges computed.")
            return image, None
        crop_x_min = random.uniform(x_min_range_start, x_min_range_end)
        crop_y_min = random.uniform(y_min_range_start, y_min_range_end)
        crop_x_max = crop_x_min + s
        crop_y_max = crop_y_min + s
        cropped_image = image[int(crop_y_min):int(crop_y_max), int(crop_x_min):int(crop_x_max)]
        target_size = (640, 640)
        resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
        crop_params = {
            'crop_x_min': crop_x_min,
            'crop_y_min': crop_y_min,
            'crop_width': s,
            'crop_height': s,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'target_width': target_size[0],
            'target_height': target_size[1]
        }
        return resized_image, crop_params

    except Exception as e:
        logging.error(f"Unexpected error in crop_and_resize_image: {e}")
        return None, None


def _apply_crop_augmentation(
        image: np.ndarray,
        annotations: list[tuple[float, float, float, float]]
) -> tuple[np.ndarray | None, list[tuple[float, float, float, float]]]:
    """
    Apply cropping augmentation to focus on objects and adjust annotations accordingly.
    """
    from utils.annotation_utils import _adjust_annotations_for_crop

    try:
        resized_image, crop_params = _crop_and_resize_image(image, annotations)
        if resized_image is None or crop_params is None:
            return image, annotations

        resized_annotations = _adjust_annotations_for_crop(annotations, crop_params)
        return resized_image, resized_annotations

    except Exception as e:
        logging.error(f"Unexpected error in apply_crop_augmentation: {e}")
        return image, annotations


def _stretch_and_resize_image(image: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
    """
    Stretch an image with random scaling factors, pad it to a square, and resize to target size.
    """
    try:
        if image is None:
            logging.error("Input image is None.")
            return None, None

        orig_height, orig_width = image.shape[:2]
        target_size = (640, 640)
        scale_x = random.uniform(0.5, 1.5)
        scale_y = random.uniform(0.5, 1.5)
        new_width = int(orig_width * scale_x)
        new_height = int(orig_height * scale_y)
        stretched_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        max_dim = max(new_width, new_height)
        pad_left = (max_dim - new_width) // 2
        pad_right = max_dim - new_width - pad_left
        pad_top = (max_dim - new_height) // 2
        pad_bottom = max_dim - new_height - pad_top
        padded_image = cv2.copyMakeBorder(
            stretched_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        final_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)
        params = {
            'pad_left': pad_left,
            'pad_top': pad_top,
            'new_width': new_width,
            'new_height': new_height,
            'max_dim': max_dim,
            'target_size': target_size
        }
        return final_image, params

    except Exception as e:
        logging.error(f"Unexpected error in stretch_and_resize_image: {e}")
        return None, None


def _apply_stretch_augmentation(
    image: np.ndarray,
    annotations: list[tuple[float, float, float, float]]
) -> tuple[np.ndarray | None, list[tuple[float, float, float, float]]]:
    """
    Apply stretch augmentation to an image and adjust its annotations.
    """
    from utils.annotation_utils import _adjust_annotations_for_stretch

    try:
        if image is None:
            logging.error("Input image is None.")
            return None, []

        if not annotations:
            logging.warning("No annotations provided. Returning original image.")
            return image, annotations

        final_image, stretch_params = _stretch_and_resize_image(image)
        if final_image is None or stretch_params is None:
            logging.warning("Image stretching failed. Returning original image and annotations.")
            return image, annotations

        adjusted_annotations = _adjust_annotations_for_stretch(annotations, stretch_params)
        return final_image, adjusted_annotations

    except Exception as e:
        logging.error(f"Unexpected error in apply_stretch_augmentation: {e}")
        return None, []


def _save_image_jpg(file_path: str, image: np.ndarray, quality: int = 100):
    """
    Save an image as a JPG file with the specified quality.
    """
    try:
        cv2.imwrite(file_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    except Exception as e:
        logging.error(f"Exception occurred while saving image: {e}")


def _resize_image_with_padding_normal(image: np.ndarray, resize_shape: int = 640) -> np.ndarray | None:
    """
    Resize an image to the specified shape with padding to maintain aspect ratio (for normal data).

    :param image: Input image as a NumPy array.
    :param resize_shape: Target size for both width and height (default: 640).
    :return: Resized and padded image, or None if processing fails.
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        if not isinstance(resize_shape, int) or resize_shape <= 0:
            raise ValueError("resize_shape must be a positive integer.")
        h, w = image.shape[:2]
        scale = min(resize_shape / w, resize_shape / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.zeros((resize_shape, resize_shape, 3), dtype=image.dtype)
        offset_x = (resize_shape - new_w) // 2
        offset_y = (resize_shape - new_h) // 2
        padded_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w, :] = resized_image
        return padded_image
    except Exception as e:
        logging.error(f"Error in resizing: {str(e)}")
        return None


def _apply_random_augmentation_normal(image: np.ndarray) -> np.ndarray:
    """
    Apply a random augmentation (flip, rotate, crop, or stretch) to the image (for normal data).

    :param image: Input image as a NumPy array.
    :return: Augmented image as a NumPy array.
    """
    try:
        aug_type = random.choice(['flip', 'rotate', 'crop', 'stretch'])
        if aug_type == 'flip':
            flip_code = random.choice([0, 1, -1])  # Vertical, horizontal, or both
            return cv2.flip(image, flip_code)
        elif aug_type == 'rotate':
            angle = random.uniform(-45, 45)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, mat, (w, h))
        elif aug_type == 'crop':
            h, w = image.shape[:2]
            crop_h = int(random.uniform(0.5, 1.0) * h)
            crop_w = int(random.uniform(0.5, 1.0) * w)
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            cropped = image[y:y + crop_h, x:x + crop_w]
            return cv2.resize(cropped, (w, h))
        elif aug_type == 'stretch':
            h, w = image.shape[:2]
            scale_w = random.uniform(0.8, 1.2)
            scale_h = random.uniform(0.8, 1.2)
            new_w = int(w * scale_w)
            new_h = int(h * scale_h)
            stretched = cv2.resize(image, (new_w, new_h))
            return cv2.resize(stretched, (w, h))
    except Exception as e:
        logging.error(f"Error in augmentation: {str(e)}")
        return image
