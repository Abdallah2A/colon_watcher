import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree
import logging
from utils.image_utils import _load_and_convert_mask, _preprocess_mask, _extract_contours


def _contour_to_annotation(
        contour: np.ndarray,
        mask_width: int,
        mask_height: int
) -> tuple[float, float, float, float] | None:
    """
    Convert a contour to a YOLO annotation in normalized format.
    """
    try:
        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / mask_width
        y_center = (y + h / 2) / mask_height
        width = w / mask_width
        height = h / mask_height
        return x_center, y_center, width, height
    except Exception as e:
        logging.error(f"Error in contour_to_annotation: {e}")
        return None


def _extract_annotations_from_mask_image(mask_path: str) -> list[tuple[float, float, float, float]]:
    """
    Load mask image, extract its annotations, and return them as YOLO annotations.
    """
    try:
        mask = _load_and_convert_mask(mask_path)
        if mask is None:
            return []

        mask_height, mask_width = mask.shape[:2]
        binary_mask = _preprocess_mask(mask)
        if binary_mask is None:
            return []

        contours = _extract_contours(binary_mask)
        annotations = []
        for contour in contours:
            annotation = _contour_to_annotation(contour, mask_width, mask_height)
            if annotation is not None:
                annotations.append(annotation)
        return annotations

    except Exception as e:
        logging.error(f"Unexpected error while processing mask image {mask_path}: {e}")
        return []


def _extract_xml_data(xml_path: str) -> tuple[None | int, None | int, list[tuple[int, int, int, int]]]:
    """
    Extract the image size and bounding boxes from the XML file.
    """
    try:
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find('size')
        if size_elem is None:
            logging.error(f"Missing <size> tag in XML: {xml_path}")
            return None, None, []

        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')
        if width_elem is None or height_elem is None:
            logging.error(f"Missing width/height in <size> tag of XML: {xml_path}")
            return None, None, []

        image_width = int(width_elem.text)
        image_height = int(height_elem.text)

        bounding_boxes = []
        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                logging.error(f"Missing <bndbox> tag in object entry of XML: {xml_path}")
                continue

            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')
            if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                logging.error(f"Incomplete bounding box data in XML: {xml_path}")
                continue

            bounding_boxes.append((
                int(xmin_elem.text), int(ymin_elem.text),
                int(xmax_elem.text), int(ymax_elem.text)
            ))
        return image_width, image_height, bounding_boxes

    except ElementTree.ParseError as e:
        logging.error(f"Failed to parse XML file: {xml_path}. Error: {e}")
    except FileNotFoundError as e:
        logging.error(f"XML file not found: {xml_path}. Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while extracting XML data from {xml_path}: {e}")
    return None, None, []


def _convert_to_yolo(
        image_width: int,
        image_height: int,
        bounding_boxes: list[tuple[int, int, int, int]]
) -> list[tuple[float, float, float, float]]:
    """
    Convert bounding boxes from pixel coordinates to normalized YOLO format.
    """
    annotations = []
    try:
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height
            annotations.append((x_center, y_center, bbox_width, bbox_height))
    except Exception as e:
        logging.error(f"Error converting bounding boxes to YOLO format: {e}")
    return annotations


def _parse_xml(xml_path: str) -> list[tuple[float, float, float, float]]:
    image_width, image_height, bounding_boxes = _extract_xml_data(xml_path)
    if image_width is None or image_height is None:
        return []
    return _convert_to_yolo(image_width, image_height, bounding_boxes)


def _check_objects_size(
        annotations: list[tuple[float, float, float, float]],
        min_size: int,
        image_width: int,
        image_height: int
) -> bool:
    """Checks if any object in the annotations is smaller than the given minimum size."""
    try:
        for _, _, width, height in annotations:
            abs_width = width * image_width
            abs_height = height * image_height
            if abs_width < min_size or abs_height < min_size:
                return True

        return False

    except Exception as e:
        logging.error(f"Unexpected error in check_objects_size: {e}")
        return False


def _clamp_annotation(
        x_center: float,
        y_center: float,
        width: float,
        height: float
) -> tuple[float, float, float, float] | None:
    """Clamp all values to the [0,1] range and ensure valid dimensions."""
    try:
        x_min = max(0.0, min(x_center - width / 2, 1.0))
        x_max = max(0.0, min(x_center + width / 2, 1.0))
        y_min = max(0.0, min(y_center - height / 2, 1.0))
        y_max = max(0.0, min(y_center + height / 2, 1.0))
        clamped_width = max(0.01, x_max - x_min)
        clamped_height = max(0.01, y_max - y_min)
        clamped_x = (x_min + x_max) / 2
        clamped_y = (y_min + y_max) / 2
        return clamped_x, clamped_y, clamped_width, clamped_height

    except Exception as e:
        logging.error(f"Unexpected error in clamp_annotation: {e}")
        return None


def _adjust_annotations_for_resize(
        annotations: list[tuple[float, float, float, float]],
        params: dict
) -> list[tuple[float, float, float, float]]:
    """
    Adjust YOLO annotations based on the resizing parameters.
    """
    new_annotations = []
    try:
        new_w = params.get('new_w')
        new_h = params.get('new_h')
        offset_x = params.get('offset_x')
        offset_y = params.get('offset_y')
        s = params.get('s')

        for annotation in annotations:
            try:
                center_x, center_y, box_width, box_height = annotation

                new_center_x = (offset_x + center_x * new_w) / s
                new_center_y = (offset_y + center_y * new_h) / s
                new_box_width = (box_width * new_w) / s
                new_box_height = (box_height * new_h) / s

                new_annotations.append(_clamp_annotation(new_center_x, new_center_y, new_box_width, new_box_height))
            except Exception as inner_e:
                logging.error(f"Error adjusting annotation {annotation}: {inner_e}")
        return new_annotations

    except Exception as e:
        logging.error(f"Error in adjust_annotations_for_resize: {e}", exc_info=True)
        return []


def _flip_annotations(
        annotations: list[tuple[float, float, float, float]],
        flip_code: int
) -> list[tuple[float, float, float, float]]:
    """
    Flip the YOLO annotations based on the provided flip code.
    """
    processed_annotations = []
    try:
        for xc, yc, w, h in annotations:
            try:
                if flip_code in {0, -1}:
                    yc = 1.0 - yc
                if flip_code in {1, -1}:
                    xc = 1.0 - xc
                processed_annotations.append(_clamp_annotation(xc, yc, w, h))
            except Exception as inner_e:
                logging.error(f"Error flipping annotation ({xc}, {yc}, {w}, {h}): {inner_e}")
        return processed_annotations
    except Exception as e:
        logging.error(f"Error processing annotations in flip_annotations: {e}")
        return []


def _adjust_annotations_with_rotation(
        annotations: list[tuple[float, float, float, float]],
        rotation_matrix: np.ndarray,
        width: int,
        height: int
) -> list[tuple[float, float, float, float]]:
    """
    Adjust YOLO annotations according to the provided rotation matrix.
    """
    rotated_annotations = []
    try:
        for x_center, y_center, box_width, box_height in annotations:
            try:
                abs_x_center = x_center * width
                abs_y_center = y_center * height
                abs_box_width = box_width * width
                abs_box_height = box_height * height
                x_min = abs_x_center - abs_box_width / 2
                y_min = abs_y_center - abs_box_height / 2
                x_max = abs_x_center + abs_box_width / 2
                y_max = abs_y_center + abs_box_height / 2
                corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                ones = np.ones((4, 1))
                corners_homogeneous = np.hstack([corners, ones])
                transformed_corners = rotation_matrix.dot(corners_homogeneous.T).T
                transformed_x = transformed_corners[:, 0]
                transformed_y = transformed_corners[:, 1]
                new_x_min = max(0, min(transformed_x))
                new_x_max = min(width, max(transformed_x))
                new_y_min = max(0, min(transformed_y))
                new_y_max = min(height, max(transformed_y))
                new_box_width = new_x_max - new_x_min
                new_box_height = new_y_max - new_y_min
                new_x_center = (new_x_min + new_x_max) / 2 / width
                new_y_center = (new_y_min + new_y_max) / 2 / height
                new_box_width_norm = new_box_width / width
                new_box_height_norm = new_box_height / height
                clamped = _clamp_annotation(new_x_center, new_y_center, new_box_width_norm, new_box_height_norm)
                rotated_annotations.append(clamped)
            except Exception as inner_e:
                logging.error(
                    f"Error processing annotation ({x_center}, {y_center}, {box_width}, {box_height}): {inner_e}")
    except Exception as e:
        logging.error(f"Error in adjust_annotations_with_rotation: {e}")
    return rotated_annotations


def _adjust_annotations_for_crop(
        annotations: list[tuple[float, float, float, float]],
        crop_params: dict
) -> list[tuple[float, float, float, float]]:
    """
    Adjust YOLO annotations for the cropped and resized image, filtering out non-intersecting objects.
    """
    adjusted_annotations = []
    try:
        if crop_params is None:
            logging.error("No crop parameters provided.")
            return annotations

        crop_x_min = crop_params['crop_x_min']
        crop_y_min = crop_params['crop_y_min']
        s = crop_params['crop_width']
        orig_width = crop_params['orig_width']
        orig_height = crop_params['orig_height']
        target_size = 640

        for x_center, y_center, box_width, box_height in annotations:
            abs_x_center = x_center * orig_width
            abs_y_center = y_center * orig_height
            abs_box_width = box_width * orig_width
            abs_box_height = box_height * orig_height
            x_min = abs_x_center - abs_box_width / 2
            y_min = abs_y_center - abs_box_height / 2
            x_max = abs_x_center + abs_box_width / 2
            y_max = abs_y_center + abs_box_height / 2
            x_min_cropped = max(0, x_min - crop_x_min)
            x_max_cropped = min(s, x_max - crop_x_min)
            y_min_cropped = max(0, y_min - crop_y_min)
            y_max_cropped = min(s, y_max - crop_y_min)
            if x_min_cropped < x_max_cropped and y_min_cropped < y_max_cropped:
                x_min_resized = (x_min_cropped / s) * target_size
                x_max_resized = (x_max_cropped / s) * target_size
                y_min_resized = (y_min_cropped / s) * target_size
                y_max_resized = (y_max_cropped / s) * target_size
                x_center_resized = (x_min_resized + x_max_resized) / 2
                y_center_resized = (y_min_resized + y_max_resized) / 2
                box_width_resized = x_max_resized - x_min_resized
                box_height_resized = y_max_resized - y_min_resized
                x_center_norm = x_center_resized / target_size
                y_center_norm = y_center_resized / target_size
                box_width_norm = box_width_resized / target_size
                box_height_norm = box_height_resized / target_size
                adjusted_annotations.append(_clamp_annotation(x_center_norm, y_center_norm,
                                                              box_width_norm, box_height_norm))

        return adjusted_annotations

    except Exception as e:
        logging.error(f"Unexpected error in adjust_annotations_for_crop: {e}")
        return []


def _adjust_annotations_for_stretch(
    annotations: list[tuple[float, float, float, float]],
    stretch_params: dict
) -> list[tuple[float, float, float, float]]:
    """
    Adjust YOLO annotations after stretching and padding.
    """
    adjusted_annotations = []
    try:
        if stretch_params is None:
            logging.error("No stretch parameters provided for annotation adjustment.")
            return annotations

        pad_left = stretch_params['pad_left']
        pad_top = stretch_params['pad_top']
        new_width = stretch_params['new_width']
        new_height = stretch_params['new_height']
        max_dim = stretch_params['max_dim']

        for x_center, y_center, box_width, box_height in annotations:
            try:
                x_center_final = (pad_left + x_center * new_width) / max_dim
                y_center_final = (pad_top + y_center * new_height) / max_dim
                box_width_final = box_width * (new_width / max_dim)
                box_height_final = box_height * (new_height / max_dim)
                x_center_final = max(0.0, min(1.0, x_center_final))
                y_center_final = max(0.0, min(1.0, y_center_final))
                box_width_final = max(0.0, min(1.0, box_width_final))
                box_height_final = max(0.0, min(1.0, box_height_final))
                adjusted_annotations.append(_clamp_annotation(x_center_final, y_center_final,
                                                              box_width_final, box_height_final))
            except Exception as inner_e:
                logging.error(f"Error processing annotation {x_center, y_center, box_width, box_height}: {inner_e}")
        return adjusted_annotations
    except Exception as e:
        logging.error(f"Unexpected error in adjust_annotations_for_stretch: {e}")
        return []


def _save_yolo_annotations(file_path: str, annotations: list[tuple[float, float, float, float]]):
    """
    Save bounding box annotations in YOLO format.
    """
    try:
        with open(file_path, 'w') as f:
            for xc, yc, w, h in annotations:
                xc, yc, w, h = _clamp_annotation(xc, yc, w, h)
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    except Exception as e:
        logging.error(f"Failed to save YOLO annotations: {e}")


def _check_black_masks(mask_path: str, threshold: int = 1) -> bool:
    """Check if mask image have objects."""
    from utils.image_utils import _load_image

    mask = _load_image(mask_path)
    return np.max(mask) <= threshold


def _check_valid_xml(xml_path: str) -> bool:
    """Check if the XML have objects."""
    _, _, bounding_boxes = _extract_xml_data(xml_path)
    if not bounding_boxes:
        return True
    return False
