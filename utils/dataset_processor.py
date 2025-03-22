import os
import logging
import numpy as np
import cv2
import random
from utils.utils_functions import _create_output_directories, _sync_directories, _get_valid_masks, \
    _calculate_required_numbers
from utils.image_utils import _load_image, _resize_image_and_annotations, _apply_flip_augmentation, \
    _apply_rotation_augmentation, _apply_crop_augmentation, _apply_stretch_augmentation, _save_image_jpg, \
    _resize_image_with_padding_normal, _apply_random_augmentation_normal
from utils.annotation_utils import _extract_annotations_from_mask_image, _parse_xml, _check_objects_size, \
    _save_yolo_annotations


def _save_image_and_annotation(
        image: np.ndarray,
        annotations: list,
        train_numbers: int,
        val_numbers: int,
        train_dirs: dict,
        val_dirs: dict,
        train_or_val: str
) -> tuple[int, int]:
    """Save an image and its YOLO annotations to the appropriate training or validation directory."""
    try:
        if train_or_val not in {'train', 'val', 'any'}:
            raise ValueError("train_or_val must be 'train', 'val', or 'any'.")

        if train_or_val == 'train' or (train_or_val == 'any' and train_numbers > 0):
            train_index = len(os.listdir(train_dirs['images'])) + 1

            image_path_out = os.path.join(train_dirs['images'], f"{train_index}.jpg")
            label_path_out = os.path.join(train_dirs['labels'], f"{train_index}.txt")

            _save_image_jpg(image_path_out, image)
            _save_yolo_annotations(label_path_out, annotations)

            train_numbers -= 1

        elif train_or_val == 'val' or (train_or_val == 'any' and val_numbers > 0):
            val_index = len(os.listdir(val_dirs['images'])) + 1

            image_path_out = os.path.join(val_dirs['images'], f"{val_index}.jpg")
            label_path_out = os.path.join(val_dirs['labels'], f"{val_index}.txt")

            _save_image_jpg(image_path_out, image)
            _save_yolo_annotations(label_path_out, annotations)

            val_numbers -= 1

        return train_numbers, val_numbers

    except Exception as e:
        logging.error(f"Error in _save_image_and_annotation: {str(e)}", exc_info=True)
        return train_numbers, val_numbers


def _apply_augmentation(
        image: np.ndarray,
        annotations: list[tuple[float, float, float, float]],
        num_of_augmentations: int,
        train_numbers: int,
        val_numbers: int,
        train_dirs: dict,
        val_dirs: dict,
        split_augmentation_ratio: int
) -> tuple[int, int]:
    """
    Apply random augmentations (flip, rotate, crop, stretch) to an image and its annotations.
    """
    if image is None:
        logging.error("Input image is None.")
        return train_numbers, val_numbers

    if not annotations:
        logging.warning("No annotations provided. Skipping augmentation.")
        return train_numbers, val_numbers

    for _ in range(num_of_augmentations):
        try:
            aug_image, aug_annotations = _apply_flip_augmentation(image, annotations)
            aug_image, aug_annotations = _apply_rotation_augmentation(aug_image, aug_annotations)
            aug_image, aug_annotations = _apply_crop_augmentation(aug_image, aug_annotations)
            aug_image, aug_annotations = _apply_stretch_augmentation(aug_image, aug_annotations)

            if _ < split_augmentation_ratio:
                train_numbers, val_numbers = _save_image_and_annotation(aug_image, aug_annotations, train_numbers,
                                                                        val_numbers, train_dirs, val_dirs, 'train')
            else:
                train_numbers, val_numbers = _save_image_and_annotation(aug_image, aug_annotations, train_numbers,
                                                                        val_numbers, train_dirs, val_dirs, 'val')

        except Exception as e:
            logging.error(f"Error during augmentation process: {e}")

    return train_numbers, val_numbers


def process_dataset(
        image_directory: str,
        mask_directory: str,
        output_root: str = "data/dataset",
        min_object_size: int = 30,
        num_of_augmentations: int = 5,
        split_ratio: float = 0.8,
        resize_shape: int = 640,
        need_augmentation: bool = True
):
    """Process masks directory and check if the mask is valid and if it needs augmentation or not and resize it and
        split it and save as YOLO annotation."""
    try:
        if not os.path.exists(image_directory):
            raise FileNotFoundError(f"Images directory not found: {image_directory}")
        if not os.path.exists(mask_directory):
            raise FileNotFoundError(f"Masks directory not found: {mask_directory}")

        if not (0 < split_ratio < 1):
            raise ValueError("split_ratio must be between 0 and 1. eg: 0.8")

        _sync_directories(image_directory, mask_directory)
        train_dirs, val_dirs = _create_output_directories(output_root)

        print("\nChecking Valid Data...\n")
        masks_paths = _get_valid_masks(mask_directory)

        print(f"\nChecking Data Ended\nNumber of Valid Data: {len(masks_paths)}\n")

        images_list = os.listdir(image_directory)
        if not images_list:
            raise FileNotFoundError("No images found in the dataset directory.")
        images_extension = images_list[0].split('.')[-1]

        num_of_masks = len(masks_paths)
        train_numbers = int(num_of_masks * split_ratio)
        val_numbers = num_of_masks - train_numbers

        from tqdm import tqdm
        progress_bar = tqdm(total=num_of_masks, desc="Processing Images", unit="img")

        for mask_path in masks_paths:
            try:
                file_name = os.path.basename(mask_path)
                base_name, masks_extension = file_name.split('.')
                image_path = os.path.join(image_directory, f"{base_name}.{images_extension}")

                if masks_extension == 'xml':
                    annotations = _parse_xml(mask_path)
                else:
                    annotations = _extract_annotations_from_mask_image(mask_path)

                image = _load_image(image_path)
                image, annotations = _resize_image_and_annotations(image, annotations, resize_shape)

                image_height, image_width = image.shape[:2]

                if (annotations and _check_objects_size(annotations, min_object_size, image_width, image_height)
                        and need_augmentation):
                    split_augmentation_ratio = int(num_of_augmentations * split_ratio)
                    added_val = num_of_augmentations - split_augmentation_ratio
                    train_numbers += split_augmentation_ratio
                    val_numbers += added_val

                    train_numbers, val_numbers = _apply_augmentation(
                        image, annotations, num_of_augmentations,
                        train_numbers, val_numbers, train_dirs,
                        val_dirs, split_augmentation_ratio
                    )

                train_numbers, val_numbers = _save_image_and_annotation(
                    image, annotations, train_numbers, val_numbers, train_dirs, val_dirs, 'any'
                )
                progress_bar.update(1)

            except Exception as img_error:
                logging.error(f"Error processing image {mask_path}: {str(img_error)}", exc_info=True)

        progress_bar.close()
        print(f"\nTotal images in train: {len(os.listdir(train_dirs['images']))}\n"
              f"Total images in val: {len(os.listdir(val_dirs['images']))}\n")

    except Exception as e:
        logging.error(f"Critical error in process_dataset: {str(e)}", exc_info=True)


def process_dataset_normal(
    image_directory: str,
    output_root: str = "data/dataset",
    resize_shape: int = 640,
    normal_ratio: float = 0.1
):
    """
    Process a dataset by loading, augmenting (if needed), resizing, and saving images to train/val directories
    (for normal data without annotations).
    """
    try:
        train_dirs, val_dirs = _create_output_directories(output_root)

        print("\nCalculating required numbers of images...")
        train_numbers, val_numbers = _calculate_required_numbers(train_dirs, val_dirs, normal_ratio)
        total_required = train_numbers + val_numbers
        print(f"\nRequired image count calculated: {train_numbers} for train,"
              f" {val_numbers} for val (Total: {total_required}).")

        available_image_paths = [os.path.join(image_directory, img) for img in os.listdir(image_directory)
                                 if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        num_available = len(available_image_paths)
        print(f"\nFound {num_available} images in '{image_directory}'.")

        if num_available == 0:
            print("No images found in the directory. Exiting.")
            return

        original_image_paths = available_image_paths.copy()
        random.shuffle(original_image_paths)

        assignments = ['train'] * train_numbers + ['val'] * val_numbers
        random.shuffle(assignments)

        train_counter = len(os.listdir(train_dirs['images'])) + 1
        val_counter = len(os.listdir(val_dirs['images'])) + 1
        aug_counter = num_available + 1

        from tqdm import tqdm
        with tqdm(total=total_required, desc="Processing and saving images", unit="img") as pbar:
            for assignment in assignments:
                if original_image_paths:
                    image_path = original_image_paths.pop(0)
                    image = _load_image(image_path)
                else:
                    random_image_path = random.choice(available_image_paths)
                    image = _load_image(random_image_path)
                    if image is not None:
                        image = _apply_random_augmentation_normal(image)
                        aug_image_path = os.path.join(image_directory, f"aug_{aug_counter}.jpg")
                        cv2.imwrite(aug_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        aug_counter += 1
                        image = _load_image(aug_image_path)

                if image is not None:
                    processed_image = _resize_image_with_padding_normal(image, resize_shape)
                    if processed_image is not None:
                        if assignment == 'train':
                            save_image_path = os.path.join(train_dirs['images'], f"{train_counter}.jpg")
                            save_label_path = os.path.join(train_dirs['labels'], f"{train_counter}.txt")
                            train_counter += 1
                        elif assignment == 'val':
                            save_image_path = os.path.join(val_dirs['images'], f"{val_counter}.jpg")
                            save_label_path = os.path.join(val_dirs['labels'], f"{val_counter}.txt")
                            val_counter += 1
                        cv2.imwrite(save_image_path, processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        with open(save_label_path, 'w'):
                            pass  # Empty file
                pbar.update(1)

        train_image_count = len(os.listdir(train_dirs['images']))
        val_image_count = len(os.listdir(val_dirs['images']))
        print(f"\nTotal images in train: {train_image_count}")
        print(f"Total images in val: {val_image_count}\n")

    except Exception as e:
        logging.error(f"Error in process_dataset_normal: {str(e)}")
        raise
