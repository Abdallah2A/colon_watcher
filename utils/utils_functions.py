import os
import random
import logging
from tqdm import tqdm

logging.basicConfig(
    filename="clean_dataset.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)


def _create_output_directories(output_root: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Create output directories for training and validation datasets.

    :param output_root: Root directory where train/val folders will be created.
    :return: Two dictionaries containing paths for training and validation image/label directories.
    """
    train_dirs = {
        'images': os.path.join(output_root, 'train', 'images'),
        'labels': os.path.join(output_root, 'train', 'labels')
    }
    val_dirs = {
        'images': os.path.join(output_root, 'val', 'images'),
        'labels': os.path.join(output_root, 'val', 'labels')
    }

    try:
        for dirs in (train_dirs, val_dirs):
            for key, path in dirs.items():
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    logging.info(f"Created directory: {key} -> {path}")
    except Exception as e:
        logging.error(f"Error creating directories: {e}")
        raise

    return train_dirs, val_dirs


def _sync_directories(images_dir_path: str, masks_dir_path: str):
    """
    Syncs two directories by deleting files that do not have a matching base name in the other directory.

    :param images_dir_path: Path to the directory containing images.
    :param masks_dir_path: Path to the directory containing segmentation masks or annotations.
    """

    try:
        if not os.path.isdir(images_dir_path):
            raise FileNotFoundError(f"Images directory not found: {images_dir_path}")
        if not os.path.isdir(masks_dir_path):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir_path}")

        files1 = {os.path.splitext(f)[0] for f in os.listdir(images_dir_path)
                  if os.path.isfile(os.path.join(images_dir_path, f))}
        files2 = {os.path.splitext(f)[0] for f in os.listdir(masks_dir_path)
                  if os.path.isfile(os.path.join(masks_dir_path, f))}

        files_to_remove_dir1 = [f for f in os.listdir(images_dir_path) if os.path.splitext(f)[0] not in files2]
        files_to_remove_dir2 = [f for f in os.listdir(masks_dir_path) if os.path.splitext(f)[0] not in files1]

        for file in files_to_remove_dir1:
            file_path = os.path.join(images_dir_path, file)
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Permission denied: Unable to delete {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        for file in files_to_remove_dir2:
            file_path = os.path.join(masks_dir_path, file)
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Permission denied: Unable to delete {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")


def _get_valid_masks(masks_dir_path: str) -> list:
    """Get valid masks paths (that have objects inside it).
    :param masks_dir_path: path directory of masks.
    :return: List of all valid masks.
    """
    from utils.annotation_utils import _check_valid_xml, _check_black_masks

    masks = []
    progress_bar = tqdm(total=len(os.listdir(masks_dir_path)), desc="Checking Valid Data", unit="img")

    for mask in os.listdir(masks_dir_path):
        masks_extension = mask.split('.')[-1]
        mask_path = os.path.join(masks_dir_path, mask)

        if masks_extension == 'xml':
            check = _check_valid_xml(mask_path)
        elif masks_extension in ['jpg', 'jpeg', 'png', 'tif']:
            check = _check_black_masks(mask_path)
        else:
            continue

        if not check:
            masks.append(mask_path)

        progress_bar.update(1)

    random.shuffle(masks)
    progress_bar.close()

    return masks


def _calculate_required_numbers(train_dirs: dict, val_dirs: dict, normal_ratio: float) -> tuple[int, int]:
    """
    Calculate the required number of images for training and validation (for normal data).

    :param train_dirs: Dictionary with paths to training image and label directories.
    :param val_dirs: Dictionary with paths to validation image and label directories.
    :param normal_ratio: Ratio to determine the number of normal images.
    :return: Tuple of (train_numbers, val_numbers) representing required image counts.
    """
    size_of_train_data = len(os.listdir(train_dirs['images']))
    size_of_val_data = len(os.listdir(val_dirs['images']))
    train_numbers = int(size_of_train_data * normal_ratio)
    val_numbers = int(size_of_val_data * normal_ratio)
    return train_numbers, val_numbers
