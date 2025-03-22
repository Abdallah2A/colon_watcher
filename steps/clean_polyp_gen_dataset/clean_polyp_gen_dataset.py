import shutil
import os
from utils.dataset_processor import process_dataset
import re
import zipfile
from zenml import step


def rename_masks(input_masks_dir):
    for mask in os.listdir(input_masks_dir):
        mask_path = os.path.join(input_masks_dir, mask)
        os.rename(mask_path, os.path.join(input_masks_dir, mask.replace('_mask', '')))


def filter_images_by_number(input_images_dir, input_masks_dir):
    # List image files (ignoring case on extensions)
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Regular expression to extract numbers from filenames
    number_pattern = re.compile(r'\d+(?=\.(jpg|png|jpeg)$)', re.IGNORECASE)
    images_with_numbers = []

    for img in image_files:
        match = number_pattern.search(img)
        if match:
            num = int(match.group())
            images_with_numbers.append((img, num))

    # Sort images by the extracted number
    images_with_numbers.sort(key=lambda x: x[1])

    kept_images = []
    removed_images = []

    # Keep the first image and then only those that are at least 10 apart
    last_kept_number = None
    for img, num in images_with_numbers:
        if last_kept_number is None or (num - last_kept_number) >= 10:
            kept_images.append((img, num))
            last_kept_number = num
        else:
            removed_images.append(img)

    # Remove the images and their corresponding mask files.
    for img in removed_images:
        img_path = os.path.join(input_images_dir, img)
        if os.path.exists(img_path):
            os.remove(img_path)
        # Assume the mask file has the same base name with an .xml extension.
        mask_filename = os.path.splitext(img)[0] + '.xml'
        mask_path = os.path.join(input_masks_dir, mask_filename)
        if os.path.exists(mask_path):
            os.remove(mask_path)


@step
def clean_polyp_gen_dataset():
    if not os.path.exists("data/PolypGen2021_jpg"):
        print("Extracting zip file PolypGen2021_jpg.zip ....")
        with zipfile.ZipFile("data/PolypGen2021_jpg.zip", 'r') as zip_ref:
            zip_ref.extractall("data/PolypGen2021_jpg")

    os.rename("data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/data_C3/images_C3/C3_EndoCV2021_00489].jpg",
              "data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/data_C3/images_C3/C3_EndoCV2021_00489.jpg")

    for i in range(1, 7):
        images_dir = f"data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/data_C{i}/images_C{i}"
        masks_dir = f"data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/data_C{i}/masks_C{i}"
        rename_masks(masks_dir)
        process_dataset(
                image_directory=images_dir,
                mask_directory=masks_dir,
        )

    for i in range(1, 24):
        images_dir = (f"data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/sequenceData/positive/"
                      f"seq{i}/images_seq{i}")
        masks_dir = (f"data/PolypGen2021_jpg/PolypGen2021_MultiCenterData_v3/sequenceData/positive/"
                     f"seq{i}/masks_seq{i}")

        rename_masks(masks_dir)
        filter_images_by_number(images_dir, masks_dir)

        process_dataset(
                image_directory=images_dir,
                mask_directory=masks_dir,
        )

    shutil.rmtree("data/PolypGen2021_jpg")
