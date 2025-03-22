import shutil
from utils.dataset_processor import process_dataset
import os
import re
from zenml import step
import zipfile


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
def clean_polyps_set_dataset():
    if not os.path.exists("data/PolypsSet"):
        print("Extracting zip file PolypsSet.zip ....")
        with zipfile.ZipFile("data/PolypsSet.zip", 'r') as zip_ref:
            zip_ref.extractall("data/PolypsSet")

    images_dir = "data/PolypsSet/PolypsSet/train2019/Image"
    masks_dir = "data/PolypsSet/PolypsSet/train2019/Annotation"

    process_dataset(
            image_directory=images_dir,
            mask_directory=masks_dir,
    )

    images_dir = "data/PolypsSet/PolypsSet/test2019/Image"
    masks_dir = "data/PolypsSet/PolypsSet/test2019/Annotation"

    for image_dir in os.listdir(images_dir):
        input_images_dir = os.path.join(images_dir, image_dir)
        input_masks_dir = os.path.join(masks_dir, image_dir)
        filter_images_by_number(input_images_dir, input_masks_dir)
        process_dataset(
                image_directory=input_images_dir,
                mask_directory=input_masks_dir,
        )

    images_dir = "data/PolypsSet/PolypsSet/val2019/Image"
    masks_dir = "data/PolypsSet/PolypsSet/val2019/Annotation"

    for image_dir in os.listdir(images_dir):
        input_images_dir = os.path.join(images_dir, image_dir)
        input_masks_dir = os.path.join(masks_dir, image_dir)
        filter_images_by_number(input_images_dir, input_masks_dir)
        process_dataset(
                image_directory=input_images_dir,
                mask_directory=input_masks_dir,
        )

    shutil.rmtree("data/PolypsSet")
