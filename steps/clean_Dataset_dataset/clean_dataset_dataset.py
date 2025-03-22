import shutil
import os
from utils.dataset_processor import process_dataset
import zipfile
from zenml import step


@step
def clean_dataset_dataset():
    if not os.path.exists("data/DataSet_"):
        print("Extracting zip file DataSet_.zip ....")
        with zipfile.ZipFile("data/DataSet_.zip", 'r') as zip_ref:
            zip_ref.extractall("data/DataSet_")

    images_dirs = ["data/DataSet_/CVC-300/images",
                   "data/DataSet_/CVC-ClinicDB/images",
                   "data/DataSet_/CVC-ColonDB/images",
                   "data/DataSet_/Kvasir/images"]

    masks_dirs = ["data/DataSet_/CVC-300/masks",
                  "data/DataSet_/CVC-ClinicDB/masks",
                  "data/DataSet_/CVC-ColonDB/masks",
                  "data/DataSet_/Kvasir/masks"]

    for i in range(len(images_dirs)):
        process_dataset(
                image_directory=images_dirs[i],
                mask_directory=masks_dirs[i],
        )

    shutil.rmtree("data/DataSet_")
