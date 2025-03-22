import shutil
import os
from utils.dataset_processor import process_dataset_normal
import zipfile
from zenml import step


@step
def clean_normal_dataset():
    if not os.path.exists("data/normal"):
        print("Extracting zip file normal.zip ....")
        with zipfile.ZipFile("data/normal.zip", 'r') as zip_ref:
            zip_ref.extractall("data/normal")

    images_dir = "data/normal/normal"

    process_dataset_normal(
        image_directory=images_dir,
    )

    shutil.rmtree(r"data/normal")
