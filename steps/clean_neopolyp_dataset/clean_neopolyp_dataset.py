import shutil
import os
from utils.dataset_processor import process_dataset
import zipfile
from zenml import step


@step
def clean_neopolyp_dataset():
    if not os.path.exists("data/bkai-igh-neopolyp"):
        print("Extracting zip file bkai-igh-neopolyp.zip ....")
        with zipfile.ZipFile("data/bkai-igh-neopolyp.zip", 'r') as zip_ref:
            zip_ref.extractall("data/bkai-igh-neopolyp")

    images_dir = "data/bkai-igh-neopolyp/train/train"
    masks_dir = "data/bkai-igh-neopolyp/train_gt/train_gt"

    process_dataset(
        image_directory=images_dir,
        mask_directory=masks_dir,
    )

    shutil.rmtree(r"data/bkai-igh-neopolyp")
