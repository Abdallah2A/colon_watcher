import os
import shutil
from utils.dataset_processor import process_dataset
import patoolib
from zenml import step


@step
def clean_larib_polyp_dataset():
    if not os.path.exists("data/ETIS-LaribPolypDB"):
        print("Extracting zip file ETIS-LaribPolypDB.rar ....")
        patoolib.extract_archive("data/ETIS-LaribPolypDB.rar", outdir="data/ETIS-LaribPolypDB")

    images_dir = "data/ETIS-LaribPolypDB/ETIS-LaribPolypDB/ETIS-LaribPolypDB"
    masks_dir = "data/ETIS-LaribPolypDB/ETIS-LaribPolypDB/Ground Truth"

    for mask in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask)
        os.rename(mask_path, os.path.join(masks_dir, mask.replace('p', '')))

    process_dataset(
        image_directory=images_dir,
        mask_directory=masks_dir,
    )

    shutil.rmtree("data/ETIS-LaribPolypDB")
