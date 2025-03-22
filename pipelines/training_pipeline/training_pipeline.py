from zenml import pipeline
from steps.clean_larib_polyp_dataset.clean_larib_polyp_dataset import clean_larib_polyp_dataset
from steps.clean_Dataset_dataset.clean_dataset_dataset import clean_dataset_dataset
from steps.clean_neopolyp_dataset.clean_neopolyp_dataset import clean_neopolyp_dataset
from steps.clean_polyp_gen_dataset.clean_polyp_gen_dataset import clean_polyp_gen_dataset
from steps.clean_polyps_set_dataset.clean_polyps_set_dataset import clean_polyps_set_dataset
from steps.clean_normal_dataset.clean_normal_dataset import clean_normal_dataset
from steps.create_yaml_file.create_yaml_file import create_yaml_file
from steps.train_model.train_model import train_model


@pipeline()
def training_pipeline():
    # Cleaning datasets and merging in one directory
    clean_larib_polyp_dataset()
    clean_dataset_dataset()
    clean_neopolyp_dataset()
    clean_polyp_gen_dataset()
    clean_polyps_set_dataset()
    clean_normal_dataset()

    # Create YAML file for the dataset
    create_yaml_file()

    # Start training the model
    train_model()
