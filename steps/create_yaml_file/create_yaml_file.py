import os.path

from zenml import step


@step
def create_yaml_file():
    yaml_content = """
path: ./
train: train/images
val: val/images

names:
  0: polyp
    """
    if not os.path.exists("data/dataset/dataset.yaml"):
        with open("data/dataset/dataset.yaml", "w") as file:
            file.write(yaml_content)
