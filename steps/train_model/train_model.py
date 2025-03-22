from ultralytics import YOLO
from zenml import step


@step
def train_model():
    model = YOLO('steps/train_model/yolov10b.pt')

    model.train(
        data='data/dataset/dataset.yaml',
        epochs=1,
        imgsz=640,
        project="saved_models",
        device=0,
        batch=4
    )


if __name__ == '__main__':
    train_model()
