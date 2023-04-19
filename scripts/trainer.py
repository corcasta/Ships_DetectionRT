import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO(str(Path(os.getcwd()).parent) + "/model/yolov8n.pt")  # load a pretrained model (recommended for training)
    
    # Train the model
    dataset = os.path.join(os.getcwd(), "datasets", "data.yaml")
    model.train(data=dataset, epochs=5, imgsz=640, plots=True)


if __name__ == "__main__":
    main()