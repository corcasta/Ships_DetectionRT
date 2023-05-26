import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load pretrained model 
    model = YOLO("yolov8l.pt")
    
    # Load dataset
    dataset = os.path.join(Path(os.getcwd()).parent, "dataset", "data.yaml")
    
    # Train the model
    model.train(device=0, data=dataset, epochs=100, imgsz=640, plots=True)


if __name__ == "__main__":
    # Setting the starting path of the script
    script_path = os.path.realpath(__file__)
    os.chdir(Path(script_path).parent)
    
    main()
    