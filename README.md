
# Marine Boat Detection
This repository has 3 objectives:
- Train YOLOv8 models for marine ship detection.
- Calibrate the camera that will be mounted on the drone (obtain distortion coefficients and intrinsic data).
- Evaluate the trained model with a benchmark.

## Object detector
Models from the YOLOv8 family (n, s, m, l) by Ultralytics.

## Dataset
The dataset consists of 16k images from 2 countries (Finland and Singapore) and follows the YOLO format. The dataset needs to be more robust, and further augmentations will be applied in future iterations.

## Requirements
Install all Python dependencies using the following command:
```bash
pip install -r requirements.txt
```
## Run benchmark
- `videos/` is the directory to placed any video for benchmark.  

Go to the model directory
```bash
cd model
```

If you want to test a particular set of weights or a video, just provide the apropriate path and modify the following snippet in `benchmark.py`

```python
def main():
    # Select model to benchmark
    weights = os.getcwd() + "/runs/detect" + "/train4/weights/best.pt"
    # Select video to benchmark
    video = str(Path(os.getcwd()).parent) + "/videos/" + "sail_amsterdam.mp4"
``` 

Then just run the script.
```bash
python benchmark.py
```

## Run camera calibration
Go to the camera directory

```bash
cd camera
```

- `pictures/` is the directory that contains the necessary chessboard images for calibration.
- `details/` is the directory where the camera's intrinsic characteristics and distortion coefficients are saved after calibration.

First, you will need to provide a set of chessboard images (a minimum of 10 images) captured by the camera you want to calibrate. These images should be placed in the `pictures/` directory.

To run the calibration script, you will need to specify the number of internal corners of the chessboard pattern.
```bash
python calibrate.py -c 7 7
```
The output files will be saved in `details` and can be later copy to the [periscope package](https://github.com/ocortina/ros_periscope).



## Run model training
Go to the model directory

```bash
cd model
```

- `runs/detect/` is the directory where the data (model weights, plots, graphs, etc.) of each training run will be saved.

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a given task.

If you want to make any changes, please refer to the list of [arguments](https://docs.ultralytics.com/modes/train/#arguments) and modify the following line in `train.py`.

```python
# Train the model
model.train(device=0, data=dataset, epochs=100, imgsz=640, plots=True)
```
Run `train.py`
    
```bash
python train.py
```

After training, the model weights are located inside `/runs/detect/train/weights` you can then copy the file **.pt** to the [periscope package](https://github.com/ocortina/ros_periscope).  

**NOTE:** If you make a second run for training, the new weights will be located in `runs/detect/train2/weights`.
