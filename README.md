
# Marine Boat Detection
Este repo tiene 3 objetivos:
 - Entrenar un modelo de deteccion de barcos marinos.
 - Calcular las características intrinsicas de una camara.
 - Evaluar el modelo entrenado con un benchmark. 

## Dataset
El dataset esta compuesto por 16k imagenes de 2 paises (Finlandia y Singapore) y sigue el formato de YOLO.

## Camera
- `pictures` Es el folder que contiene las imagenes de chessboard necesarias para hacer la calibracion.

- `details` Es el folder donde se guarda la caracteristicas intrinsicas de la camara y los coeficientes de distorcion despues de la calibracion.

## Object detector
The object detection model used YOLOv8 from ultralytics. 

## Run camera calibration
Go to the camera directory

```bash
cd camera
```

First you will need to provide a set of chess board images (minimum 10 images) taken by the camera you are looking to calibrate. They should be placed in `pictures`.  

To run the calibration script you will need to provide the number of internal corners of the chessboard pattern.

```bash
python calibrate.py -c 7 7
```
The output files will be saved in `details` and can be later copy to the periscope package.

## Run model training
Go to the model directory

```bash
cd model
```

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a given task.

Si se desea hacer algun cambio favor de consultar la lista de [argumentos](https://docs.ultralytics.com/modes/train/#arguments) y modificar en `train.py` esta linea:

```python
# Train the model
model.train(device=0, data=dataset, epochs=100, imgsz=640, plots=True)
```
Finalmente corre `train.py`
    
```bash
  python train.py
```

After training, the model weights are located inside `/runs/detect/train/weights` you can then copy the file **.pt** to the periscope package.


## Run benchmark
Go to the model directory
```bash
cd model
```

If you want to test a particular set of weights or a video, just provide the apropriate path and modify the following snippet in `benchmark.py`

```python
def main():
    # Select model to benchmark
    weights = os.getcwd() + "/runs/detect" + "/train4/weights/best.pt"
    # SElect video to benchmark
    video = str(Path(os.getcwd()).parent) + "/videos/" + "sail_amsterdam.mp4"
``` 

Then just run the script
```bash
  python benchmark.py
```

