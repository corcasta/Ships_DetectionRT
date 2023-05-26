
# Marine Boat Detection
Este repo tiene 3 objetivos:
 - Entrenar un modelo de deteccion de barcos marinos.
 - Calcular las características intrinsicas de una camara.
 - Evaluar el modelo entrenado con un benchmark. 

## Dataset
El dataset esta compuesto por 16k imagenes de 2 paises (Finlandia y Singapore) y sigue el formato de YOLO.

## Camera
- `pictures`Es el folder que contiene las imagenes de chessboard necesarias para hacer la calibracion.

- `details` Es el folder donde se guarda la caracteristicas intrinsicas de la camara y los coeficientes de distorcion despues de la calibracion.

## Object detector
The object detection model used YOLOv8 from ultralytics. 
## Run camera calibration

Clone the project

Go to the project directory

```bash
  cd camera
```

You will need to provide the number of internal corners of the chessboard images.

```bash
  python calibrate.py -c 7 7
```
The output will be saved in `details` and can be later copy in to the periscope package.
## Run model training

Install my-project with npm

    
```bash
  python train.py
```

After training, the model weights are located inside `/runs/detect/train/weights` you can then copy the file **.pt** for any other project.
