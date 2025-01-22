# SAMVD-YOLO
Scale-adaptive method for road vehicle detection based on an improved YOLOv8.
# Title
### Precision Multidimensional Vehicle Information Perception via Video Data Analytics
# Submission to the Journal
### The Visual Computer
# Code
you can get the official YOLOv8 code at https://github.com/ultralytics/ultralytics
# Environment
```conda create -n SAMVD-YOLO python=3.9
conda activate SAMVD-YOLO
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# Train
```import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/SAMVD/SAMVD-YOLO.yaml')
    model.train(data=r'./dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                amp=True,  
                project='Run/train',
                name='SAMVD-YOLO',
                )
```
