import warnings
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
                project='Result Save Location',
                name='Saved Result Filename',
                )
