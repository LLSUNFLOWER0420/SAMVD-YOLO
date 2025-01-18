import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/SAMVD/SAMVD-YOLO.yaml')
    model.val(data=r'./data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              project='Result Save Location',
              name='Saved Result Filename',
              )

