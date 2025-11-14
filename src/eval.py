from ultralytics import YOLO
model = YOLO('/home/long/longdata/cigarette-detection-using-yolov12-and-sahi/cigarv12.pt')
model.val(data='/home/long/longdata/cigarette-detection-using-yolov12-and-sahi/data/data.yaml')