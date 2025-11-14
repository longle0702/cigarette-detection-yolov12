from ultralytics import YOLO

model = YOLO("../cigarette-detection-using-yolov12-and-sahi/models/yolo12m.pt")

results = model.train(data="/mnt/Data/Long-Data/cigarette-detection-using-yolov12-and-sahi/data/data.yaml", 
                      epochs=500, 
                      imgsz=640, 
                      patience = 100,
                      plots=True,
                      cos_lr=True
                     )

model.save('cigarv12.pt')
model.export(format="onnx")