from ultralytics import YOLO

model = YOLO("../cigarette-detection-using-yolov12-and-sahi/models/yolo12s.pt")

results = model.train(data="../cigarette-detection-using-yolov12-and-sahi/data/data.yaml", 
                      epochs=100, 
                      imgsz=640, 
                      patience = 10,
                      plots=True,
                      degrees = 15.0, 
                      flipud=0.5, 
                      fliplr=0.5, 
                      shear = 15.0, 
                      translate = 0.2,
                      device="mps"
                     )

model.save('cigarv12.pt')
model.to("cpu").export(format="onnx")