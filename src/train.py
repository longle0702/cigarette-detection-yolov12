from ultralytics import YOLO

model = YOLO("models/yolo12s.pt")

results = model.train(data="data/data.yaml", 
                      epochs=500, 
                      imgsz=640, 
                      patience = 100,
                      plots=True,
                      cos_lr=True
                     )

model.save('models/cigarv12.pt')