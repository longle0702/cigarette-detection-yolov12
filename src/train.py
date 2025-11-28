from ultralytics import YOLO

model = YOLO("models/yolo12s.pt")

results = model.train(
    data="data/data.yaml",
    epochs=500,
    imgsz=960,
    batch=8,                    
    patience=100,
    cos_lr=True,
    plots=True
)

model.save('models/cigarv12.pt')