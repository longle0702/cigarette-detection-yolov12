from ultralytics import YOLO
model = YOLO('/models/best.pt')

url = 'https://www.youtube.com/watch?v=nWHbvibBQjs'

results = model.predict(
    source = url,
    save = True
)

