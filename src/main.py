from ultralytics import YOLO
import os

base_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(base_path) 

model_path = os.path.join(root, 'models', 'best.pt')
image = os.path.join(root, 'image.png')

model = YOLO(model_path)
results = model.predict(source=image, save=True, project=os.path.join(root, 'runs'), name='detect')
print(f"Success!")