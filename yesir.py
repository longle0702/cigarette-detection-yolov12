import yaml
from pathlib import Path

# Load the YAML file
yaml_path = "/home/long/longdata/cigarette-detection-using-yolov12-and-sahi/data/data.yaml"
with open(yaml_path) as f:
    data = yaml.safe_load(f)

# Function to count image files in a folder
def count_images(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    return len(list(folder.rglob("*.jpg"))) + len(list(folder.rglob("*.png")))

# Count images in each split
train_count = count_images(data['train'])
val_count = count_images(data['val'])
test_count = count_images(data.get('test', []))  # optional

print(f"Train images: {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")
