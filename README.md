# 🚬 Cigarette Detection – YOLOv12

This project is an upgrade of my original computer vision assignment, using **YOLOv12** for real-time cigarette detection.

## 📁 Project Structure

```
cigarette-detection-yolov12/
│
├── src/                    # Source code
│   ├── main.py            # Main inference script
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
│
├── models/                 # Trained models
│   └── best.pt            # Best performing model
│
├── runs/                   # Training and detection runs
│   ├── detect/            # Detection outputs
│   └── mlflow/            # MLflow tracking data
│
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── image.png             # Sample image
└── README.md             # This file
```

## 🐳 Docker Setup

This project uses Docker for consistent development and deployment environments.

### Prerequisites
- Docker
- Docker Compose

### Building and Running

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Run inference:**
   ```bash
   docker-compose run app python src/main.py
   ```

### Docker Configuration
- **Base Image:** `ultralytics/ultralytics:latest-cpu`
- **Platform:** `linux/amd64`
- **Working Directory:** `/usr/src/app`
- **Volume Mounting:** Local directory is mapped to container for seamless development

## 🚀 Performance

| Metric                        | Value          |
| ----------------------------- | -------------- |
| **Average Inference Latency** | **9.38 ms**    |
| **Inference-Only FPS**        | **106.63 FPS** |
| **End-to-End FPS**            | **41.13 FPS**  |

## 🧪 Validation Results

| Metric        | Value     |
| ------------- | --------- |
| **mAP50**     | **0.59**  |
| **Precision** | **0.675** |
| **Recall**    | **0.614** |
