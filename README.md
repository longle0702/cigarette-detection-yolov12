# 🚬 Cigarette Detection – YOLOv12

This project is an upgrade of my original computer vision assignment, using **YOLOv12** for real-time cigarette detection. It includes a **FastAPI inference REST API** and a **browser-based live demo UI** for easy interaction and integration.

## 📁 Project Structure

```
cigarette-detection-yolov12/
│
├── src/                   # Source code
│   ├── api.py             # FastAPI inference API
│   ├── schemas.py         # Pydantic request/response models
│   ├── main.py            # Standalone inference script
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   └── static/
│       └── index.html     # Browser-based live demo UI
│
├── models/                # Trained models
│   └── best.pt            # Best performing model
│
├── runs/                  # Training and detection runs
│   ├── detect/            # Detection outputs
│   └── mlflow/            # MLflow tracking data
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── image.png              # Sample image
└── README.md              # This file
```

## 🖥️ Live Demo UI

A fully interactive browser-based demo is available at **`http://localhost:8000`**.

**Features:**

- 🖼️ Drag & drop image upload with instant preview
- 🎛️ Live sliders for Confidence, IoU, and Image Size parameters
- 🔀 Toggle to include/exclude annotated bounding-box image in response
- 📊 Stats bar showing detection count, inference time, and image dimensions
- 🖼️ Annotated result image with bounding boxes drawn
- 📋 Detections table with label, confidence bar, and bbox coordinates

## 🌐 FastAPI Inference API

### Endpoints

| Method | Path          | Description                         |
| ------ | ------------- | ----------------------------------- |
| `GET`  | `/`           | Browser live demo UI                |
| `GET`  | `/health`     | Liveness probe – model readiness    |
| `GET`  | `/model/info` | Model metadata (task, class names)  |
| `POST` | `/detect`     | Run cigarette detection on an image |
| `GET`  | `/docs`       | Interactive Swagger UI              |
| `GET`  | `/redoc`      | ReDoc API documentation             |

### `/detect` Parameters

| Field          | Type  | Default | Description                                |
| -------------- | ----- | ------- | ------------------------------------------ |
| `file`         | file  | —       | Image (JPEG / PNG / WebP / BMP, ≤ 20 MB)   |
| `conf`         | float | `0.25`  | Confidence threshold (0.01 – 1.0)          |
| `iou`          | float | `0.45`  | NMS IoU threshold (0.01 – 1.0)             |
| `imgsz`        | int   | `640`   | Inference image size in pixels (32 – 1920) |
| `return_image` | bool  | `false` | Return annotated image as base64 PNG       |

### Example Response

```json
{
  "filename": "photo.jpg",
  "image_width": 1280,
  "image_height": 720,
  "num_detections": 2,
  "detections": [
    {
      "label": "cigarette",
      "class_id": 0,
      "confidence": 0.8731,
      "bbox_xyxy": [312.4, 201.1, 478.9, 389.6],
      "bbox_xywhn": [0.3091, 0.4101, 0.1301, 0.2618]
    }
  ],
  "inference_ms": 9.42,
  "annotated_image_b64": null
}
```

### Quick Test with `curl`

```bash
# Basic detection
curl -X POST http://localhost:8000/detect \
  -F "file=@image.png" \
  -F "conf=0.25"

# With annotated image in response
curl -X POST http://localhost:8000/detect \
  -F "file=@image.png" \
  -F "conf=0.25" \
  -F "return_image=true"
```

## 🐳 Docker Setup

### Prerequisites

- Docker
- Docker Compose

### Running the API

```bash
# Build and start the server on port 8000
docker compose up --build
```

Once running, the following URLs are available:

| URL                                | Description            |
| ---------------------------------- | ---------------------- |
| `http://localhost:8000`            | 🖥️ Live demo UI        |
| `http://localhost:8000/docs`       | 📄 Swagger UI          |
| `http://localhost:8000/redoc`      | 📄 ReDoc documentation |
| `http://localhost:8000/health`     | ❤️ Health check        |
| `http://localhost:8000/model/info` | ℹ️ Model info          |

### Docker Configuration

- **Base Image:** `ultralytics/ultralytics:latest-cpu`
- **Platform:** `linux/amd64`
- **Port:** `8000`
- **Working Directory:** `/usr/src/app`
- **Health Check:** `GET /health` every 30 s

## 🚀 Local Development (without Docker)

```bash
pip install -r requirements.txt

# Run the API
PYTHONPATH=. uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Performance

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

## ❤️ Acknowledgements
I would like to thank my former teammates who contributed to my earlier Computer Vision projects during my Bachelor's journey. Your collaboration, discussions, and support helped lay the foundation for this work.
