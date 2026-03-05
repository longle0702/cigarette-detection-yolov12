# 🚬 Cigarette Detection – YOLOv12

This project is an upgrade of my original computer vision assignment, using **YOLOv12** for real-time cigarette detection. It features a **Streamlit web app** for easy interaction: upload an image, tune detection parameters, and view annotated results instantly in your browser.

## 📁 Project Structure

```
cigarette-detection-yolov12/
│
├── src/                   # Source code
│   ├── app.py             # Streamlit web app
│   ├── main.py            # Standalone inference script
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
│
├── models/                # Trained models
│   └── best.pt            # Best performing model weights
│
├── runs/                  # Training and detection output runs
│
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── image.png              # Sample image
└── README.md              # This file
```

## 🖥️ Streamlit Web App

A fully interactive web demo is available at **`http://localhost:8501`** after starting the app.

**Features:**

- 🖼️ Drag & drop image upload (JPEG / PNG / WebP / BMP, up to 20 MB)
- 🎛️ Sidebar sliders for Confidence threshold, IoU threshold, and Inference image size
- 📊 At-a-glance metrics — detection count, inference time, image dimensions, top confidence
- 🖼️ Side-by-side view of the original and annotated image (toggle original on/off)
- 📋 Per-detection cards with label, confidence, and bounding box coordinates
- ⬇️ Download the annotated image as PNG

## 🐳 Running with Docker (recommended)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Start the app

```bash
docker compose up --build
```

Then open **http://localhost:8501** in your browser.

### URL reference

| URL                            | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `http://localhost:8501`        | 🖥️ Streamlit web app                                     |
| `http://172.x.x.x:8501`        | 📡 Docker internal network (container-to-container only) |
| `http://<your-public-ip>:8501` | 🌍 External access (requires firewall/router port open)  |

### Docker configuration

| Setting          | Value                                |
| ---------------- | ------------------------------------ |
| **Base image**   | `ultralytics/ultralytics:latest-cpu` |
| **Platform**     | `linux/amd64`                        |
| **Exposed port** | `8501`                               |
| **Health check** | `GET /_stcore/health` every 30 s     |

## 🚀 Running Locally (without Docker)

```bash
# Install dependencies into your virtual environment
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run src/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

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
