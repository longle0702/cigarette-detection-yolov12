# 🚬 Cigarette Detection – YOLOv12

This project is an upgrade of my original computer vision assignment. Now powered by **YOLOv12**, it delivers real-time cigarette detection with smooth performance and strong accuracy.

## 📁 Project Structure

```
cigarette-detection-yolov12/
│
├── data/          
├── models/        
├── runs/          
└── src/           
```

## 🚀 Performance

| Metric                        | Value          |
| ----------------------------- | -------------- |
| **Average Inference Latency** | **8.49 ms**    |
| **Inference-Only FPS**        | **117.82 FPS** |
| **End-to-End FPS**            | **94.97 FPS**  |

## 🧪 Validation Results

```
Speed: 0.8ms preprocess, 9.2ms inference, 4.2ms postprocess per image
mAP50: 0.871
mAP50-95: 0.567
Precision: 0.917
Recall: 0.778
```