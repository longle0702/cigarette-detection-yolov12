from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.val(data='data/data.yaml', split='test', plots=True)

latency_ms = results.speed['inference']
print(f"Average Inference Latency: {latency_ms:.2f} ms")
total_time_ms = results.speed['preprocess'] + results.speed['inference'] + results.speed['postprocess']

total_time_s = total_time_ms / 1000.0 
fps = 1.0 / total_time_s
print(f"End-to-End FPS: {fps:.2f}")

inference_fps = 1000.0 / latency_ms
print(f"Inference-Only FPS: {inference_fps:.2f}")