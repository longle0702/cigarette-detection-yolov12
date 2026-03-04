import io
import os
import time
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
from ultralytics import YOLO
from src.schemas import (
    BoundingBox,
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cigarette-api")

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
model_path = os.path.join(root_dir, "models", "best.pt")
static_dir = os.path.join(base_dir, "static")

_model: Optional[YOLO] = None

def get_model():
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Try again shortly.",
        )
    return _model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Loading YOLOv12 model from %s …", model_path)
    _model = YOLO(model_path)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down API.")
    _model = None

app = FastAPI(
    title="🚬 Cigarette Detection API",
    description=(
        "REST API for real-time cigarette detection using YOLOv12.\n\n"
        "Upload an image and receive bounding-box predictions with confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _load_image(raw):
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the uploaded image. Ensure it is a valid image file.",
        )
    return img

def _encode_image_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def model_info():
    model = get_model()
    return ModelInfoResponse(
        model_path=model_path,
        task=model.task,
        names=model.names,
    )

@app.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Run cigarette detection on an uploaded image",
)
async def detect(
    file: UploadFile = File(..., description="Image file (JPEG / PNG / WebP / BMP)"),
    conf: float = Form(default=0.25, ge=0.01, le=1.0, description="Confidence threshold (0.01 – 1.0)"),
    iou: float = Form(default=0.45, ge=0.01, le=1.0, description="IoU threshold for NMS (0.01 – 1.0)"),
    imgsz: int = Form(default=640, ge=32, le=1920, description="Inference image size (pixels)"),
    return_image: bool = Form(default=False, description="Include annotated image (base64 PNG) in response"),
):

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(raw) / 1024 / 1024:.1f} MB). Maximum allowed: 20 MB.",
        )

    img_bgr = _load_image(raw)
    h, w = img_bgr.shape[:2]

    model = get_model()
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    detections: list[BoundingBox] = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        xywhn = box.xywhn[0].tolist()
        class_id = int(box.cls[0])
        label = model.names.get(class_id, str(class_id))
        confidence = float(box.conf[0])
        detections.append(
            BoundingBox(
                label=label,
                class_id=class_id,
                confidence=round(confidence, 4),
                bbox_xyxy=[round(v, 2) for v in xyxy],
                bbox_xywhn=[round(v, 6) for v in xywhn],
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    annotated_b64: Optional[str] = None
    if return_image:
        annotated = result.plot()         
        annotated_b64 = _encode_image_b64(annotated)

    logger.info(
        "Detected %d cigarette(s) in %.1f ms | conf=%.2f iou=%.2f imgsz=%d | file=%s",
        len(detections),
        inference_ms,
        conf,
        iou,
        imgsz,
        file.filename,
    )

    return DetectionResponse(
        filename=file.filename,
        image_width=w,
        image_height=h,
        num_detections=len(detections),
        detections=detections,
        inference_ms=round(inference_ms, 2),
        annotated_image_b64=annotated_b64,
    )
