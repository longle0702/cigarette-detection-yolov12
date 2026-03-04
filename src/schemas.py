from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    label: str = Field(..., description="Class label")
    class_id: int = Field(..., description="Numeric class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox_xyxy: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Absolute bounding box [x1, y1, x2, y2] in pixels",
    )
    bbox_xywhn: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalised bounding box [cx, cy, w, h] relative to image size",
    )

class DetectionResponse(BaseModel):
    filename: Optional[str] = Field(None, description="Name of the uploaded file")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    num_detections: int = Field(..., ge=0, description="Total number of detections")
    detections: List[BoundingBox] = Field(default_factory=list, description="List of detections")
    inference_ms: float = Field(..., description="Inference latency in milliseconds")
    annotated_image_b64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the annotated image (only when return_image=true)",
    )

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the service is running")
    model_loaded: bool = Field(..., description="True when the YOLO model is ready")

class ModelInfoResponse(BaseModel):
    model_path: str = Field(..., description="Absolute path to the loaded model weights")
    task: str = Field(..., description="Model task type (e.g. 'detect')")
    names: Dict[int, str] = Field(..., description="Mapping of class ID → class name")
