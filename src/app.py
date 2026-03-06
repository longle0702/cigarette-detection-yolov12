import os
import time
import logging
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io as _io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cigarette-streamlit")

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
model_path = os.path.join(root_dir, "models", "best.pt")

st.set_page_config(
    page_title="Cigarette Detection",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp { background: #0b0b0d; color: #ffffff; }
    section[data-testid="stSidebar"] { background: #08080a; border-right: 1px solid #1f1f22; }

    [data-testid="stImage"] button, 
    [data-testid="stImageHoverContainer"] button {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }

    [data-testid="stWidgetLabel"] + div div,
    [data-testid="stSlider"] div {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stFileUploader"] {
        opacity: 1 !important;
        border: 2px dashed #2d2d32;
    }

    .result-card, .metric-tile {
        background: linear-gradient(145deg, #0b0b0d, #111113);
        border: 1px solid #2a2a2d;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }

    .metric-tile { text-align: center; }
    .metric-tile h2 { font-size: 1.9rem; font-weight: 700; margin: 0; color: #ffb347; }
    .metric-tile p { margin: 0.35rem 0 0; font-size: 0.75rem; color: #9a9a9a; text-transform: uppercase; }

    .badge {
        display: inline-block;
        background: rgba(255, 179, 71, 0.12);
        border: 1px solid rgba(255, 179, 71, 0.35);
        color: #ffb347;
        padding: 3px 12px;
        border-radius: 18px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }

    .hero { text-align: center; padding: 2rem 1rem 1.2rem; }
    .hero h1 { font-size: 2.3rem; font-weight: 700; color: #e6e6e6; }
    .hero h1 span { color: #ffb347; }

    .stButton > button, [data-testid="stDownloadButton"] button {
        background: #111113;
        color: #ffffff;
        border: 1px solid #2f2f33;
        border-radius: 8px;
        padding: 0.5rem 1.3rem;
        font-weight: 600;
        transition: all 0.18s ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background: #ffb347;
        border-color: #ffb347;
        color: #000000;
    }

    hr { border-color: #1f1f22; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading YOLOv12 model …")
def load_model():
    model = YOLO(model_path)
    return model

def run_inference(model, img_bgr, conf, iou, imgsz):
    t0 = time.perf_counter()
    results = model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results[0], elapsed_ms

def annotated_pil(result):
    ann_bgr = result.plot()
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(ann_rgb)

def pil_from_upload(upload) -> tuple:
    pil_img = Image.open(upload).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, bgr

st.markdown('<div class="hero"><h1>Cigarette Detection</h1><p>Real-time detection powered by YOLOv12</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Inference Settings")
    st.markdown("---")
    conf_thresh = st.slider("Confidence threshold", 0.01, 1.0, 0.15, 0.01, help="Minimum confidence score for valid detections")
    iou_thresh = st.slider("IoU threshold (NMS)", 0.01, 1.0, 0.45, 0.01, help="Controls overlap removal in NMS")
    imgsz = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 1024, 1280, 1920], value=640, help="The size to which the input image is resized before inference")
    show_original = st.toggle("Show original image", value=True)
    st.markdown("---")
    model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

if uploaded is not None:
    pil_orig, bgr = pil_from_upload(uploaded)
    h, w = bgr.shape[:2]

    with st.spinner("Running inference …"):
        result, inference_ms = run_inference(model, bgr, conf_thresh, iou_thresh, imgsz)

    detections = []
    for box in result.boxes:
        detections.append({
            "label": model.names.get(int(box.cls[0]), "cigarette"),
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(v, 2) for v in box.xyxy[0].tolist()],
            "bbox_xywhn": [round(v, 6) for v in box.xywhn[0].tolist()]
        })
    
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    ann_pil = annotated_pil(result)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    def metric_tile(col, value, label):
        col.markdown(f'<div class="metric-tile"><h2>{value}</h2><p>{label}</p></div>', unsafe_allow_html=True)
    
    metric_tile(col1, len(detections), "Detections")
    metric_tile(col2, f"{inference_ms:.0f} ms", "Inference Time")
    metric_tile(col3, f"{w}×{h}", "Image Size")
    metric_tile(col4, f"{max((d['confidence'] for d in detections), default=0):.0%}" if detections else "—", "Top Confidence")

    st.markdown("<br>", unsafe_allow_html=True)
    if show_original:
        img_col1, img_col2 = st.columns(2, gap="large")
        with img_col1:
            st.markdown("**Original**")
            st.image(pil_orig, use_container_width=True)
        with img_col2:
            st.markdown("**Annotated**")
            st.image(ann_pil, use_container_width=True)
    else:
        st.image(ann_pil, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, _, btn_col = st.columns([1, 1, 1])
    with btn_col:
        buf = _io.BytesIO()
        ann_pil.save(buf, format="PNG")
        st.download_button(
            label="Download annotated image",
            data=buf.getvalue(),
            file_name=f"annotated_{uploaded.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Detections")
    if not detections:
        st.info("No cigarettes detected.")
    else:
        for i, det in enumerate(detections, 1):
            st.markdown(f"""
                <div class="result-card">
                    <span class="badge">#{i}</span> <span class="badge">{det['label']}</span> <span class="badge">conf {det['confidence']:.1%}</span><br><br>
                    <strong>BBox (xyxy):</strong> {det['bbox_xyxy']}<br>
                    <strong>BBox (norm):</strong> {det['bbox_xywhn']}
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center;padding:4rem 1rem;opacity:0.4;"><div style="font-size:5rem;">📂</div><p>Upload an image above to get started</p></div>', unsafe_allow_html=True)