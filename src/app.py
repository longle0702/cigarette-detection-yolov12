import os
import time
import base64
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best.pt")
st.set_page_config(
    page_title="🚬 Cigarette Detection",
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

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #222;
    }

    /* Main background */
    .stApp {
        background: #0a0a0a;
        color: #f0f0f0;
    }

    /* Cards */
    .result-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }

    /* Detection badge */
    .badge {
        display: inline-block;
        background: #222;
        border: 1px solid #3a3a3a;
        color: #d0d0d0;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin: 2px;
    }

    /* Metric tiles */
    .metric-tile {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: center;
    }
    .metric-tile h2 {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    .metric-tile p {
        margin: 0.3rem 0 0;
        font-size: 0.75rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.2rem;
    }
    .hero h1 {
        font-size: 2.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero h1 span {
        color: #aaaaaa;
    }
    .hero p {
        color: #555;
        font-size: 0.95rem;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #111;
        border: 2px dashed #2e2e2e;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Buttons */
    .stButton > button, [data-testid="stDownloadButton"] button {
        background: #1e1e1e;
        color: #f0f0f0;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.5rem 1.3rem;
        font-weight: 600;
        font-size: 0.88rem;
        transition: background 0.15s, border-color 0.15s;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background: #2a2a2a;
        border-color: #555;
    }

    /* Divider */
    hr { border-color: #1e1e1e; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading YOLOv12 model …")
def load_model():
    logger.info("Loading model from %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded.")
    return model

def run_inference(model, img_bgr, conf, iou, imgsz):
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
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

st.markdown(
    """
    <div class="hero">
        <h1>🚬 <span>Cigarette</span> Detection</h1>
        <p>Real-time detection powered by <strong>YOLOv12</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Inference Settings")
    st.markdown("---")

    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Minimum confidence score to keep a detection.",
    )
    iou_thresh = st.slider(
        "IoU threshold (NMS)",
        min_value=0.01,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Intersection-over-Union threshold for non-maximum suppression.",
    )
    imgsz = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 1024, 1280, 1920],
        value=640,
        help="Image will be resized to this value before inference.",
    )
    show_original = st.toggle("Show original image", value=True)

    st.markdown("---")
    st.markdown("### 📦 Model Info")
    model = load_model()
    st.caption(f"**Path:** `{MODEL_PATH}`")
    st.caption(f"**Task:** `{model.task}`")
    st.caption(f"**Classes:** {', '.join(model.names.values())}")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    pil_orig, bgr = pil_from_upload(uploaded)
    h, w = bgr.shape[:2]

    with st.spinner("Running inference …"):
        result, inference_ms = run_inference(model, bgr, conf_thresh, iou_thresh, imgsz)

    detections = []
    for box in result.boxes:
        xyxy = [round(v, 2) for v in box.xyxy[0].tolist()]
        xywhn = [round(v, 6) for v in box.xywhn[0].tolist()]
        class_id = int(box.cls[0])
        label = model.names.get(class_id, str(class_id))
        confidence = round(float(box.conf[0]), 4)
        detections.append(
            {
                "label": label,
                "class_id": class_id,
                "confidence": confidence,
                "bbox_xyxy": xyxy,
                "bbox_xywhn": xywhn,
            }
        )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    ann_pil = annotated_pil(result)

    logger.info(
        "Detected %d cigarette(s) in %.1f ms | conf=%.2f iou=%.2f imgsz=%d | file=%s",
        len(detections),
        inference_ms,
        conf_thresh,
        iou_thresh,
        imgsz,
        uploaded.name,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    def metric_tile(col, value, label):
        col.markdown(
            f'<div class="metric-tile"><h2>{value}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    metric_tile(col1, len(detections), "Detections")
    metric_tile(col2, f"{inference_ms:.0f} ms", "Inference Time")
    metric_tile(col3, f"{w}×{h}", "Image Size")
    metric_tile(
        col4,
        f"{max((d['confidence'] for d in detections), default=0):.0%}" if detections else "—",
        "Top Confidence",
    )

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

    buf = _io.BytesIO()
    ann_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name=f"annotated_{uploaded.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Detections")

    if not detections:
        st.info("No cigarettes detected at the current confidence threshold. Try lowering it in the sidebar.")
    else:
        for i, det in enumerate(detections, 1):
            with st.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="badge">#{i}</span>
                        <span class="badge">{det['label']}</span>
                        <span class="badge">conf {det['confidence']:.1%}</span>
                        <br><br>
                        <strong>BBox (x1,y1,x2,y2):</strong> {det['bbox_xyxy']}<br>
                        <strong>BBox (cx,cy,w,h norm):</strong> {det['bbox_xywhn']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

else:
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 1rem;opacity:0.4;">
            <div style="font-size:5rem;">📂</div>
            <p style="font-size:1.1rem;margin-top:1rem;">Upload an image above to get started</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
