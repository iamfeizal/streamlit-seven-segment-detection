import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")  # path model YOLO11L Anda
    return model

model = load_model()

st.set_page_config(
    page_title="YOLOv11L Demo",
    layout="wide"
)

st.title("🔍 YOLOv11L Object Detection Demo")
st.write("Upload gambar untuk mendeteksi objek menggunakan model YOLOv11L Anda.")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        # read image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        st.subheader("Original Image")
        st.image(image)

    with col2:
        # run prediction
        # st.subheader("Running YOLOv11L inference...")
        start = time.time()
        results = model(img_np, verbose=False)
        end = time.time()

        # -----------------------------
        # Draw bounding boxes
        # -----------------------------
        result = results[0]
        det_list = []
        for box in result.boxes:
            det_list.append({
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
            
        # ambil bbox[0] yaitu x1
        detections_sorted = sorted(det_list, key=lambda d: d["bbox"][0])

        # hanya ambil class_id dalam urutan kiri → kanan
        class_sequence = [d["class_id"] for d in detections_sorted]

        annotated_img = result.plot()  # ultralytics provides this directly

        st.subheader(f"Detected Objects (Inference time: {end - start:.2f} sec)")
        st.image(annotated_img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detect (left→right)")
        st.subheader(class_sequence)
        

    with col2:
        # -----------------------------
        # Show detection data
        # -----------------------------
        st.subheader("Detection Details (class, confidence, bbox)")
        
        

        st.json(detections_sorted)

else:
    st.info("Silakan upload gambar untuk memulai deteksi.")
