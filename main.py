import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

st.title("Deteksi Postur Tubuh (Segmentasi Citra Otomatis)")

def segment_people_mediapipe(pil_img, threshold=0.3):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    img = np.array(pil_img.convert("RGB"))
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        results = selfie_seg.process(img)
        mask = results.segmentation_mask
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        output = img.copy()
        output[binary_mask == 0] = [255,255,255]
    return output

def segment_kmeans(pil_img, K=4):
    img = np.array(pil_img.convert("RGB"))
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

uploaded_file = st.file_uploader("Upload gambar", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar asli', use_column_width=True)
    
    # Pilih metode segmentasi
    metode = st.selectbox("Pilih metode segmentasi", ["MediaPipe (Hilangkan background)", "K-Means (Cluster warna)"])
    
    if metode == "MediaPipe (Hilangkan background)":
        threshold = st.slider("Threshold MediaPipe", 0.05, 0.5, 0.3, 0.05)
        seg_img = segment_people_mediapipe(image, threshold)
        st.image(seg_img, caption='Hasil segmentasi MediaPipe', use_container_width=True)
    elif metode == "K-Means (Cluster warna)":
        K = st.slider("Jumlah cluster K-Means", 2, 8, 4, 1)
        seg_img = segment_kmeans(image, K)
        st.image(seg_img, caption=f'Hasil segmentasi K-Means (K={K})', use_container_width=True)
    else:
        st.warning("Pilih metode segmentasi terlebih dahulu.")
        seg_img = np.array(image)
    
    # --- Resize, normalisasi, prediksi ---
    img = cv2.resize(seg_img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Load model postur (hasil training siluet)
    model = tf.keras.models.load_model("model_postur_finetune_v2.keras")
    labels = ["membungkuk", "berbaring", "duduk", "berdiri"]
    preds = model.predict(img)
    idx = np.argmax(preds[0])
    st.markdown(f"""
<span style="background-color:#d1fae5; color:#065f46; padding: 10px 20px; border-radius: 8px; font-size:20px; font-weight: bold;">
Prediksi: {labels[idx].capitalize()}
</span>
""", unsafe_allow_html=True)
