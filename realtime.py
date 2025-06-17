import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp

model = tf.keras.models.load_model("model_postur_finetune_v2.keras")
labels = ["membungkuk", "berbaring", "duduk", "berdiri"]

st.title("Realtime Posture Detection with Automatic Segmentation (Webcam)")

metode = st.sidebar.selectbox(
    "Pilih metode segmentasi",
    ["MediaPipe (hilangkan background)", "KMeans (cluster warna)", "Tanpa segmentasi"],
)

if metode == "MediaPipe (hilangkan background)":
    threshold = st.sidebar.slider("Threshold MediaPipe", 0.05, 0.5, 0.3, 0.05)
elif metode == "KMeans (cluster warna)":
    K = st.sidebar.slider("Jumlah cluster K-Means", 2, 8, 4, 1)

def segment_people_mediapipe(img, threshold=0.3):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        results = selfie_seg.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        output = img.copy()
        output[binary_mask == 0] = [255, 255, 255]
    return output, binary_mask

def get_bounding_box_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

def segment_kmeans(img, K=4):
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

class PostureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        bbox = None
        if metode == "MediaPipe (hilangkan background)":
            seg_img, mask = segment_people_mediapipe(img, threshold)
            bbox = get_bounding_box_from_mask(mask)
        elif metode == "KMeans (cluster warna)":
            seg_img = segment_kmeans(img, K)
            bbox = (0, 0, img.shape[1], img.shape[0])
        else:
            seg_img = img
            bbox = (0, 0, img.shape[1], img.shape[0])

        img_resized = cv2.resize(seg_img, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        preds = model.predict(img_input)
        idx = np.argmax(preds[0])
        label = labels[idx]

        # Tampilkan bounding box
        if bbox is not None:
            x, y, w, h = bbox
            # Kotak Merah Tebal
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Background kotak untuk label (agar tulisan putih tetap jelas)
            label_bg_w = 210
            label_bg_h = 40
            cv2.rectangle(img, (x, y - label_bg_h), (x + label_bg_w, y), (0, 0, 0), -1)

            # Tulis label dengan warna putih, tebal
            cv2.putText(
                img, f"{label.capitalize()}",
                (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA
            )
        else:
            # Jika tidak ada bbox, tetap tulis label di pojok kiri atas
            cv2.putText(
                img, f"{label.capitalize()}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA
            )
        return img

webrtc_streamer(
    key="realtime-posture",
    video_transformer_factory=PostureTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.markdown(
    "**Arahkan webcam ke objek manusia. Prediksi postur dan bounding box akan muncul secara otomatis di video!**"
)
