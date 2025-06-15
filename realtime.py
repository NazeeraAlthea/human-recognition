import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model sekali saja
model = tf.keras.models.load_model("model_postur.keras")
labels = ["Bending", "Lying", "Sitting", "Standing"]  # Urutkan sesuai class_indices

st.title("Realtime Posture Detection (Webcam)")

class PostureTransformer(VideoTransformerBase):
    def __init__(self):
        self.label = ""
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        preds = model.predict(img_input)
        idx = np.argmax(preds[0])
        self.label = labels[idx]
        # Tampilkan label di atas frame
        cv2.putText(
            img,
            f"{self.label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img

import cv2
webrtc_streamer(
    key="realtime-posture",
    video_transformer_factory=PostureTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
st.markdown("**Arahkan webcam ke objek. Prediksi postur akan muncul secara otomatis di video!**")
