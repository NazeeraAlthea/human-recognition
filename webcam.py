import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model_postur.h5")
labels = ["Bending", "Lying", "Sitting", "Standing"]  # Sesuaikan urutan dengan class_indices hasil training

def predict(image):
    if image is None:
        return "No image"
    img = Image.fromarray(image).resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    idx = np.argmax(preds[0])
    return labels[idx]

gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["upload", "webcam"], label="Upload atau Webcam"),
    outputs=gr.Label(label="Prediksi Postur"),
    title="Deteksi Postur Tubuh dari Webcam atau Upload Gambar"
).launch()
