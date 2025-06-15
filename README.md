# 🚶‍♂️ Deteksi Postur Tubuh Otomatis (MobileNetV2 + Streamlit)

![Python](https://img.shields.io/badge/python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📖 Ringkasan

Proyek ini adalah sistem deteksi postur tubuh manusia (berdiri, duduk, berbaring, membungkuk) berbasis deep learning menggunakan MobileNetV2 dan Streamlit.  
Dilengkapi segmentasi otomatis (MediaPipe/KMeans) agar hasil prediksi tetap akurat pada gambar dengan background variatif.

---

## ✨ Fitur

- **Deteksi 4 postur utama:** Berdiri, Duduk, Membungkuk, Berbaring
- **Upload gambar langsung di web (Streamlit)**
- **Segmentasi otomatis:** 
  - **MediaPipe:** Hilangkan background, fokus ke manusia
  - **KMeans:** Segmentasi area warna
- **Augmentasi data otomatis** (rotasi, flip, brightness, dsb)
- **Training + Fine-tuning transfer learning**
- **Model versioning** (setiap eksperimen, model disimpan berbeda)
- **Visualisasi akurasi dan confusion matrix**

---

## 🏗️ Arsitektur
dataset/
├─ berdiri/
├─ duduk/
├─ membungkuk/
└─ berbaring/
train_v1.py
train_v2.py
main.py
model_postur_finetune_vX.keras
requirements.txt
README.md


---

## 🛠️ Instalasi

1. **Clone repo & buat virtual environment**
    ```bash
    git clone https://github.com/username/nama-repo.git
    cd nama-repo
    python -m venv venv
    venv\Scripts\activate        # Windows
    # source venv/bin/activate  # Linux/Mac
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📁 Struktur Dataset
Link Download:
https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture

dataset/
├─ bending/
├─ lying/
├─ sitting/
└─ standing/

## 🚀 Cara Penggunaan
 // notes:  gunakan salah satu train yang anda inginkan v1 atau v2
            jika menggunakan v1 maka ganti baris 
            v1 : model = tf.keras.models.load_model("model_postur_finetune.keras")
            v2 : model = tf.keras.models.load_model("model_postur_finetune_v2.keras")
1. **Training Model**
    ```bash
    python train_v1.py
    ```
    - Model akan disimpan otomatis sebagai `model_postur_finetune.keras`.

2. **Jalankan Aplikasi Web**
    ```bash
    streamlit run main.py
    ```
    - Akses di browser: [http://localhost:8501](http://localhost:8501)

---
