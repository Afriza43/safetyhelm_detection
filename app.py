import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# Inisialisasi model YOLOv8
model = YOLO('best.pt')

# Definisikan classes yang digunakan oleh modelG
classes = ['with helmet', 'no helmet', 'person']

# Fungsi untuk melakukan deteksi dan menampilkan bounding box dan label


def detect_objects(image):
    results = model(image)
    helmet_missing = False  # Flag untuk mendeteksi apakah ada "no helmet"

    # Ambil bounding box, confidence score, dan label prediksi
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy()
        class_id = int(result.cls[0])
        confidence = result.conf[0].cpu().numpy()

        # Tentukan warna bounding box berdasarkan label
        if classes[class_id] == 'with helmet':
            color = (0, 255, 0)  # Hijau untuk with helmet
        elif classes[class_id] == 'no helmet':
            color = (0, 0, 255)  # Merah untuk no helmet
        else:
            color = (255, 0, 0)  # Biru untuk person

        # Gambar bounding box
        cv2.rectangle(image, (int(bbox[0]), int(
            bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (int(bbox[0]), int(
            bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Cek apakah ada objek dengan label 'no helmet'
        if classes[class_id] == 'no helmet':
            helmet_missing = True

    return image, helmet_missing

# Fungsi untuk memproses video


def process_video(input_video):
    # Buat temporary file untuk menyimpan output video
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = tmp_output.name

    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Codec untuk output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    helmet_missing = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lakukan deteksi objek pada frame
        detected_frame, helmet_flag = detect_objects(frame)

        # Simpan frame hasil deteksi ke output video
        out.write(detected_frame)

        # Periksa apakah ada orang tanpa helm
        if helmet_flag:
            helmet_missing = True

    cap.release()
    out.release()

    return output_path, helmet_missing


# Streamlit app
st.title("Safety Helmet Detection")

# Pilihan mode deteksi: upload gambar atau video
uploaded_file = st.file_uploader("Upload an image or video", type=[
                                 'jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Lakukan deteksi pada gambar
        detected_image, helmet_missing = detect_objects(image_np)
        st.image(detected_image, caption="Detected Image",
                 use_column_width=True)

        # Jika ada objek 'no helmet', tampilkan peringatan
        if helmet_missing:
            st.warning("Warning: A person without a helmet was detected!")
        else:
            st.success("All persons are wearing helmets.")

    elif uploaded_file.type.startswith('video'):
        # Simpan video sementara untuk diproses
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Lakukan deteksi pada video
        output_video_path, helmet_missing = process_video(tfile.name)

        # Jika ada objek 'no helmet', tampilkan peringatan
        if helmet_missing:
            st.warning("Warning: A person without a helmet was detected!")
        else:
            st.success("All persons are wearing helmets.")

        # Tambahkan opsi untuk mengunduh video hasil deteksi
        with open(output_video_path, 'rb') as f:
            st.download_button('Download Detected Video', f,
                               file_name='detected_video.mp4')
