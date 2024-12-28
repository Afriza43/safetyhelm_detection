import cv2
from google.colab.patches import cv2_imshow  # Import untuk menampilkan gambar di Colab
from ultralytics import YOLO

# Inisialisasi model YOLOv8 dengan file model yang sudah dilatih
model = YOLO('/content/best.pt')

# Definisikan classes yang digunakan oleh model (sesuai dengan dataset)
classes = ['with helmet', 'no helmet', 'person']


# Fungsi untuk melakukan deteksi dan menampilkan bounding box dan label
def detect_objects(image):
    # Lakukan prediksi pada gambar yang diinput
    results = model(image)

    # Ambil bounding box, confidence score, dan label prediksi
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy()  # Bounding box koordinat
        class_id = int(result.cls[0])  # ID class
        confidence = result.conf[0].cpu().numpy()  # Confidence score

        # Gambar bounding box di gambar asli
        cv2.rectangle(image,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      (0, 255, 0), 2)

        # Tampilkan label class dan confidence di atas bounding box
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.putText(image, label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Tampilkan gambar hasil deteksi di Colab
    cv2_imshow(image)



# Fungsi untuk membaca gambar atau video input
def process_input(input_path):
    # Cek apakah input berupa video atau gambar
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        # Jika input berupa video
        cap = cv2.VideoCapture(input_path)

        # Baca frame per frame dari video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi objek di frame video
            detect_objects(frame)

            # Tampilkan frame hasil deteksi
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Lepaskan resource video setelah selesai
        cap.release()
    else:
        # Jika input berupa gambar
        image = cv2.imread(input_path)

        # Deteksi objek di gambar
        detect_objects(image)

        # Tunggu sampai pengguna menekan tombol untuk menutup jendela
        cv2.waitKey(0)

    # Tutup semua jendela OpenCV setelah selesai
    cv2.destroyAllWindows()

# Input file (bisa berupa video atau foto)
input_path = '/content/foto1.jpg'  # Ubah sesuai dengan path video atau gambar Anda
process_input(input_path)