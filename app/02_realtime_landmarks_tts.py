from pathlib import Path  # (keterangan) path
import time  # (keterangan) timestamp/cooldown
from collections import deque  # (keterangan) smoothing prediksi

import cv2  # (keterangan) webcam
import numpy as np  # (keterangan) numerik
import joblib  # (keterangan) load model
import mediapipe as mp  # (keterangan) hand landmarker
import pyttsx3  # (keterangan) TTS offline

# ====== ROOT PROJECT ======
SCRIPT_PATH = Path(__file__).resolve()  # (keterangan) lokasi file ini
ROOT = SCRIPT_PATH.parents[1]  # (keterangan) root project

# ====== PATH MODEL ======
MODEL_PATH = ROOT / "models" / "knn_landmarks.joblib"  # (keterangan) model KNN
LE_PATH = ROOT / "models" / "label_encoder.joblib"  # (keterangan) label encoder
TASK_PATH = ROOT / "assets" / "hand_landmarker.task"  # (keterangan) model MediaPipe

if not MODEL_PATH.exists():  # (keterangan) pastikan model ada
    raise FileNotFoundError(f"Model KNN tidak ditemukan: {MODEL_PATH}\nJalankan training dulu.")  # (keterangan) stop

if not LE_PATH.exists():  # (keterangan) pastikan encoder ada
    raise FileNotFoundError(f"Label encoder tidak ditemukan: {LE_PATH}\nJalankan training dulu.")  # (keterangan) stop

if not TASK_PATH.exists():  # (keterangan) pastikan task ada
    raise FileNotFoundError(f"hand_landmarker.task tidak ditemukan: {TASK_PATH}")  # (keterangan) stop

model = joblib.load(MODEL_PATH)  # (keterangan) load pipeline
le = joblib.load(LE_PATH)  # (keterangan) load label encoder

# ====== PARAMETER REALTIME ======
MIRROR = True  # (keterangan) mirror agar seperti kaca
CONF_TH = 0.60  # (keterangan) threshold confidence minimum
SMOOTH_N = 9  # (keterangan) ukuran smoothing window
SPEAK_COOLDOWN = 0.7  # (keterangan) minimal jeda bicara saat huruf berubah
REPEAT_EVERY = 2.0  # (keterangan) ulangi huruf yang sama tiap N detik (agar tidak cuma sekali)

history = deque(maxlen=SMOOTH_N)  # (keterangan) simpan prediksi terakhir
last_spoken = None  # (keterangan) huruf terakhir yang diucapkan
last_spoken_time = 0.0  # (keterangan) waktu terakhir speak

# ====== TTS: EXTERNAL EVENT LOOP (SUPAYA TIDAK MACET) ======
engine = pyttsx3.init()  # (keterangan) init engine
engine.setProperty("rate", 170)  # (keterangan) kecepatan bicara
engine.setProperty("volume", 1.0)  # (keterangan) volume
engine.startLoop(False)  # (keterangan) external loop (kita panggil iterate() sendiri) :contentReference[oaicite:9]{index=9}

def speak(text: str) -> None:  # (keterangan) enqueue ucapan (tanpa runAndWait)
    engine.say(text)  # (keterangan) antrikan teks ke engine

# ====== MEDIAPIPE HAND LANDMARKER: VIDEO MODE ======
BaseOptions = mp.tasks.BaseOptions  # (keterangan) alias
HandLandmarker = mp.tasks.vision.HandLandmarker  # (keterangan) class
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions  # (keterangan) options
VisionRunningMode = mp.tasks.vision.RunningMode  # (keterangan) enum mode

options = HandLandmarkerOptions(  # (keterangan) buat options
    base_options=BaseOptions(model_asset_path=str(TASK_PATH)),  # (keterangan) path task
    running_mode=VisionRunningMode.VIDEO,  # (keterangan) VIDEO untuk webcam (detect_for_video)
    num_hands=1,  # (keterangan) 1 tangan
    min_hand_detection_confidence=0.5,  # (keterangan) threshold deteksi
    min_hand_presence_confidence=0.5,  # (keterangan) threshold presence
    min_tracking_confidence=0.5,  # (keterangan) threshold tracking
)  # (keterangan) selesai

def landmarks_to_features(lms) -> np.ndarray:  # (keterangan) landmark -> fitur (1,63)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # (keterangan) 21x3
    pts = pts - pts[0]  # (keterangan) wrist origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))  # (keterangan) skala
    if scale < 1e-6:  # (keterangan) cegah nol
        scale = 1.0  # (keterangan) fallback
    pts = pts / scale  # (keterangan) normalisasi skala
    return pts.reshape(1, -1)  # (keterangan) (1,63)

cap = cv2.VideoCapture(0)  # (keterangan) buka webcam index 0
if not cap.isOpened():  # (keterangan) jika webcam gagal
    engine.endLoop()  # (keterangan) tutup loop TTS sebelum raise
    raise RuntimeError("Webcam tidak bisa dibuka. Coba ganti index camera (0/1/2).")  # (keterangan) stop

start_time = time.time()  # (keterangan) start untuk timestamp_ms

try:  # (keterangan) try-finally agar cleanup pasti jalan
    with HandLandmarker.create_from_options(options) as landmarker:  # (keterangan) buat landmarker
        while True:  # (keterangan) loop realtime
            ok, frame = cap.read()  # (keterangan) baca frame
            if not ok:  # (keterangan) jika gagal baca
                break  # (keterangan) keluar

            if MIRROR:  # (keterangan) mirror jika dipilih
                frame = cv2.flip(frame, 1)  # (keterangan) flip horizontal

            H, W = frame.shape[:2]  # (keterangan) ukuran frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (keterangan) BGR->RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # (keterangan) mp.Image

            timestamp_ms = int((time.time() - start_time) * 1000)  # (keterangan) timestamp untuk VIDEO mode
            result = landmarker.detect_for_video(mp_image, timestamp_ms)  # (keterangan) deteksi video :contentReference[oaicite:10]{index=10}

            label_text = "NO_HAND"  # (keterangan) default
            conf = 0.0  # (keterangan) default confidence

            if result.hand_landmarks:  # (keterangan) jika ada tangan
                lms = result.hand_landmarks[0]  # (keterangan) tangan pertama

                # (keterangan) gambar titik landmark kecil (opsional)
                for lm in lms:  # (keterangan) loop 21 landmark
                    cx, cy = int(lm.x * W), int(lm.y * H)  # (keterangan) normalisasi -> pixel
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)  # (keterangan) gambar titik

                X = landmarks_to_features(lms)  # (keterangan) buat fitur

                # (keterangan) pakai proba kalau tersedia (Pipeline KNN biasanya ada)
                if hasattr(model, "predict_proba"):  # (keterangan) cek method
                    proba = model.predict_proba(X)[0]  # (keterangan) probabilitas tiap kelas
                    conf = float(np.max(proba))  # (keterangan) confidence max
                    pred_idx = int(np.argmax(proba))  # (keterangan) index prediksi
                else:  # (keterangan) fallback
                    pred_idx = int(model.predict(X)[0])  # (keterangan) prediksi langsung
                    conf = 1.0  # (keterangan) anggap yakin

                if conf < CONF_TH:  # (keterangan) jika tidak yakin
                    label_text = "UNKNOWN"  # (keterangan) tampilkan unknown
                    history.clear()  # (keterangan) reset smoothing
                else:  # (keterangan) jika yakin
                    history.append(pred_idx)  # (keterangan) simpan prediksi
                    stable = max(set(history), key=list(history).count)  # (keterangan) voting mayoritas
                    label_text = le.inverse_transform([stable])[0]  # (keterangan) index -> huruf

                    now = time.time()  # (keterangan) waktu sekarang
                    # (keterangan) speak jika huruf berubah, atau huruf sama tapi sudah lewat REPEAT_EVERY
                    should_speak = (label_text != last_spoken and (now - last_spoken_time) > SPEAK_COOLDOWN) \
                                   or (label_text == last_spoken and (now - last_spoken_time) > REPEAT_EVERY)  # (keterangan) kondisi speak

                    if should_speak:  # (keterangan) jika memenuhi
                        speak(label_text)  # (keterangan) enqueue ucapan
                        last_spoken = label_text  # (keterangan) update huruf terakhir
                        last_spoken_time = now  # (keterangan) update waktu

            # (keterangan) pump event loop TTS (ini yang bikin tidak “cuma sekali”) :contentReference[oaicite:11]{index=11}
            engine.iterate()  # (keterangan) proses queue TTS tanpa blocking

            cv2.putText(frame, f"Pred: {label_text} conf={conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # (keterangan) overlay teks
            cv2.imshow("ASL Landmarks (KNN) + TTS", frame)  # (keterangan) tampilkan frame

            key = cv2.waitKey(1) & 0xFF  # (keterangan) baca tombol
            if key == 27 or key == ord("q"):  # (keterangan) ESC atau q
                break  # (keterangan) keluar loop

finally:  # (keterangan) cleanup
    cap.release()  # (keterangan) lepas webcam
    cv2.destroyAllWindows()  # (keterangan) tutup window
    engine.endLoop()  # (keterangan) akhiri loop TTS :contentReference[oaicite:12]{index=12}

#.\.venv\Scripts\python.exe .\app\realtime_sign_mnist.py