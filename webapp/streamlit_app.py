# webapp/streamlit_app.py
# ============================================================
# WEB APP: Realtime ASL (KNN Landmarks) + Toggle Mirror + TTS
# Fix utama: TTS pakai thread khusus + runAndWait per utterance
# (lebih stabil daripada startLoop/iterate yang sering “macet setelah pertama”)
# ============================================================

from __future__ import annotations

from pathlib import Path
import time
import threading
import queue
from collections import deque

import numpy as np
import cv2
import joblib
import mediapipe as mp
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer


# ============================================================
# 1) PATH ROOT PROJECT
# ============================================================
APP_PATH = Path(__file__).resolve()
ROOT = APP_PATH.parents[1]  # root = 1 level di atas folder webapp

TASK_PATH = ROOT / "assets" / "hand_landmarker.task"
MODEL_PATH = ROOT / "models" / "knn_landmarks.joblib"
LE_PATH = ROOT / "models" / "label_encoder.joblib"


def must_exist(p: Path, msg: str):
    if not p.exists():
        raise FileNotFoundError(f"{msg}: {p}")


must_exist(TASK_PATH, "hand_landmarker.task tidak ditemukan")
must_exist(MODEL_PATH, "Model KNN tidak ditemukan")
must_exist(LE_PATH, "Label encoder tidak ditemukan")


# ============================================================
# 2) LOAD MODEL + LABEL ENCODER
# ============================================================
model = joblib.load(MODEL_PATH)  # biasanya pipeline scaler+knn
le = joblib.load(LE_PATH)


# ============================================================
# 3) TTS WORKER (pyttsx3) - THREAD KHUSUS
#    Kunci: semua pemanggilan pyttsx3 dilakukan hanya di thread ini.
#    Untuk stabilitas, gunakan runAndWait per teks.
#    + fallback: jika error/stall, re-init engine (trik umum utk “speak once”)
# ============================================================
class TTSWorker(threading.Thread):
    def __init__(self, rate: int = 170, volume: float = 1.0):
        super().__init__(daemon=True)
        self.q: "queue.Queue[str]" = queue.Queue()
        self.stop_event = threading.Event()
        self.rate = rate
        self.volume = volume

        import pyttsx3
        self.pyttsx3 = pyttsx3  # simpan modul untuk re-init
        self.engine = self._new_engine()

    def _new_engine(self):
        engine = self.pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        return engine

    def say(self, text: str):
        # cegah spam antrian
        if self.q.qsize() > 20:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(text)

    def run(self):
        while not self.stop_event.is_set():
            try:
                text = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            # sentinel untuk stop
            if text is None:
                break

            # 1) coba speak normal
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                continue
            except Exception:
                pass

            # 2) fallback: re-init engine lalu coba sekali lagi
            try:
                self.engine = self._new_engine()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                # kalau tetap gagal, skip agar app tidak crash
                continue

    def stop(self):
        self.stop_event.set()
        try:
            self.q.put_nowait(None)
        except Exception:
            pass


def ensure_tts_worker() -> TTSWorker:
    """
    Buat worker TTS sekali per session Streamlit.
    (Streamlit rerun berkali-kali, jadi simpan di session_state)
    """
    if "tts_worker" not in st.session_state:
        st.session_state.tts_worker = TTSWorker()
        st.session_state.tts_worker.start()
    else:
        w = st.session_state.tts_worker
        if not w.is_alive():
            st.session_state.tts_worker = TTSWorker()
            st.session_state.tts_worker.start()
    return st.session_state.tts_worker


# ============================================================
# 4) MediaPipe HandLandmarker (VIDEO mode)
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(TASK_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


def landmarks_to_features_from_xyz(pts_xyz: np.ndarray) -> np.ndarray:
    """
    pts_xyz: (21,3) -> output (1,63)
    normalisasi:
    - wrist jadi origin
    - scale normalize
    """
    pts = pts_xyz.astype(np.float32).copy()
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.reshape(1, -1)


def draw_landmarks(frame_bgr: np.ndarray, lms):
    h, w = frame_bgr.shape[:2]
    for lm in lms:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 0), -1)


# ============================================================
# 5) VideoProcessor (streamlit-webrtc)
# ============================================================
class VideoProcessor:
    def __init__(self):
        self.landmarker = HandLandmarker.create_from_options(hand_options)

        # realtime params (diupdate dari UI)
        self.conf_th = 0.60
        self.smooth_n = 9
        self.mirror_view = True

        self.enable_tts = False
        self.speak_cooldown = 0.7
        self.repeat_every = 2.0
        self.tts_worker: TTSWorker | None = None  # disuntik dari main thread

        self.history = deque(maxlen=self.smooth_n)
        self.last_spoken = None
        self.last_spoken_time = 0.0

        self.t0 = time.monotonic()  # timestamp untuk detect_for_video

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        # (A) mirror untuk tampilan
        if self.mirror_view:
            display_bgr = cv2.flip(img_bgr, 1)
        else:
            display_bgr = img_bgr

        # (B) deteksi pada frame tampilan (agar cocok yang dilihat)
        detect_bgr = display_bgr
        detect_rgb = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_rgb)

        timestamp_ms = int((time.monotonic() - self.t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        label_text = "NO_HAND"
        conf = 0.0

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            draw_landmarks(display_bgr, lms)

            # (C) landmark -> fitur
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

            # kalau view mirror, unmirror untuk fitur agar konsisten training
            if self.mirror_view:
                pts[:, 0] = 1.0 - pts[:, 0]

            X = landmarks_to_features_from_xyz(pts)

            # (D) prediksi + confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf = float(np.max(proba))
                pred_idx = int(np.argmax(proba))
            else:
                pred_idx = int(model.predict(X)[0])
                conf = 1.0

            # (E) threshold + smoothing
            if conf >= self.conf_th:
                if self.history.maxlen != self.smooth_n:
                    self.history = deque(self.history, maxlen=self.smooth_n)

                self.history.append(pred_idx)
                stable = max(set(self.history), key=list(self.history).count)
                label_text = le.inverse_transform([stable])[0]

                # (F) TTS: ngomong jika huruf berubah / ulang periodik
                if self.enable_tts and self.tts_worker is not None:
                    now = time.time()
                    should_speak = (
                        (label_text != self.last_spoken and (now - self.last_spoken_time) > self.speak_cooldown)
                        or (label_text == self.last_spoken and (now - self.last_spoken_time) > self.repeat_every)
                    )
                    if should_speak:
                        self.tts_worker.say(label_text)
                        self.last_spoken = label_text
                        self.last_spoken_time = now
            else:
                label_text = "UNKNOWN"
                self.history.clear()

        # overlay teks
        cv2.putText(
            display_bgr,
            f"Pred: {label_text}  conf={conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(display_bgr, format="bgr24")


# ============================================================
# 6) STREAMLIT UI
# ============================================================
st.set_page_config(page_title="ASL KNN Landmarks - Realtime", layout="wide")
st.title("ASL Realtime (KNN Landmarks) — Web App")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kontrol")

    view_mode = st.selectbox("Tampilan Kamera", ["Mirror (selfie)", "Normal (tidak mirror)"], index=0)
    mirror_view = (view_mode == "Mirror (selfie)")

    enable_tts = st.checkbox("Aktifkan TTS (ngomong huruf)", value=False)
    conf_th = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.01)
    smooth_n = st.slider("Smoothing window (voting)", 1, 15, 9, 1)
    speak_cooldown = st.slider("Cooldown bicara (detik)", 0.1, 3.0, 0.7, 0.1)
    repeat_every = st.slider("Ulang huruf sama tiap (detik)", 0.5, 5.0, 2.0, 0.1)

with col2:
    st.subheader("Video Realtime")
    st.caption("Klik START. Browser minta izin kamera, klik Allow.")

# Kalau TTS ON, pastikan worker dibuat di MAIN THREAD (bukan di recv callback)
tts_worker = ensure_tts_worker() if enable_tts else None

ctx = webrtc_streamer(
    key="asl-realtime",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Update parameter ke processor
if ctx.video_processor:
    ctx.video_processor.mirror_view = mirror_view
    ctx.video_processor.conf_th = conf_th
    ctx.video_processor.smooth_n = smooth_n

    ctx.video_processor.enable_tts = enable_tts
    ctx.video_processor.speak_cooldown = speak_cooldown
    ctx.video_processor.repeat_every = repeat_every
    ctx.video_processor.tts_worker = tts_worker

st.info("Jika suara masih hanya sekali: tutup semua terminal Streamlit lama (jangan ada 2 app jalan bersamaan), lalu jalankan ulang.")
