from pathlib import Path  # (keterangan) untuk path file/folder yang aman
import csv  # (keterangan) untuk menulis CSV
import random  # (keterangan) untuk sampling acak
import cv2  # (keterangan) OpenCV untuk baca gambar
import numpy as np  # (keterangan) numerik
import mediapipe as mp  # (keterangan) MediaPipe hand landmarker

# ====== ROOT PROJECT (berdasarkan lokasi file .py ini) ======
SCRIPT_PATH = Path(__file__).resolve()  # (keterangan) path file script saat ini
ROOT = SCRIPT_PATH.parents[1]  # (keterangan) root project = 2 level di atas app/

# ====== PATH INPUT/OUTPUT ======
TASK_PATH = ROOT / "assets" / "hand_landmarker.task"  # (keterangan) model task MediaPipe
DATASET_ROOT = ROOT / "data" / "raw" / "asl-alphabet"  # (keterangan) folder dataset
OUT_CSV = ROOT / "data" / "landmarks" / "asl_alphabet_landmarks.csv"  # (keterangan) output CSV

# ====== PARAMETER EKSTRAK ======
MAX_PER_CLASS = 300  # (keterangan) batasi gambar per kelas agar cepat (naikkan kalau mau akurasi lebih bagus)
SEED = 42  # (keterangan) seed random agar reproducible
EXCLUDE = {"nothing", "space", "del", "delete"}  # (keterangan) kelas opsional untuk di-skip

# ====== VALIDASI FILE/FOLDER ======
if not TASK_PATH.exists():  # (keterangan) pastikan model ada
    raise FileNotFoundError(f"Model tidak ditemukan: {TASK_PATH}")  # (keterangan) stop jika tidak ada

if not DATASET_ROOT.exists():  # (keterangan) pastikan dataset root ada
    raise FileNotFoundError(f"Folder dataset tidak ditemukan: {DATASET_ROOT}")  # (keterangan) stop jika tidak ada

# ====== AUTO-DETECT FOLDER TRAIN (anti path salah / folder dobel) ======
def find_train_dir(base: Path) -> Path:  # (keterangan) cari folder yang benar berisi A/B/C dll
    candidates = [  # (keterangan) kandidat struktur yang umum
        base / "asl_alphabet_train",  # (keterangan) struktur normal
        base / "asl_alphabet_train" / "asl_alphabet_train",  # (keterangan) struktur folder dobel
    ]  # (keterangan) selesai kandidat

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}  # (keterangan) ekstensi gambar valid

    for c in candidates:  # (keterangan) cek kandidat satu per satu
        if c.exists() and c.is_dir():  # (keterangan) kandidat harus ada dan folder
            subdirs = [d for d in c.iterdir() if d.is_dir()]  # (keterangan) subfolder kelas
            if len(subdirs) >= 5:  # (keterangan) indikasi folder train (banyak kelas)
                for d in subdirs[:10]:  # (keterangan) cek beberapa kelas pertama
                    has_img = any(  # (keterangan) True jika ada minimal 1 gambar di folder kelas
                        (p.is_file() and p.suffix in exts)  # (keterangan) file gambar
                        for p in d.iterdir()  # (keterangan) iterasi file di kelas
                    )  # (keterangan) selesai any
                    if has_img:  # (keterangan) jika ada gambar
                        return c  # (keterangan) kandidat ini adalah train dir

    # (keterangan) fallback scan jika kandidat tidak cocok
    for c in base.rglob("*"):  # (keterangan) scan semua subfolder
        if c.is_dir() and (c / "A").is_dir():  # (keterangan) sinyal kuat ada folder A
            return c  # (keterangan) gunakan folder ini

    raise FileNotFoundError("Tidak ketemu folder train yang berisi subfolder kelas A/B/C dan file gambar.")  # (keterangan) error jika gagal

TRAIN_DIR = find_train_dir(DATASET_ROOT)  # (keterangan) tentukan folder train
print("✅ TRAIN_DIR:", TRAIN_DIR)  # (keterangan) tampilkan untuk debugging

# ====== BUAT FOLDER OUTPUT ======
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)  # (keterangan) pastikan folder data/landmarks ada

# ====== LIST GAMBAR (cek cepat agar tidak 0) ======
all_imgs = list(TRAIN_DIR.rglob("*.jpg")) + list(TRAIN_DIR.rglob("*.jpeg")) + list(TRAIN_DIR.rglob("*.png"))  # (keterangan) hitung semua gambar
print("🖼️ Total gambar terdeteksi:", len(all_imgs))  # (keterangan) harus > 0

if len(all_imgs) == 0:  # (keterangan) kalau 0 pasti salah extract/path
    raise RuntimeError(f"Tidak ada gambar ditemukan di {TRAIN_DIR}. Pastikan dataset sudah diekstrak dengan benar.")  # (keterangan) stop

# ====== MEDIA PIPE HAND LANDMARKER (IMAGE MODE) ======
BaseOptions = mp.tasks.BaseOptions  # (keterangan) alias BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker  # (keterangan) class HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions  # (keterangan) opsi HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode  # (keterangan) enum running mode

options = HandLandmarkerOptions(  # (keterangan) buat options
    base_options=BaseOptions(model_asset_path=str(TASK_PATH)),  # (keterangan) path model
    running_mode=VisionRunningMode.IMAGE,  # (keterangan) IMAGE untuk dataset gambar (detect())
    num_hands=1,  # (keterangan) cukup 1 tangan
    min_hand_detection_confidence=0.5,  # (keterangan) threshold deteksi
    min_hand_presence_confidence=0.5,  # (keterangan) threshold presence
    min_tracking_confidence=0.5,  # (keterangan) threshold tracking (tidak terlalu penting di image)
)  # (keterangan) selesai options

def landmarks_to_features(lms) -> np.ndarray:  # (keterangan) 21 landmark -> 63 fitur
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # (keterangan) ambil x,y,z (21x3)
    pts = pts - pts[0]  # (keterangan) translasi-invariant: jadikan wrist (0) sebagai origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))  # (keterangan) skala = jarak 2D maksimum
    if scale < 1e-6:  # (keterangan) cegah nol
        scale = 1.0  # (keterangan) fallback
    pts = pts / scale  # (keterangan) scale-invariant: normalisasi skala
    return pts.reshape(-1)  # (keterangan) flatten jadi (63,)

# ====== TULIS HEADER CSV ======
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:  # (keterangan) buat file CSV baru
    writer = csv.writer(f)  # (keterangan) CSV writer
    writer.writerow(["label"] + [f"f{i}" for i in range(63)])  # (keterangan) header: label + 63 fitur

random.seed(SEED)  # (keterangan) set seed
class_folders = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]  # (keterangan) list folder kelas

if not class_folders:  # (keterangan) kalau tidak ada folder kelas
    raise RuntimeError(f"Tidak ada subfolder kelas di {TRAIN_DIR}. Cek struktur dataset.")  # (keterangan) stop

total_written = 0  # (keterangan) total baris sukses ditulis
total_nohand = 0  # (keterangan) total gambar yang tidak terdeteksi tangan

with HandLandmarker.create_from_options(options) as landmarker:  # (keterangan) buat landmarker
    for folder in sorted(class_folders, key=lambda p: p.name.lower()):  # (keterangan) loop tiap kelas (urut)
        label_lower = folder.name.lower()  # (keterangan) nama kelas lowercase
        if label_lower in EXCLUDE:  # (keterangan) skip kelas tertentu jika di-set
            continue  # (keterangan) lanjut kelas berikutnya

        imgs = list(folder.rglob("*.jpg")) + list(folder.rglob("*.jpeg")) + list(folder.rglob("*.png"))  # (keterangan) semua gambar kelas
        if not imgs:  # (keterangan) jika kelas kosong
            print(f"[SKIP] {folder.name}: tidak ada gambar")  # (keterangan) info
            continue  # (keterangan) lanjut

        random.shuffle(imgs)  # (keterangan) acak urutan
        imgs = imgs[:MAX_PER_CLASS]  # (keterangan) batasi jumlah
        wrote_this = 0  # (keterangan) counter per kelas

        for img_path in imgs:  # (keterangan) loop setiap gambar
            bgr = cv2.imread(str(img_path))  # (keterangan) baca gambar BGR
            if bgr is None:  # (keterangan) kalau gagal baca
                continue  # (keterangan) skip

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (keterangan) convert ke RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # (keterangan) bungkus mp.Image

            result = landmarker.detect(mp_image)  # (keterangan) IMAGE mode => detect()
            if not result.hand_landmarks:  # (keterangan) kalau tidak ada tangan
                total_nohand += 1  # (keterangan) tambah nohand
                continue  # (keterangan) skip

            feat = landmarks_to_features(result.hand_landmarks[0])  # (keterangan) ambil tangan pertama -> fitur

            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:  # (keterangan) append ke CSV
                csv.writer(f).writerow([folder.name.upper()] + feat.tolist())  # (keterangan) tulis label + fitur

            total_written += 1  # (keterangan) tambah total sukses
            wrote_this += 1  # (keterangan) tambah total kelas

        print(f"[{folder.name.upper()}] wrote={wrote_this}/{len(imgs)}")  # (keterangan) ringkas per kelas

print("\n✅ Selesai ekstraksi!")  # (keterangan) info
print("CSV:", OUT_CSV)  # (keterangan) lokasi output
print("Total ditulis:", total_written)  # (keterangan) harus >= 100 (syarat tugas)
print("Skip (no hand):", total_nohand)  # (keterangan) statistik gambar tanpa tangan


#.\.venv\Scripts\python.exe .\app\00_extract_landmarks_from_asl_alphabet.py