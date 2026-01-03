from pathlib import Path  # () path aman
import joblib  # (keterangan) simpan/load model
import pandas as pd  # (keterangan) baca CSV
import numpy as np  # (keterangan) array numerik

import matplotlib.pyplot as plt  # (keterangan) visualisasi

try:  # (keterangan) supaya display() bisa jalan di notebook, tapi tetap aman di .py biasa
    from IPython.display import display  # (keterangan) display DataFrame (Jupyter)
except Exception:  # (keterangan) fallback jika IPython tidak ada
    display = None  # (keterangan) display tidak tersedia

from sklearn.model_selection import train_test_split  # (keterangan) split data
from sklearn.preprocessing import StandardScaler, LabelEncoder  # (keterangan) scaling & encode label
from sklearn.pipeline import Pipeline  # (keterangan) pipeline preprocessing+model
from sklearn.neighbors import KNeighborsClassifier  # (keterangan) model KNN :contentReference[oaicite:5]{index=5}
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix  # (keterangan) metrik evaluasi
from sklearn.metrics import classification_report  # <-- TAMBAHAN (kalau belum)

SCRIPT_PATH = Path(__file__).resolve()  # (keterangan) lokasi script
ROOT = SCRIPT_PATH.parents[1]  # (keterangan) root project

CSV_PATH = ROOT / "data" / "landmarks" / "asl_alphabet_landmarks.csv"  # (keterangan) input CSV
MODEL_DIR = ROOT / "models"  # (keterangan) output folder model

if not CSV_PATH.exists():  # (keterangan) jika CSV belum ada
    raise FileNotFoundError(f"CSV tidak ditemukan: {CSV_PATH}\nJalankan 00_extract... dulu.")  # (keterangan) stop

df = pd.read_csv(CSV_PATH)  # (keterangan) baca data
if len(df) == 0:  # (keterangan) jika kosong
    raise RuntimeError("CSV kosong (n_samples=0). Pastikan ekstraksi tidak 0.")  # (keterangan) stop

# ====== X dan y ======
X = df.drop(columns=["label"]).values.astype(np.float32)  # (keterangan) fitur 63 kolom
y = df["label"].values  # (keterangan) label huruf

# ====== Encode label: huruf -> angka ======
le = LabelEncoder()  # (keterangan) encoder label
y_enc = le.fit_transform(y)  # (keterangan) huruf -> integer

# ====== Split train/test ======
X_train, X_test, y_train, y_test = train_test_split(  # (keterangan) split train/test
    X,  # (keterangan) fitur
    y_enc,  # (keterangan) label
    test_size=0.2,  # (keterangan) 20% test
    random_state=42,  # (keterangan) reproducible
    stratify=y_enc,  # (keterangan) menjaga proporsi kelas
)  # (keterangan) selesai split

# (keterangan) pipeline: scaling -> KNN
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),  # (keterangan) standardisasi fitur
        ("knn", KNeighborsClassifier(  # (keterangan) classifier KNN :contentReference[oaicite:6]{index=6}
            n_neighbors=7,  # (keterangan) K
            weights="distance",  # (keterangan) bobot berdasarkan jarak (lebih stabil untuk gesture)
            metric="euclidean",  # (keterangan) jarak euclidean
        )),
    ]
)

# ====== Training ======
model.fit(X_train, y_train)  # (keterangan) training

# ====== Prediksi ======
pred = model.predict(X_test)  # (keterangan) prediksi

# ====== Evaluasi ======
acc = accuracy_score(y_test, pred)  # (keterangan) accuracy
prec, rec, f1, _ = precision_recall_fscore_support(  # (keterangan) precision/recall/f1
    y_test,  # (keterangan) label asli
    pred,  # (keterangan) label pred
    average="macro",  # (keterangan) macro average
    zero_division=0,  # (keterangan) aman jika ada pembagian nol
)
cm = confusion_matrix(y_test, pred)  # (keterangan) confusion matrix

print("Accuracy :", acc)  # (keterangan) tampilkan
print("Precision:", prec)  # (keterangan) tampilkan
print("Recall   :", rec)  # (keterangan) tampilkan
print("F1       :", f1)  # (keterangan) tampilkan
print("Confusion Matrix:\n", cm)  # (keterangan) tampilkan

MODEL_DIR.mkdir(exist_ok=True)  # (keterangan) pastikan folder model ada
joblib.dump(model, MODEL_DIR / "knn_landmarks.joblib")  # (keterangan) simpan model
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")  # (keterangan) simpan encoder
print("\n✅ Saved:", MODEL_DIR / "knn_landmarks.joblib")  # (keterangan) info
print("✅ Saved:", MODEL_DIR / "label_encoder.joblib")  # (keterangan) info

# ====== Visualisasi Confusion Matrix (rapi + biru + teks kontras) ======
labels = le.classes_                 # nama kelas asli (A, B, C, ...)
label_ids = np.arange(len(labels))   # id kelas 0..n-1

# Recompute CM khusus untuk plot agar ukuran selalu konsisten (meski ada kelas yang tidak muncul di test)
cm_plot = confusion_matrix(y_test, pred, labels=label_ids)

cm_plot_f = cm_plot.astype(float)
row_sum = cm_plot_f.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm_plot_f, row_sum, out=np.zeros_like(cm_plot_f), where=row_sum != 0)  # aman dari pembagian 0

fig, ax = plt.subplots(figsize=(14, 12))  # <-- lebih rapi untuk banyak kelas
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")  # <-- biru
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Proporsi (per kelas asli)", rotation=90)

ax.set(
    xticks=np.arange(len(labels)),
    yticks=np.arange(len(labels)),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted label",
    ylabel="True label",
    title="Confusion Matrix (Count + % per True Class)"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.tick_params(axis="both", labelsize=9)

# Tulis Count + Persentase di tiap sel (warna teks otomatis biar kebaca)
thr = 0.5  # ambang gelap/terang (0–1). coba 0.4 kalau mau lebih banyak putih
fs = 7 if len(labels) > 20 else 9

for i in range(cm_plot.shape[0]):
    for j in range(cm_plot.shape[1]):
        count = int(cm_plot[i, j])
        pct = cm_norm[i, j] * 100
        txt_color = "white" if cm_norm[i, j] >= thr else "black"

        ax.text(
            j, i,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="center",
            fontsize=fs,
            color=txt_color,
            fontweight="bold"
        )

ax.set_ylim(len(labels) - 0.5, -0.5)  # bugfix umum di matplotlib
ax.grid(False)
plt.tight_layout()
plt.show()

# ====== Classification Report per Kelas ======
report_text = classification_report(
    y_test, pred,
    labels=label_ids,
    target_names=labels,
    zero_division=0
)
print("🔍 Classification Report per Kelas:")
print(report_text)

# ====== Classification Report per Kelas (DataFrame) ======
report_dict = classification_report(
    y_test, pred,
    labels=label_ids,
    target_names=labels,
    zero_division=0,
    output_dict=True
)

df_report = pd.DataFrame(report_dict).T
if display is not None:  # (keterangan) jika di notebook, tampilkan tabel cantik
    display(df_report)
else:  # (keterangan) jika di terminal, tetap tampilkan
    print(df_report)

#python .\app\01_train_knn_landmarks_from_csv.py
