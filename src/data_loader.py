"""Unduh otomatis dataset Bank Customer Churn dari Kaggle jika belum ada di data/.

Berkas CSV sengaja TIDAK disertakan pada repositori karena status lisensi distribusi ulangnya
tidak dapat dipastikan. Dataset: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
"""
import shutil
import sys
from pathlib import Path

KAGGLE_HANDLE = "shrutimechlearn/churn-modelling"
FILENAME = "Churn_Modelling.csv"

_PETUNJUK = f"""
Dataset tidak ditemukan dan unduhan otomatis gagal.

Cara manual:
  1. Unduh {FILENAME} dari https://www.kaggle.com/datasets/{KAGGLE_HANDLE} (perlu akun Kaggle).
  2. Letakkan berkas di folder data/ pada proyek ini.
  3. Jalankan ulang skrip.

Cara otomatis (butuh akun Kaggle):
  pip install kagglehub
  Buat token di Kaggle (Settings > API), lalu atur variabel lingkungan
  KAGGLE_USERNAME dan KAGGLE_KEY, atau jalankan:  python -c "import kagglehub; kagglehub.login()"
  Setelah itu jalankan ulang skrip.
"""


def _gagal(alasan: str):
    print(f"[DATA] {alasan}")
    print(_PETUNJUK)
    sys.exit(1)


def ensure_dataset(root) -> Path:
    """Pastikan data/Churn_Modelling.csv ada; jika tidak, coba unduh dari Kaggle."""
    target = Path(root) / "data" / FILENAME
    if target.exists():
        return target
    try:
        import kagglehub
    except ImportError:
        _gagal("Pustaka 'kagglehub' belum terpasang (pip install kagglehub).")
    try:
        folder = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
        sumber = next(folder.rglob(FILENAME))
    except Exception as e:  # jaringan, autentikasi, atau nama berkas berubah
        _gagal(f"Unduhan dari Kaggle gagal: {type(e).__name__}: {e}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sumber, target)
    print(f"[DATA] Dataset diunduh ke {target}")
    return target
