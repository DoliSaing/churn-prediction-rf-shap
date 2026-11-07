# Prediksi Churn Nasabah Perbankan Menggunakan Random Forest dengan Pendekatan Explainable AI (SHAP)

Repositori ini menyediakan **kode reproduksi lengkap** untuk naskah:

> Saing, D. M. (2026). *Prediksi Churn Nasabah Perbankan Menggunakan Random Forest dengan Pendekatan Explainable AI (SHAP): Studi Kasus Dataset Bank Churn Kaggle*. Diajukan ke JOINS (Journal of Information System).

Seluruh tabel dan gambar pada naskah (Tabel 1; Gambar 1–8) dapat direproduksi ulang dengan menjalankan skrip utama dan skrip analisis tambahan, karena seluruh proses (split data, hyperparameter tuning, SMOTE, dan pelatihan model) menggunakan `random_state=42` yang tetap (fixed seed).

## 📁 Struktur Repositori

```
churn-prediction-rf-shap/
├── data/
│   └── Churn_Modelling.csv      # Dataset (lihat sumber di bawah)
├── src/
│   └── churn_prediction_pipeline.py   # Skrip utama (end-to-end)
├── figures/                      # Output: 8 gambar (dibuat otomatis saat dijalankan)
├── results/                      # Output: tabel hasil dalam format JSON
├── requirements.txt
└── README.md
```

## 📊 Sumber Dataset

Dataset **"Bank Customer Churn (Churn_Modelling)"** bersumber dari [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling), diunggah oleh Shruti Iyyer, dan sebelumnya dipublikasikan oleh SuperDataScience Team sebagai dataset template untuk pelatihan Artificial Neural Network. Dataset berisi 10.000 baris data nasabah dengan 14 atribut awal (lihat naskah Bab 2.1 untuk detail lengkap).

> **Catatan lisensi:** Status lisensi distribusi ulang dataset ini tidak dapat dipastikan secara eksplisit dari halaman Kaggle saat repositori ini disiapkan. Karena itu, **file CSV tidak disertakan langsung** dalam repositori ini. Silakan unduh sendiri dari tautan Kaggle di atas (memerlukan akun Kaggle), lalu letakkan file `Churn_Modelling.csv` ke dalam folder `data/` sebelum menjalankan skrip. Mohon periksa kembali ketentuan penggunaan/lisensi pada halaman dataset Kaggle sebelum menggunakannya untuk publikasi.

## ⚙️ Instalasi

```bash
git clone <URL_REPOSITORI_INI>
cd churn-prediction-rf-shap
pip install -r requirements.txt
```

Disarankan menggunakan virtual environment (`venv` atau `conda`) untuk menghindari konflik versi pustaka.

## ▶️ Menjalankan Reproduksi

```bash
python src/churn_prediction_pipeline.py
```

Skrip ini akan secara berurutan:

| Langkah | Bagian Naskah | Output |
|---|---|---|
| 1. Pra-pemrosesan data | Bab 2.1–2.2 | — |
| 2. Random Forest Baseline | Bab 2.3 | Baris 1, Tabel 1 |
| 3. Hyperparameter Tuning (RandomizedSearchCV) | Bab 2.4 | Baris 2, Tabel 1 |
| 4. Random Forest + SMOTE | Bab 2.4 | Baris 3, Tabel 1 |
| 5. XGBoost, LightGBM, CatBoost | Bab 2.5 | Baris 4–6, Tabel 1 |
| 6. Evaluasi (Confusion Matrix, ROC) | Bab 2.6, 3.2 | `figures/gambar2_*.png`, `gambar3_*.png` |
| 7. SHAP Global | Bab 2.7, 3.3 | `figures/gambar4_*.png`, `gambar5_*.png` |
| 8. SHAP Lokal (Waterfall) | Bab 2.7, 3.4 | `figures/gambar6_*.png` s.d. `gambar8_*.png` |

Estimasi waktu eksekusi: ±2–4 menit pada CPU standar (1 core).

## ✅ Verifikasi Hasil

Jalankan skrip dan bandingkan output tabel di terminal dengan Tabel 1 pada naskah:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest (Baseline) | 0,837 | 0,591 | 0,644 | 0,616 | 0,863 |
| **Random Forest (Tuned)** | 0,826 | 0,555 | 0,720 | 0,627 | 0,866 |
| Random Forest + SMOTE (Tuned) | 0,812 | 0,531 | 0,658 | 0,588 | 0,843 |
| XGBoost | 0,814 | 0,532 | 0,695 | 0,603 | 0,850 |
| LightGBM | 0,810 | 0,526 | 0,700 | 0,601 | 0,855 |
| CatBoost | 0,812 | 0,528 | 0,747 | 0,619 | 0,867 |

Model **Random Forest (Tuned)** adalah model utama yang digunakan pada seluruh analisis SHAP (global dan lokal) di naskah.

## 🔍 Tentang Interpretasi SHAP Lokal

Tiga sampel representatif dipilih secara otomatis dan deterministik (bukan dipilih manual):

1. **Nasabah berisiko churn tinggi** — true positive dengan probabilitas churn tertinggi
2. **Nasabah loyal** — true negative dengan probabilitas churn terendah
3. **Kasus false positive** — prediksi churn salah dengan probabilitas tertinggi (paling "percaya diri tapi salah")

## 📦 Pustaka yang Digunakan

scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, CatBoost, SHAP — lihat `requirements.txt` untuk versi minimum.

## 📄 Lisensi

Kode pada repositori ini disediakan untuk tujuan transparansi dan reproduktivitas ilmiah, dapat digunakan secara bebas untuk keperluan akademik dengan menyertakan sitasi ke naskah terkait.

## ✉️ Kontak

Doli Marolop Saing — Program Studi Sistem Informasi Bisnis, Akademi Manajemen Informatika dan Komputer Universal
Email: dolymarolop@gmail.com

## 🔁 Analisis Tambahan untuk Revisi Naskah

Dua skrip tambahan dijalankan **setelah** `churn_prediction_pipeline.py` (membaca `results/model_comparison.json`):

```bash
python src/churn_prediction_pipeline.py   # Tabel hasil utama dan Gambar 2-8
python src/analisis_tambahan.py           # CI bootstrap, CV berulang 5x3, pembanding sederhana, PR-AUC, ambang, audit subkelompok, Gambar 1
python src/tuning_boosting.py             # tuning setara (12 kombinasi, 3-fold, F1) untuk XGBoost, LightGBM, CatBoost
```

| Skrip | Keluaran | Bagian naskah revisi |
|---|---|---|
| `analisis_tambahan.py` | `results/analisis_tambahan.json`, `figures/gambar1_pr_curve.png` | Tabel 4a, 4b, 5, 6; Gambar 1 |
| `tuning_boosting.py` | `results/tuning_boosting.json` | Tabel 7 |
| `churn_prediction_pipeline.py` | `results/sebaran_numofproducts.csv`, `results/shap_jumlah_produk_per_kelompok.csv`, `figures/gambar2_*.png`–`gambar8_*.png` | Tabel 2; SHAP per jumlah produk; Gambar 2–8 |

Semua skrip memakai `random_state = 42` dan sudah dijalankan ulang dua kali dengan hasil identik (scikit-learn 1.8.0, shap 0.54.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, imbalanced-learn 0.14.2).

**Catatan interpretasi:** RF *Tuned* dipilih karena fokus penelitian pada *Random Forest*, bukan karena unggul secara statistik. Skor model belum dikalibrasi dan nilai SHAP menjelaskan model, bukan sebab-akibat.

## ⚠️ Versi Pustaka dan Reproduktivitas

Angka pada naskah revisi dihasilkan dengan **scikit-learn 1.9.1** (lihat `requirements.txt`). Versi scikit-learn lain (mis. 1.8.0) menghasilkan model *Random Forest* yang sedikit berbeda meskipun `random_state = 42` sama; XGBoost, LightGBM, dan CatBoost tidak terpengaruh. Hasil utama untuk verifikasi: RF *Baseline* F1 0,631 / AUC-ROC 0,864; RF *Tuned* F1 0,623 / AUC-ROC 0,860 dengan parameter terbaik `n_estimators` 200, `max_depth` 12, `min_samples_split` 5, `min_samples_leaf` 2, `max_features` log2.

## 📥 Dataset (unduhan otomatis)

Berkas `Churn_Modelling.csv` **tidak disertakan** pada repositori ini. Saat skrip dijalankan dan berkas belum ada di `data/`, skrip mencoba mengunduhnya dari Kaggle lewat `kagglehub` (lihat `src/data_loader.py`). Unduhan ini membutuhkan akun Kaggle:

```bash
pip install -r requirements.txt
# atur kredensial Kaggle (Settings > API di Kaggle), lalu salah satu:
set KAGGLE_USERNAME=<username>      # Windows (di Linux/macOS: export)
set KAGGLE_KEY=<key>
# atau:  python -c "import kagglehub; kagglehub.login()"
python src/churn_prediction_pipeline.py
```

Jika unduhan otomatis gagal, unduh berkas secara manual dari https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling dan letakkan di `data/Churn_Modelling.csv`. Mohon periksa ketentuan lisensi pada halaman dataset sebelum menggunakannya.
