"""
==============================================================================
 PREDIKSI CHURN NASABAH PERBANKAN MENGGUNAKAN RANDOM FOREST
 DENGAN PENDEKATAN EXPLAINABLE AI (SHAP)
 Studi Kasus: Dataset Bank Churn Kaggle

 Author : Doli Marolop Saing
 Affil. : Program Studi Sistem Informasi Bisnis,
          Akademi Manajemen Informatika dan Komputer Universal

 Script ini mereproduksi SELURUH hasil pada naskah, mulai dari:
   1. Pra-pemrosesan data
   2. Model Random Forest baseline (hyperparameter manual)
   3. Hyperparameter tuning (RandomizedSearchCV) -> RF Tuned
   4. Penyeimbangan data dengan SMOTE -> RF SMOTE Tuned
   5. Perbandingan dengan XGBoost, LightGBM, CatBoost
   6. Evaluasi (Tabel perbandingan model pada naskah)
   7. Interpretasi SHAP global (Gambar 4 & 5)
   8. Interpretasi SHAP lokal / waterfall plot (Gambar 6, 7, 8)

 REVISI REVIEWER (poin 1: kontribusi & ukuran kelompok 3-4 produk):
   - Menghitung sebaran NumOfProducts pada seluruh dataset dan data uji
     (angka n, persentase, dan tingkat churn per kelompok)
   - Meringkas nilai SHAP fitur Jumlah Produk per kelompok pada data uji
   Bagian yang ditambahkan ditandai dengan komentar "REVISI REVIEWER".

 Menjalankan skrip ini akan menghasilkan ulang tabel dan Gambar 2-8.
 Gambar 1 (kurva Precision-Recall) dibuat oleh analisis_tambahan.py.

 Cara menjalankan:
     pip install -r requirements.txt
     python src/churn_prediction_pipeline.py
==============================================================================
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)
plt.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
from data_loader import ensure_dataset
DATA_PATH = ensure_dataset(ROOT)
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

RENAME_MAP = {
    "CreditScore": "Skor Kredit", "Age": "Usia", "Tenure": "Lama Menjadi Nasabah",
    "Balance": "Saldo Rekening", "NumOfProducts": "Jumlah Produk",
    "HasCrCard": "Punya Kartu Kredit", "IsActiveMember": "Anggota Aktif",
    "EstimatedSalary": "Estimasi Gaji", "Geography_Germany": "Negara: Jerman",
    "Geography_Spain": "Negara: Spanyol", "Gender_Male": "Jenis Kelamin: Pria",
}


# --------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING  (Bab 2.1 - 2.2 pada naskah)
# --------------------------------------------------------------------------
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset asli: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("Distribusi kelas Exited:\n", df["Exited"].value_counts(normalize=True))

    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # ----------------------------------------------------------------------
    # REVISI REVIEWER (1/3): sebaran NumOfProducts pada SELURUH dataset
    # Bahan untuk [X] (n kelompok 3-4 produk), [Y] (% dari 10.000), dan
    # tabel sebaran jumlah produk di Bab II.A.
    # Dihitung SEBELUM get_dummies agar kolom NumOfProducts masih utuh.
    # ----------------------------------------------------------------------
    prod_all = (
        df.groupby("NumOfProducts")["Exited"]
        .agg(jumlah="count", churn="sum", tingkat_churn="mean")
    )
    prod_all["persen_dataset"] = prod_all["jumlah"] / len(df) * 100
    prod_all["tingkat_churn"] = prod_all["tingkat_churn"] * 100
    print("\n[REVISI] Sebaran NumOfProducts (seluruh dataset, n = "
          f"{len(df)}):\n{prod_all.round(2)}")

    n_34 = int(df["NumOfProducts"].isin([3, 4]).sum())
    churn_34 = float(df.loc[df["NumOfProducts"].isin([3, 4]), "Exited"].mean() * 100)
    print(f"[REVISI] Kelompok 3-4 produk: n = {n_34} "
          f"({n_34 / len(df) * 100:.2f}% dari dataset), "
          f"tingkat churn = {churn_34:.2f}%")
    prod_all.round(2).to_csv(RESULTS_DIR / "sebaran_numofproducts.csv")

    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")

    # ----------------------------------------------------------------------
    # REVISI REVIEWER (2/3): sebaran NumOfProducts pada DATA UJI
    # Bahan untuk [Z] (n kelompok 3-4 produk pada data uji, dari 2.000).
    # ----------------------------------------------------------------------
    prod_test = X_test["NumOfProducts"].value_counts().sort_index()
    n_34_test = int(X_test["NumOfProducts"].isin([3, 4]).sum())
    print(f"\n[REVISI] Sebaran NumOfProducts pada data uji:\n{prod_test}")
    print(f"[REVISI] Kelompok 3-4 produk pada data uji: n = {n_34_test} "
          f"dari {len(X_test)}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Helper evaluasi
# --------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, results):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    m = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    results[name] = m
    print(f"\n=== {name} ===")
    for k, v in m.items():
        print(f"  {k:<10}: {v:.4f}")
    return m


# --------------------------------------------------------------------------
# 2-5. MODELING: BASELINE, TUNING, SMOTE, BOOSTING COMPARISON (Bab 2.3-2.6)
# --------------------------------------------------------------------------
def train_all_models(X_train, X_test, y_train, y_test):
    results = {}

    # --- RF Baseline (hyperparameter manual) ---
    rf_baseline = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    rf_baseline.fit(X_train, y_train)
    evaluate("RF_Baseline", rf_baseline, X_test, y_test, results)

    # --- RF Tuned (RandomizedSearchCV, dioptimasi untuk F1-score) ---
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        param_distributions=param_dist, n_iter=12, scoring="f1",
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        random_state=42, n_jobs=1, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    print("\nHyperparameter terbaik (RF Tuned):", rf_search.best_params_)
    rf_tuned = rf_search.best_estimator_
    evaluate("RF_Tuned", rf_tuned, X_test, y_test, results)

    # --- RF + SMOTE (Tuned) ---
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"\nSMOTE: {len(X_train)} -> {len(X_train_sm)} data latih "
          f"(distribusi kelas: {y_train_sm.value_counts().to_dict()})")
    rf_smote = RandomForestClassifier(**rf_search.best_params_, random_state=42, n_jobs=1)
    rf_smote.fit(X_train_sm, y_train_sm)
    evaluate("RF_SMOTE_Tuned", rf_smote, X_test, y_test, results)

    # --- XGBoost ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss", n_jobs=1,
    )
    xgb.fit(X_train, y_train)
    evaluate("XGBoost", xgb, X_test, y_test, results)

    # --- LightGBM ---
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    evaluate("LightGBM", lgbm, X_test, y_test, results)

    # --- CatBoost ---
    cat = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_state=42, verbose=0,
    )
    cat.fit(X_train, y_train)
    evaluate("CatBoost", cat, X_test, y_test, results)

    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"results": results, "best_rf_params": rf_search.best_params_}, f, indent=2)

    print("\n=== PERBANDINGAN ENAM MODEL ===")
    print(f"{'Model':<20}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
    for name, m in results.items():
        print(f"{name:<20}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['roc_auc']:>8.3f}")

    return rf_tuned, results


# --------------------------------------------------------------------------
# 6. EVALUASI VISUAL: CONFUSION MATRIX & ROC (Gambar 2 & 3)
# --------------------------------------------------------------------------
def plot_confusion_and_roc(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Tidak Churn", "Churn"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix\nRandom Forest (Tuned)", fontsize=13)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gambar2_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"Random Forest (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC - Random Forest (Tuned)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gambar3_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix:\n{cm}")
    print(f"AUC-ROC: {auc_val:.4f}")


# --------------------------------------------------------------------------
# 7-8. INTERPRETASI SHAP GLOBAL & LOKAL (Gambar 4-8)
# --------------------------------------------------------------------------
def shap_analysis(model, X_test, y_test):
    X_test_renamed = X_test.rename(columns=RENAME_MAP)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        sv = shap_values[1]
        base_val = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        sv = shap_values
        base_val = explainer.expected_value

    # --- Global: summary plot (Gambar 4) ---
    plt.figure(figsize=(7.5, 5.5))
    shap.summary_plot(sv, X_test_renamed, show=False, max_display=11)
    plt.title("SHAP Summary Plot - Random Forest (Tuned)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gambar4_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Global: bar chart (Gambar 5) ---
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    feat_sorted = X_test_renamed.columns[order]
    val_sorted = mean_abs[order]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.barh(feat_sorted[::-1], val_sorted[::-1], color="#2563eb")
    ax.set_xlabel("Rata-rata |Nilai SHAP|")
    ax.set_title("Peringkat Kepentingan Fitur (SHAP)\nRandom Forest (Tuned)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gambar5_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nPeringkat SHAP global:")
    for f, v in zip(feat_sorted, val_sorted):
        print(f"  {f:<25}: {v:.4f}")

    # ----------------------------------------------------------------------
    # REVISI REVIEWER (3/3): ringkasan nilai SHAP fitur Jumlah Produk
    # per kelompok pada data uji. Mendukung (dan menguji kehati-hatian)
    # klaim pola non-linear pada Bab III.C: tampilkan n per kelompok agar
    # jelas kelompok mana yang kecil.
    # ----------------------------------------------------------------------
    col_idx = list(X_test.columns).index("NumOfProducts")
    prod_shap = pd.DataFrame({
        "NumOfProducts": X_test["NumOfProducts"].values,
        "shap_jumlah_produk": sv[:, col_idx],
    })
    ringkasan = (
        prod_shap.groupby("NumOfProducts")["shap_jumlah_produk"]
        .agg(n="count", rata_rata="mean", minimum="min", maksimum="max")
    )
    print("\n[REVISI] Nilai SHAP fitur Jumlah Produk per kelompok (data uji):")
    print(ringkasan.round(4))
    ringkasan.round(4).to_csv(RESULTS_DIR / "shap_jumlah_produk_per_kelompok.csv")

    # --- Lokal: pilih 3 sampel representatif (Gambar 6, 7, 8) ---
    y_pred = model.predict(X_test)
    y_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    pred_s = pd.Series(y_pred, index=X_test.index)

    tp_idx = X_test.index[(pred_s == 1) & (y_test == 1)]
    tn_idx = X_test.index[(pred_s == 0) & (y_test == 0)]
    fp_idx = X_test.index[(pred_s == 1) & (y_test == 0)]

    samples = {
        "gambar6_waterfall_high_risk_TP": y_proba.loc[tp_idx].idxmax(),
        "gambar7_waterfall_loyal_TN": y_proba.loc[tn_idx].idxmin(),
        "gambar8_waterfall_false_positive": y_proba.loc[fp_idx].idxmax(),
    }

    for fname, idx in samples.items():
        pos = X_test.index.get_loc(idx)
        exp = shap.Explanation(
            values=sv[pos], base_values=base_val,
            data=X_test_renamed.loc[idx].values,
            feature_names=list(X_test_renamed.columns),
        )
        plt.figure(figsize=(8, 5.2))
        shap.plots.waterfall(exp, show=False, max_display=11)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n[{fname}] idx={idx} | aktual={y_test.loc[idx]} | "
              f"prediksi={y_pred[pos]} | probabilitas={y_proba.loc[idx]:.3f}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print(" REPRODUKSI HASIL: Prediksi Churn Nasabah Perbankan - RF + SHAP")
    print("=" * 78)

    X_train, X_test, y_train, y_test = load_and_preprocess()
    rf_tuned, results = train_all_models(X_train, X_test, y_train, y_test)
    plot_confusion_and_roc(rf_tuned, X_test, y_test)
    shap_analysis(rf_tuned, X_test, y_test)

    print(f"\nSemua figur tersimpan di: {FIG_DIR}")
    print(f"Tabel hasil tersimpan di: {RESULTS_DIR / 'model_comparison.json'}")
    print(f"Sebaran jumlah produk  : {RESULTS_DIR / 'sebaran_numofproducts.csv'}")
    print(f"SHAP per kelompok produk: {RESULTS_DIR / 'shap_jumlah_produk_per_kelompok.csv'}")
    print("\nSELESAI. Bandingkan dengan tabel dan Gambar 1-8 pada naskah.")