"""
train.py — AutoML Training Pipeline via PyCaret
================================================
Setup PyCaret, compare models, finalize, dan save.
Dioptimalkan untuk memproses maksimal 30.000 data.
"""

import os
import warnings
import pandas as pd
import logging

from config import (
    LABEL_COL,
    MODEL_DIR,
    SESSION_ID,
    TRAIN_SIZE,
    N_TOP_MODELS,
)

# Suppress warnings agar output terminal lebih bersih
warnings.filterwarnings("ignore")

# Mematikan log LightGBM secara standar (tanpa hack/monkey patch yang bikin error)
os.environ["LGBM_WARNING"] = "0"
logging.getLogger("lightgbm").setLevel(logging.ERROR)


# ══════════════════════════════════════════════
# ⚙️ SETUP PYCARET
# ══════════════════════════════════════════════

def setup_pycaret(df: pd.DataFrame):
    from pycaret.classification import setup

    print("\n⚙️  Menginisialisasi PyCaret...")
    print(f"   Kolom teks  : cleaned_text")
    print(f"   Kolom label : {LABEL_COL}")

    # Hanya ambil kolom yang diperlukan untuk menghemat RAM
    df_model = df[["cleaned_text", LABEL_COL]].copy()

    s = setup(
        data=df_model,
        target=LABEL_COL,
        text_features=["cleaned_text"],
        session_id=SESSION_ID,
        train_size=TRAIN_SIZE,
        verbose=True,
        html=False, # Set False agar lancar di terminal VS Code
        use_gpu=False,
        fold=3, # Diubah ke 3 agar proses validasi jauh lebih cepat
    )
    print("✅ PyCaret setup selesai!")
    return s


# ══════════════════════════════════════════════
# 🏟️ MODEL ARENA — COMPARE MODELS
# ══════════════════════════════════════════════

def compare_all_models(sort: str = "F1", n_select: int = None):
    from pycaret.classification import compare_models
    if n_select is None:
        n_select = N_TOP_MODELS

    print(f"\n🏟️  Memulai Model Arena (sort by {sort})...")
    print("   Proses ini akan memakan waktu beberapa menit. Harap tunggu...")
    
    # Mengecualikan model XGBoost karena sering memakan waktu sangat lama tanpa GPU
    best = compare_models(sort=sort, n_select=n_select, exclude=['xgboost'])
    
    print(f"✅ Selesai! Top {n_select} model telah dipilih.")
    return best


# ══════════════════════════════════════════════
# 💾 FINALIZE & EXPORT
# ══════════════════════════════════════════════

def finalize_and_save(model, filename: "nlp_pipeline_final"):
    from pycaret.classification import finalize_model, save_model
    
    print("\n💾 Memfinalisasi model (retrain pada seluruh data)...")
    final = finalize_model(model)

    save_path = os.path.join(MODEL_DIR, filename)
    save_model(final, save_path)
    print(f"✅ Model berhasil disimpan di: {save_path}.pkl")
    return final


# ──────────────────────────────────────────────
# Eksekusi Script Utama
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import load_and_clean
    
    target_path = os.path.join("data", "dataset.csv")
    
    try:
        # 1. Baca & Bersihkan Data (Menggunakan fungsi yang sudah dioptimasi di preprocess.py)
        df = load_and_clean(csv_path=target_path, limit=30000)
        
        # 2. Safety Check Ekstra: Pastikan data benar-benar maksimal 30.000
        if len(df) > 30000:
            print("\n--- Memotong data menjadi 30.000 sampel agar RAM aman ---")
            df = df.sample(n=30000, random_state=SESSION_ID).reset_index(drop=True)
        
        # 3. Setup PyCaret
        setup_pycaret(df)
        
        # 4. Cari Model Terbaik (Compare Models)
        best_models = compare_all_models()
        best = best_models[0] if isinstance(best_models, list) else best_models
        
        # 5. Finalisasi dan Simpan Model sebagai file .pkl
        finalize_and_save(best, "nlp_pipeline_final")
        
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat training: {str(e)}")