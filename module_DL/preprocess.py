"""
preprocess.py — Pipeline Pembersihan Teks
==========================================
Terkoneksi dengan config.py untuk menangani dataset besar (2GB).
Update: Mapping label disesuaikan dengan nilai dataset (-1=Negative, 1=Positive)
"""

import re
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 🔗 MENGHUBUNGKAN KE CONFIG (PENTING!)
try:
    from config import (
        DATA_PATH, LABEL_LIST, RANDOM_SEED, 
        LABEL_ENCODER_PATH, SAMPLE_SIZE
    )
    # Mapping kolom (sesuaikan dengan isi dataset.csv kamu)
    TEXT_COL = "review_text"   
    LABEL_COL = "review_score" 
    RAW_CSV = DATA_PATH # Mengambil dari config
except ImportError:
    # Fallback jika config tidak terbaca
    TEXT_COL = "review_text"
    LABEL_COL = "review_score"
    LABEL_LIST = ["Negative", "Positive"]
    RANDOM_SEED = 42
    LABEL_ENCODER_PATH = "models/label_encoder.json"
    RAW_CSV = "data/dataset.csv"
    SAMPLE_SIZE = 100000

# ──────────────────────────────────────────────
# 🧹 FUNGSI PEMBERSIHAN
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────
# 📥 LOAD & CLEAN
# ──────────────────────────────────────────────

def load_and_clean(csv_path: str, sample_size: int | None = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {csv_path}")

    print(f"📂 Membaca dataset: {csv_path}")
    
    # Tips: Untuk file 2GB, kita hanya baca kolom yang perlu saja agar hemat RAM
    df = pd.read_csv(csv_path, usecols=[TEXT_COL, LABEL_COL])

    # Normalisasi Nama Kolom
    df.columns = df.columns.str.strip().str.lower()
    t_col = TEXT_COL.lower()
    l_col = LABEL_COL.lower()
    
    df = df[[t_col, l_col]].copy()
    df.rename(columns={t_col: "text", l_col: "label"}, inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    
    # --- PERBAIKAN MAPPING: -1=Negative, 1=Positive ---
    id_to_label = {"-1": "Negative", "1": "Positive"}
    df["label"] = df["label"].astype(float).astype(int).astype(str).map(id_to_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    print(f"📊 Distribusi kelas ditemukan:\n{df['label'].value_counts().to_string()}\n")

    # ── Sampling stratified agar tidak crash di RAM ──
    if sample_size and sample_size < len(df):
        print(f"🔀 Mengambil sampel {sample_size:,} baris secara seimbang...")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)

    # ── Pembersihan teks ──
    print("🧹 Membersihkan teks...")
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    # ── Encode label ──
    le = LabelEncoder()
    le.fit(LABEL_LIST)
    df["label_encoded"] = le.transform(df["label"])

    if not os.path.exists("models"):
        os.makedirs("models")
        
    label_mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
    with open(LABEL_ENCODER_PATH, "w") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"✅ Selesai. {len(df):,} baris siap digunakan.")
    return df

def show_cleaning_examples(df: pd.DataFrame, n: int = 5) -> None:
    print(f"{'='*70}\nCONTOH PEMBERSIHAN TEKS\n{'='*70}")
    n_samples = min(n, len(df))
    for i, row in df.sample(n=n_samples, random_state=RANDOM_SEED).iterrows():
        print(f"\n[{row['label']}]")
        print(f"  Asli    : {row['text'][:120]}...")
        print(f"  Bersih  : {row['cleaned_text'][:120]}...")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    try:
        # Menggunakan nilai dari config
        processed_df = load_and_clean(RAW_CSV, SAMPLE_SIZE)
        show_cleaning_examples(processed_df)
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")