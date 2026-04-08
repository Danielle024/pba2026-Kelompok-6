"""
preprocess.py — Custom Text Cleaning untuk Chat Gamer Indonesia
================================================================
Pipeline: lowercase -> hapus URL/mention -> normalisasi leetspeak ->
ekspansi slang -> hapus karakter non-alfabet -> strip whitespace.
Optimasi: Hanya mengambil 30,000 baris pertama untuk kecepatan.
"""

import re
import os
import pandas as pd
from config import LEETSPEAK_MAP, SLANG_DICT, TEXT_COL, LABEL_COL, RAW_CSV


# ==============================================
# FUNGSI PEMBANTU (HELPERS)
# ==============================================

def normalize_leetspeak(text: str) -> str:
    """Konversi angka/simbol leetspeak ke huruf biasa."""
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            # Cek apakah ada huruf di sekitar (sebelum atau sesudah)
            prev_is_alpha = (i > 0 and text[i - 1].isalpha())
            next_is_alpha = (i < len(text) - 1 and text[i + 1].isalpha())

            if prev_is_alpha or next_is_alpha:
                result.append(LEETSPEAK_MAP[char])
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def expand_slang(text: str) -> str:
    """Ekspansi singkatan & slang gamer ke bentuk lengkap."""
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """Pipeline pembersihan teks lengkap."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Hapus mention
    text = re.sub(r"@\w+", "", text)

    # 4. Normalisasi leetspeak
    text = normalize_leetspeak(text)

    # 5. Ekspansi slang
    text = expand_slang(text)

    # 6. Hapus karakter non-alfabet (kecuali spasi)
    text = re.sub(r"[^a-z\s]", "", text)

    # 7. Hapus whitespace berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================================
# FUNGSI UTAMA (MAIN FUNCTIONS)
# ==============================================

def load_and_clean(csv_path: str = None, limit: int = 30000) -> pd.DataFrame:
    """Baca CSV dataset dan jalankan pipeline pembersihan teks."""
    if csv_path is None:
        csv_path = RAW_CSV

    print(f"--- Membaca dataset dari: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {csv_path}")

    # OPTIMASI: nrows digunakan untuk membatasi jumlah baris yang dibaca
    df = pd.read_csv(csv_path, nrows=limit)

    print(f"    Jumlah baris yang dimuat: {len(df):,}")
    print(f"    Kolom: {list(df.columns)}")

    # Pastikan kolom yang dicari ada di CSV
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        fallback_text_cols = [TEXT_COL, "chat", "review_text", "text"]
        fallback_label_cols = [LABEL_COL, "label", "review_score", "sentiment"]

        found_text_col = next((c for c in fallback_text_cols if c in df.columns), None)
        found_label_col = next((c for c in fallback_label_cols if c in df.columns), None)

        if found_text_col and found_label_col:
            if found_text_col != TEXT_COL or found_label_col != LABEL_COL:
                print(
                    f"    Mendeteksi kolom teks '{found_text_col}' dan label '{found_label_col}'. "
                    f"Rename sementara ke '{TEXT_COL}' dan '{LABEL_COL}'."
                )
                df = df.rename(columns={found_text_col: TEXT_COL, found_label_col: LABEL_COL})
        else:
            raise KeyError(
                f"Kolom '{TEXT_COL}' atau '{LABEL_COL}' tidak ditemukan! "
                f"Kolom tersedia: {list(df.columns)}"
            )

    # Hapus baris kosong
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    # Bersihkan teks
    print(f"--- Sedang membersihkan {len(df)} baris teks...")
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)

    # Hapus baris yang setelah dibersihkan jadi kosong
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    print(f"SELESAI! Jumlah baris bersih: {len(df):,}")
    return df


def show_cleaning_examples(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Tampilkan contoh before vs after."""
    if df is None or df.empty:
        return pd.DataFrame()
        
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return sample[[TEXT_COL, "cleaned_text", LABEL_COL]].rename(
        columns={TEXT_COL: "original", "cleaned_text": "cleaned", LABEL_COL: "label"}
    )


# ==============================================
# Eksekusi Script
# ==============================================
if __name__ == "__main__":
    # Menentukan path manual ke data\dataset.csv
    target_path = os.path.join("data", "dataset.csv")
    
    try:
        # Menjalankan fungsi dengan limit 30,000 baris
        df_result = load_and_clean(csv_path=target_path, limit=30000)
        
        if df_result is not None:
            print("\n--- Contoh Hasil Pembersihan (30k Sample) ---")
            print(show_cleaning_examples(df_result).to_string(index=False))
            
    except Exception as e:
        print(f"\nTerjadi kesalahan: {str(e)}")