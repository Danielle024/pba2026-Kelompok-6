"""
config.py — Konfigurasi & Konstanta untuk Workshop NLP Sesi 1
=============================================================
Berisi path, mapping leetspeak, kamus slang gamer Indonesia,
dan daftar stopwords dasar.
"""

import os

# ----------------------------------------------
# 1. PATH (Lokasi Folder & File)
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Mengarah ke file dataset.csv di dalam folder data
RAW_CSV = os.path.join(DATA_DIR, "dataset.csv")

# Otomatis buat folder jika belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🔤 LEETSPEAK MAPPING
# ──────────────────────────────────────────────
LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
}

# ──────────────────────────────────────────────
# 💬 KAMUS SLANG GAMER INDONESIA
# ──────────────────────────────────────────────
SLANG_DICT = {
    # --- Kata kasar / toxic ---
    "anj": "anjing", "anjg": "anjing", "anjr": "anjing", "anjir": "anjing",
    "anjer": "anjing", "ajg": "anjing", "gblk": "goblok", "gblg": "goblok",
    "goblog": "goblok", "bgo": "bego", "bngst": "bangsat", "bgst": "bangsat",
    "kntl": "kontol", "mmk": "memek", "jnck": "jancok", "jncok": "jancok",
    "jncuk": "jancok", "tll": "tolol", "tlol": "tolol", "bdsm": "bodoh",
    "bdh": "bodoh",

    # --- Slang umum ---
    "gw": "gue", "gua": "gue", "lu": "lo", "elu": "lo", "lo": "lo",
    "loe": "lo", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "g": "tidak", "tdk": "tidak", "gk": "tidak",
    "kyk": "kayak", "kek": "kayak", "emg": "emang", "emng": "emang",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "dah": "sudah", "blm": "belum",
    "blom": "belum", "yg": "yang", "dgn": "dengan", "dg": "dengan",
    "sm": "sama", "sma": "sama", "tp": "tapi", "tpi": "tapi",
    "org": "orang", "ornag": "orang", "krn": "karena", "krna": "karena",
    "jgn": "jangan", "jng": "jangan", "bkn": "bukan", "gpp": "tidak apa-apa",
    "otw": "on the way", "btw": "by the way", "cmn": "cuman", "lg": "lagi",
    "lgi": "lagi", "aja": "saja", "aj": "saja", "bs": "bisa", "bsa": "bisa",
    "dr": "dari", "dri": "dari", "utk": "untuk", "trs": "terus",
    "trus": "terus", "msh": "masih", "masi": "masih", "jd": "jadi",
    "jdi": "jadi", "skrg": "sekarang", "skrng": "sekarang",

    # --- Gaming terms ---
    "noob": "pemula", "newbie": "pemula", "pro": "profesional",
    "gg": "good game", "wp": "well played", "afk": "away from keyboard",
    "ez": "easy", "lag": "lag", "dc": "disconnect", "bcs": "karena",
}

# ──────────────────────────────────────────────
# 🛑 KOLOM DATASET
# ──────────────────────────────────────────────
# PENTING: Sesuaikan nama kolom sesuai header di dataset.csv.
# Saat ini dataset ini memakai:
#   - teks  : 'review_text'
#   - label : 'review_score'
# Jika dataset Anda menggunakan kolom lain (misal 'chat' dan 'label'),
# ganti TEXT_COL dan LABEL_COL di bawah ini.
TEXT_COL = "review_text"
# Kolom label (1 = Recommended, -1 = Not Recommended)
LABEL_COL = "review_score"

# ----------------------------------------------
# 5. PYCARET SETTINGS
# ----------------------------------------------
SESSION_ID = 42
TRAIN_SIZE = 0.8
N_TOP_MODELS = 5