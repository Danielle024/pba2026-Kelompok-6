# 🎮 pba2026-Kelompok-6  
## Tugas Besar Pemrosesan Bahasa Alami (PBA)

Repositori ini berisi implementasi Tugas Besar mata kuliah **Pemrosesan Bahasa Alami (NLP)** Institut Teknologi Sumatera.  
Proyek ini berfokus pada **perbandingan performa Machine Learning (ML) dan Deep Learning (DL)** dalam tugas klasifikasi teks.

---

## 👥 Anggota Kelompok

| Nama                     | NIM        | GitHub |
|--------------------------|------------|--------|
| Abit Ahmad Oktarian      | 122450042  | -      |
| Fadhil Fitra Wijaya      | 122450082  | -      |
| Dhafin Razaqa            | 122450133  | -      |

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan **analisis sentimen** terhadap ulasan pengguna pada platform Steam, khususnya untuk game:

- 🎮 Dota 2  
- 🎯 Counter-Strike: Global Offensive (CS:GO)

Model akan mengklasifikasikan teks review menjadi:
- ✅ **Positive (Recommended)**
- ❌ **Negative (Not Recommended)**

---

## 📊 Dataset

Dataset yang digunakan adalah:

🔗 https://www.kaggle.com/datasets/andrewmvd/steam-reviews

### 📁 Deskripsi Dataset:
Dataset ini berisi jutaan review pengguna Steam dengan atribut utama:
- `review_text` → teks ulasan pengguna
- `recommended` → label sentimen (True/False)
- metadata tambahan (playtime, helpful votes, dll)

Dataset akan difilter untuk mengambil data khusus:
- Dota 2
- Counter-Strike: Global Offensive

---
