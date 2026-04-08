# 🎮 pba2026-Kelompok-6  
## Tugas Besar Pemrosesan Bahasa Alami (PBA)

Repositori ini berisi implementasi Tugas Besar mata kuliah **Pemrosesan Bahasa Alami (NLP)** Institut Teknologi Sumatera.  
Proyek ini berfokus pada **perbandingan performa Machine Learning (ML) dan Deep Learning (DL)** dalam tugas klasifikasi teks.

---

## 👥 Anggota Kelompok

| Nama                     | NIM        | GitHub            |
|--------------------------|------------|-------------------|
| Abit Ahmad Oktarian      | 122450042  | Danielle024       |
| Fadhil Fitra Wijaya      | 122450082  | epwefwe           |
| Dhafin Razaqa            | 122450133  | DhafinRazaqaLuthfi|

---

## 📌 Deskripsi Proyek

## 📌 Deskripsi Proyek
 
Proyek ini bertujuan untuk melakukan **analisis sentimen** terhadap ulasan pengguna pada platform Steam. Berdasarkan hasil eksplorasi data awal (EDA), 30.000 baris pertama dataset mencakup ulasan dari game-game berikut:
 
- 🎯 Counter-Strike
- 🪆 Rag Doll Kung Fu
- 🏗️ Silo 2
- 🪖 Call of Duty: World at War
- 👑 King's Quest Collection
- 🚀 Space Quest Collection
- 🌌 Aces of the Galaxy
- ⏱️ TimeShift
- ⛳ 3D Ultra Minigolf Adventures Deluxe
- 🤖 Prototype
- 🪖 Call of Duty: Modern Warfare 2
 
Model akan mengklasifikasikan teks review menjadi:
 
- ✅ **Positive (Recommended)**
- ❌ **Negative (Not Recommended)**

---

## 📊 Dataset

## 📁 Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari platform Kaggle dan dapat diakses melalui tautan berikut:

🔗 https://www.kaggle.com/datasets/andrewmvd/steam-reviews

Dataset ini berisi kumpulan ulasan (review) pengguna dari platform Steam dengan jumlah data yang sangat besar (jutaan entri). Dataset ini sangat cocok digunakan untuk tugas analisis sentimen karena mengandung opini pengguna dalam bentuk teks yang beragam. Setiap entri dalam dataset merepresentasikan satu ulasan pengguna yang terdiri dari teks review serta label sentimen yang menunjukkan apakah pengguna merekomendasikan game tersebut atau tidak. Label ini digunakan sebagai acuan (ground truth) dalam proses pelatihan model klasifikasi.

### 🔑 Atribut Utama:
- `review_text`: berisi teks ulasan yang ditulis oleh pengguna
- `recommended`: label sentimen dalam bentuk boolean (True = positif, False = negatif)
- metadata tambahan, seperti:
  - `playtime`: lama waktu bermain pengguna
  - `helpful_votes`: jumlah pengguna lain yang menilai review tersebut membantu
 
Dalam proyek ini, dilakukan eksplorasi awal (EDA) terhadap 30.000 baris pertama dataset untuk memahami distribusi game dan karakteristik data. Hasil EDA menunjukkan bahwa data mencakup berbagai judul game dari berbagai genre, mulai dari FPS, adventure, hingga casual. Data kemudian akan difilter sesuai kebutuhan analisis lebih lanjut.
 
Sebelum digunakan dalam proses pemodelan, dataset akan melalui tahap preprocessing untuk membersihkan teks dan mempersiapkan data agar dapat diproses secara optimal oleh model.

### 🔍 Karakteristik Teks Ulasan Steam

Ulasan pada platform Steam memiliki karakteristik unik yang menjadi tantangan tersendiri dalam NLP:

| Karakteristik | Keterangan |
|---|---|
| 🗣️ Bahasa informal & gaul | Banyak singkatan, slang gaming, dan bahasa tidak baku |
| 🌐 Mixed language | Beberapa ulasan mencampur berbagai bahasa |
| 😏 Sarkasme & ironi | Ulasan negatif terkadang ditulis dengan nada positif |
| 📏 Variatif dalam panjang | Dari satu kata hingga beberapa paragraf |
| 🎮 Domain-specific vocabulary | Istilah khusus gaming: *nerf*, *buff*, *meta*, *AFK*, *feed*, dll. |

---
 
## 🔗 Link Demo & Deployment
 
| Model | Platform | Link |
|-------|----------|------|
| ML Model (PyCaret) | Hugging Face Spaces | *(akan diperbarui)* |
| DL Model | Hugging Face Spaces | *(akan diperbarui)* |

---
