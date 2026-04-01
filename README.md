# 🎮 pba2026-Kelompok-6  
## Tugas Besar Pemrosesan Bahasa Alami (PBA)

Repositori ini berisi implementasi Tugas Besar mata kuliah **Pemrosesan Bahasa Alami (NLP)** Institut Teknologi Sumatera.  
Proyek ini berfokus pada **perbandingan performa Machine Learning (ML) dan Deep Learning (DL)** dalam tugas klasifikasi teks.

---

## 👥 Anggota Kelompok

| Nama                     | NIM        | GitHub           |
|--------------------------|------------|------------------|
| Abit Ahmad Oktarian      | 122450042  | Danielle024      |
| Fadhil Fitra Wijaya      | 122450082  | epwefwe          |
| Dhafin Razaqa            | 122450133  | -                |

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

Dalam proyek ini, dataset tidak digunakan secara keseluruhan. Dilakukan proses filtering untuk mengambil data yang lebih spesifik, yaitu ulasan dari dua game populer:
- Dota 2
- Counter-Strike: Global Offensive (CS:GO)

Pemilihan kedua game ini didasarkan pada jumlah ulasan yang besar serta variasi opini pengguna yang tinggi, sehingga memberikan data yang representatif untuk membandingkan performa model Machine Learning dan Deep Learning dalam tugas klasifikasi sentimen. Sebelum digunakan dalam proses pemodelan, dataset akan melalui tahap preprocessing untuk membersihkan teks dan mempersiapkan data agar dapat diproses secara optimal oleh model.

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
