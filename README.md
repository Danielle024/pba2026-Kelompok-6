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
 
Proyek ini bertujuan untuk melakukan **analisis sentimen** terhadap ulasan pengguna pada platform Steam. Berdasarkan hasil eksplorasi data awal (EDA) dan batasan memori komputasi, model dilatih menggunakan **30.000 baris pertama** dataset yang mencakup ulasan dari game-game berikut:
 
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
 
Model dikembangkan untuk mengklasifikasikan teks review ke dalam dua kelas utama:
- ✅ **Positive (Recommended)** -> Dilambangkan dengan label `1`
- ❌ **Negative (Not Recommended)** -> Dilambangkan dengan label `-1`

---

## 📊 Dataset & Preprocessing

### 📁 Deskripsi Dataset
Dataset yang digunakan berasal dari platform Kaggle dan dapat diakses melalui tautan berikut:  
🔗 [Steam Reviews Dataset (AndrewMVD)](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)

Setiap entri dalam dataset merepresentasikan satu ulasan pengguna yang terdiri dari teks review serta label sentimen yang menunjukkan apakah pengguna merekomendasikan game tersebut atau tidak. Label ini digunakan sebagai acuan (*ground truth*) dalam proses pelatihan model klasifikasi.

### 🔍 Karakteristik Teks Ulasan Steam
Ulasan pada platform Steam memiliki karakteristik unik yang menjadi tantangan tersendiri dalam NLP:

| Karakteristik | Keterangan |
|---|---|
| 🗣️ **Bahasa informal & gaul** | Banyak singkatan, slang gaming, dan bahasa tidak baku. |
| 🌐 **Mixed language** | Beberapa ulasan mencampur berbagai bahasa. |
| 😏 **Sarkasme & ironi** | Ulasan negatif terkadang ditulis dengan nada positif. |
| 📏 **Variatif dalam panjang** | Dari satu kata hingga beberapa paragraf. |
| 🎮 **Domain-specific vocabulary**| Istilah khusus gaming: *nerf*, *buff*, *meta*, *AFK*, *feed*, dll. |

### 🛠️ Pipeline Preprocessing Teks
Untuk mengatasi tantangan teks Steam yang kotor dan tidak terstruktur, kami membangun modul `preprocess.py` dengan *pipeline* pembersihan khusus sebagai berikut:
1. **Lowercasing:** Mengubah seluruh teks menjadi huruf kecil.
2. **URL & Mention Removal:** Menghapus tautan `http/https` dan *mention* `@`.
3. **Leetspeak Normalization:** Mengembalikan angka yang dijadikan huruf (contoh: `g0bl0k` menjadi `goblok`).
4. **Slang Expansion:** Mengubah kata gaul/singkatan gaming menjadi kata baku menggunakan kamus khusus (`SLANG_DICT`).
5. **Non-Alphabet Removal:** Membuang tanda baca, emoji, dan simbol (menyisakan huruf `a-z` dan spasi).
6. **Whitespace Stripping:** Merapikan spasi yang berlebih.

---

## 🤖 Model yang Digunakan
 
### 🔬 Machine Learning (PyCaret)
Pendekatan ML dikembangkan menggunakan **PyCaret**, sebuah framework AutoML untuk membandingkan performa berbagai algoritma secara otomatis. 

**Konfigurasi Eksperimen (Setup):**
- **Text Embedding:** TF-IDF (*Term Frequency-Inverse Document Frequency*)
- **Sampling:** 30.000 sampel data. Disini kami hanya mengambil sebagian kecil data dikarenakan untuk total baris pada dataset kami mencapai 6.417.106
- **Train/Test Split:** 80% Data Latih, 20% Data Uji.
- **Cross-Validation:** Stratified K-Fold (Fold = 3)
 
Berikut adalah hasil perbandingan model menggunakan fungsi `compare_models()`, diurutkan berdasarkan **F1 Score** tertinggi:
 
| Rank | Model | F1 | Kappa | MCC | TT (Sec) |
|------|-------|----|-------|-----|----------|
| 🥇 | **LightGBM** (`lightgbm`) | 0.9448 | 0.3172 | 0.3697 | 11.47 |
| 🥈 | **Ridge Classifier** (`ridge`) | 0.9401 | 0.2360 | 0.3165 | 148.90 |
| 🥉 | **Ada Boost** (`ada`) | 0.9392 | 0.2623 | 0.2910 | 108.45 |
| 4 | Gradient Boosting (`gbc`) | 0.9365 | 0.1830 | 0.2639 | 169.32 |
| 5 | Logistic Regression (`lr`) | 0.9362 | 0.1712 | 0.2699 | 29.53 |

> ✅ **Keputusan:** Model **LightGBM** dipilih untuk tahap *deployment* karena meraih F1 Score tertinggi (0.9448) dan memiliki waktu pelatihan (TT) paling efisien dibandingkan algoritma terbaik lainnya. Model diekspor dalam bentuk `.pkl`.
 
---
 
## 🔗 Link Demo & Deployment
Model LightGBM terbaik telah berhasil di-*deploy* menjadi aplikasi web interaktif menggunakan antarmuka **Gradio** dengan kustomisasi tema visual ala *Steam Dark Mode*. Aplikasi ini berjalan secara *live* pada *environment* Python 3.10.
 
| Model | Platform | Link |
|-------|----------|------|
| **Sentimen Model (ML)** | Hugging Face Spaces | [🎮 Coba Demo AI di Sini! (Bitlancka/GameSteam-Review-Sentiment)](https://huggingface.co/spaces/Bitlancka/GameSteam-Review-Sentiment)|

---
*Beberapa bantuan asisten dari AI Gemini dan Copilot:* 🔗 [Folder Dokumentasi AI (Google Drive)](https://drive.google.com/drive/folders/1kwjQTFWiDKAUz9QmkHJgf3FlqjfmkJXK?usp=sharing)
