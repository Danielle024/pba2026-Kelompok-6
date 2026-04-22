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

### 🧠 Deep Learning (PyTorch — BiLSTM + Attention)

Pendekatan DL dikembangkan menggunakan **PyTorch** dengan arsitektur **BiLSTM + Attention Mechanism**. Arsitektur ini dipilih karena kemampuannya membaca konteks kalimat dari dua arah (*bidirectional*) sekaligus memberikan bobot perhatian pada bagian teks yang paling relevan terhadap sentimen.

**Konfigurasi Model:**
- **Arsitektur:** BiLSTM + Attention
- **Total Parameter:** 4.929.027
- **Embedding Dimension:** 128
- **Hidden Dimension:** 256
- **Split Data:** Train: 38.744 | Val: 4.843 | Test: 4.843
- **Early Stopping:** Aktif

**Proses Training per Epoch:**

| Epoch | Loss | Accuracy | Val Loss | Val Time |
|-------|------|----------|----------|----------|
| 01 | 0.4763 | 0.7517 | 0.4004 | 406.3s |
| 02 | 0.3705 | 0.8203 | 0.3584 | 438.0s |
| 03 | 0.3140 | 0.8543 | 0.3501 | 608.1s |
| 04 | 0.2706 | 0.8763 | 0.3438 | 1398.2s |
| 05 | 0.2351 | 0.8945 | 0.3420 | 646.5s |
| 06 | 0.1996 | 0.9105 | 0.4168 | 344.0s |
| 07 | 0.1726 | 0.9245 | 0.4630 | 296.3s |
| 08 | 0.1473 | 0.9373 | 0.5093 | 295.3s |

> ⚠️ **Early stopping** diaktifkan pada **Epoch 8** karena *validation loss* mulai meningkat sejak epoch 6, mengindikasikan gejala *overfitting*.

**Hasil Evaluasi pada Data Uji (Test Set):**

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.49 | 0.90 | 0.63 | 776 |
| Positive | 0.98 | 0.82 | 0.89 | 4067 |
| **Accuracy** | | | **0.83** | **4843** |
| Macro Avg | 0.73 | 0.86 | 0.76 | 4843 |
| Weighted Avg | 0.90 | 0.83 | 0.85 | 4843 |

**Analisis Hasil:**
- Model mencapai **akurasi keseluruhan sebesar 83%** pada data uji.
- Untuk kelas **Positive**, model menunjukkan performa sangat baik dengan precision 0.98 dan F1-score 0.89, mencerminkan kemampuan model dalam mengidentifikasi ulasan positif dengan tepat.
- Untuk kelas **Negative**, recall sebesar 0.90 menandakan model cukup mampu mendeteksi ulasan negatif, namun precision yang rendah (0.49) mengindikasikan banyak *false positive* — yaitu ulasan positif yang salah diklasifikasikan sebagai negatif. Hal ini kemungkinan disebabkan oleh **ketidakseimbangan kelas** (*class imbalance*) pada dataset (776 negatif vs 4067 positif).
- **Weighted average F1-score sebesar 0.85** menunjukkan performa keseluruhan yang baik dengan mempertimbangkan distribusi kelas yang tidak seimbang.

> ✅ **Kesimpulan:** Model BiLSTM+Attention berhasil dilatih dengan baik dan siap untuk tahap *deployment*. Model diekspor dalam format `.pt` (PyTorch checkpoint).

---
 
## 🔗 Link Demo & Deployment
### 🔬 Machine Learning (LightGBM)

Model **LightGBM** terbaik telah berhasil di-*deploy* menjadi aplikasi web interaktif menggunakan antarmuka **Gradio** dengan kustomisasi tema visual ala *Steam Dark Mode*. Model ini memanfaatkan representasi teks berbasis **TF-IDF** dan dipilih karena memiliki performa terbaik berdasarkan evaluasi (F1 Score tertinggi) serta efisiensi waktu pelatihan. Aplikasi ini berjalan secara *live* pada *environment* Python 3.10.

### 🤖 Deep Learning (BiLSTM + Attention)

Model **Deep Learning (BiLSTM + Attention)** juga berhasil di-*deploy* sebagai aplikasi web interaktif menggunakan **Gradio**. Arsitektur ini dirancang untuk menangkap konteks urutan kata dalam teks serta memberikan bobot perhatian (*attention*) pada bagian kalimat yang paling relevan terhadap sentimen. Pendekatan ini memungkinkan model memahami nuansa bahasa yang lebih kompleks seperti konteks, ironi, dan pola kalimat. Aplikasi ini juga berjalan secara *live* pada *environment* Python 3.10.
 
| Model | Platform | Link |
|-------|----------|------|
| **Sentimen Model (ML - LightGBM)** | Hugging Face Spaces | [🎮 Coba Demo ML (GameSteam-Review-Sentiment)](https://huggingface.co/spaces/Bitlancka/GameSteam-Review-Sentiment) |
| **Sentimen Model (DL - BiLSTM + Attention)** | Hugging Face Spaces | [🤖 Coba Demo DL (Sentimen-SteamReview-NLPxDeeplearning)](https://huggingface.co/spaces/Bitlancka/Sentimen-SteamReview-NLPxDeeplearning) |

---
*Beberapa bantuan asisten dari AI Gemini dan Copilot:* 🔗 [Folder Dokumentasi AI (Google Drive)](https://drive.google.com/drive/folders/1kwjQTFWiDKAUz9QmkHJgf3FlqjfmkJXK?usp=sharing)
