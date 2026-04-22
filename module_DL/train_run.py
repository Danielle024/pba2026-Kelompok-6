"""
train_run.py — Pipeline Pelatihan Lengkap (Adjusted for Steam Reviews)
=========================================================
"""

import sys
import os
import time
import torch
# Gunakan try-except untuk transformers jika belum terinstall atau DLL error
try:
    from transformers import DistilBertTokenizerFast
except ImportError:
    print("⚠️ Transformers tidak ditemukan, skip bagian BERT.")

from config import (
    DEVICE, SAMPLE_SIZE, LABEL_LIST,
    VOCAB_SIZE, MAX_LEN, LSTM_BATCH_SIZE, LSTM_LR, LSTM_EPOCHS, LSTM_PATIENCE,
    BERT_MAX_LEN, BERT_BATCH_SIZE, BERT_LR, BERT_EPOCHS, BERT_PATIENCE,
    BILSTM_MODEL_PATH, BILSTM_ATT_MODEL_PATH, DISTILBERT_MODEL_DIR,
    VOCAB_PATH, PLOT_DIR,
)

# Jika preprocess.py kamu masih menggunakan hardcode kolom, pastikan di sini sinkron
from preprocess import load_and_clean, show_cleaning_examples
from dataset import Vocabulary, get_lstm_dataloaders, get_bert_dataloaders
from models import BiLSTMClassifier, BiLSTMAttentionClassifier, DistilBERTClassifier, count_parameters
from train import (
    set_seed, train_model,
    evaluate_lstm, evaluate_bert,
    get_criterion,
    plot_training_curves, plot_confusion_matrix,
    print_classification_report, compare_models,
)

import matplotlib.pyplot as plt
import seaborn as sns

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")

def main():
    set_seed(42)
    # FORCE CPU jika masih WinError 1114
    current_device = torch.device("cpu") if not torch.cuda.is_available() else DEVICE
    
    print(f"🖥️  Device: {current_device}")
    print(f"📊 Sample size: {'All (50k)' if SAMPLE_SIZE is None else SAMPLE_SIZE}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 1. DATA PATH ──────────────────────────────
    section("1. Persiapan Data")
    # Lewati download karena kamu sudah ada file data/dataset_50k.csv
    csv_path = os.path.join("data", "dataset_50k.csv")

    # ── 2. PREPROCESS ────────────────────────────
    section("2. Preprocessing")
    # Gunakan SAMPLE_SIZE=None sesuai permintaanmu untuk 50k data
    df = load_and_clean(csv_path, sample_size=SAMPLE_SIZE)
    show_cleaning_examples(df, n=3)

    label_counts = df["label"].value_counts().to_dict()

    # Plot distribusi kelas
    plt.figure(figsize=(9, 4))
    df["label"].value_counts().sort_values().plot(kind="barh", color=sns.color_palette("Set2"))
    plt.title("Distribusi Kelas Sentiment")
    plt.xlabel("Jumlah Data")
    plt.tight_layout()
    dist_path = os.path.join(PLOT_DIR, "distribusi_kelas.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. BUILD VOCABULARY ──────────────────────
    section("3. Membangun Vocabulary")
    vocab = Vocabulary()
    vocab.build_vocab(df["cleaned_text"].tolist(), max_size=VOCAB_SIZE)
    vocab.save(VOCAB_PATH)

    # ── 4. DATALOADER (LSTM) ─────────────────────
    section("4. Membuat DataLoaders (LSTM)")
    train_loader, val_loader, test_loader = get_lstm_dataloaders(
        df, vocab, max_len=MAX_LEN, batch_size=LSTM_BATCH_SIZE
    )

    # ── 5. TRAIN BILSTM + ATTENTION ───────────────
    # Saya sarankan fokus ke Attention karena ini yang paling performant untuk NLP
    section("5. Training BiLSTM + Attention")
    bilstm_att = BiLSTMAttentionClassifier(vocab_size=len(vocab))
    
    total_params = count_parameters(bilstm_att)
    print(f"   Parameter: {total_params:,}")
    
    if total_params > 10_000_000:
        print("⚠️ PERINGATAN: Parameter melebihi 10 juta! Kurangi VOCAB_SIZE di config.")

    hist_bilstm_att = train_model(
        model=bilstm_att,
        train_loader=train_loader, val_loader=val_loader,
        model_type="lstm_att", save_path=BILSTM_ATT_MODEL_PATH,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, patience=LSTM_PATIENCE,
        device=current_device, label_counts=label_counts,
    )
    plot_training_curves(hist_bilstm_att, "BiLSTM+Attention")

    # Evaluasi
    criterion = get_criterion(label_counts)
    _, _, preds_att, labels_att = evaluate_lstm(
        bilstm_att, test_loader, criterion, current_device
    )
    plot_confusion_matrix(labels_att, preds_att, "BiLSTM+Attention")
    metrics_att = print_classification_report(labels_att, preds_att, "BiLSTM+Attention")
    metrics_att["training_time_min"] = hist_bilstm_att["total_time"] / 60

    print("\n✅ Pipeline Selesai! Model BiLSTM+Attention siap dideploy.")

if __name__ == "__main__":
    main()