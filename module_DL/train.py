"""
train.py — Training Loop, Evaluasi, dan Visualisasi
=====================================================
Menyediakan fungsi modular untuk melatih model BiLSTM dan BERT.
Sudah disesuaikan untuk menangani dataset 3 kelas (Steam Reviews).
"""

import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# ──────────────────────────────────────────────
# ⚙️  PENYESUAIAN GLOBAL (BYPASS DLL ERROR)
# ──────────────────────────────────────────────

try:
    from config import (
        DEVICE as CONF_DEVICE, LABEL_LIST as CONF_LABELS, NUM_CLASSES as CONF_CLASSES,
        LSTM_LR, LSTM_PATIENCE, BERT_LR, BERT_PATIENCE,
        PLOT_DIR, MODEL_DIR,
    )
    DEVICE = CONF_DEVICE
    LABEL_LIST = CONF_LABELS
    NUM_CLASSES = CONF_CLASSES
except Exception:
    # Fallback jika DLL error (WinError 1114) menghalangi import torch dari config
    DEVICE = torch.device("cpu") 
    
    # --- PERBAIKAN: Ubah menjadi 2 Kelas di sini juga ---
    LABEL_LIST = ["Negative", "Positive"]
    NUM_CLASSES = 2
    
    LSTM_LR, LSTM_PATIENCE = 1e-3, 3
    BERT_LR, BERT_PATIENCE = 2e-5, 2
    PLOT_DIR, MODEL_DIR = "plots", "models"
    print("⚠️ Warning: PyTorch DLL Error terdeteksi. Menggunakan CPU Mode.")


# ──────────────────────────────────────────────
# ⚙️  HELPER: seed & criterion
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_criterion(label_counts: dict | None = None) -> nn.Module:
    if label_counts is None:
        return nn.CrossEntropyLoss()

    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(lbl, 1)) for lbl in LABEL_LIST],
        dtype=torch.float,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────
# 🔄 TRAINING & EVALUASI: LSTM
# ──────────────────────────────────────────────

def train_one_epoch_lstm(model, dataloader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        x, labels, lengths = batch
        x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        if hasattr(model, "attention"):
            logits, _ = model(x, lengths)
        else:
            logits = model(x, lengths)

        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate_lstm(model, dataloader, criterion, device=DEVICE):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch
            x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)
            if hasattr(model, "attention"):
                logits, _ = model(x, lengths)
            else:
                logits = model(x, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 🔄 TRAINING & EVALUASI: BERT
# ──────────────────────────────────────────────

def train_one_epoch_bert(model, dataloader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate_bert(model, dataloader, criterion, device=DEVICE):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 🏋️  TRAINING LOOP UTAMA
# ──────────────────────────────────────────────

def train_model(model, train_loader, val_loader, model_type, save_path, epochs, lr, patience, device=DEVICE, label_counts=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_criterion(label_counts)

    use_bert = (model_type == "bert")
    train_fn = train_one_epoch_bert if use_bert else train_one_epoch_lstm
    eval_fn  = evaluate_bert        if use_bert else evaluate_lstm

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    start_time = time.time()

    print(f"\n🚀 Start Training: {model_type.upper()}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_fn(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Early stopping di epoch {epoch}")
                break

    history["total_time"] = time.time() - start_time
    model.load_state_dict(best_model_state)
    return history


# ──────────────────────────────────────────────
# 📊 VISUALISASI
# ──────────────────────────────────────────────

def plot_training_curves(history, model_name, save=True):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train"); axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train"); axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    if save: plt.savefig(os.path.join(PLOT_DIR, f"curves_{model_name.lower()}.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save=True):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_LIST, yticklabels=LABEL_LIST)
    plt.title(f"Confusion Matrix - {model_name}")
    if save: plt.savefig(os.path.join(PLOT_DIR, f"cm_{model_name.lower()}.png"))
    plt.close()

def print_classification_report(y_true, y_pred, model_name):
    print(f"\nReport: {model_name}")
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0))
    return {"accuracy": accuracy_score(y_true, y_pred), 
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted')}

def compare_models(results):
    print("\nKomparasi Model:")
    for name, m in results.items():
        print(f"{name}: Acc {m['accuracy']:.4f}")