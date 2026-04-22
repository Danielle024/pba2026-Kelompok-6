"""
models.py — Definisi Arsitektur Deep Learning
==============================================
Tiga model untuk klasifikasi Steam Reviews (3 kelas):
1. BiLSTMClassifier
2. BiLSTMAttentionClassifier
3. DistilBERTClassifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import dengan penanganan error jika config bermasalah karena DLL
try:
    from config import (
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
        NUM_CLASSES, BERT_MODEL,
    )
except ImportError:
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 20000, 128, 256, 2, 0.3
    NUM_CLASSES = 3
    BERT_MODEL = "distilbert-base-uncased"

# ──────────────────────────────────────────────
# MODEL 1: BiLSTM
# ──────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT,
        pad_idx: int      = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits

# ──────────────────────────────────────────────
# MODEL 2: BiLSTM + Attention
# ──────────────────────────────────────────────

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT,
        pad_idx: int      = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        scores  = torch.tanh(self.attention(output)).squeeze(-1)
        max_len = output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attention_weights = F.softmax(scores, dim=1)

        context = (attention_weights.unsqueeze(-1) * output).sum(dim=1)
        out    = self.dropout(context)
        logits = self.fc(out)
        return logits, attention_weights

# ──────────────────────────────────────────────
# MODEL 3: DistilBERT (Penyesuaian untuk import)
# ──────────────────────────────────────────────

class DistilBERTClassifier(nn.Module):
    def __init__(
        self,
        bert_model: str  = BERT_MODEL,
        num_classes: int = NUM_CLASSES,
        dropout: float   = DROPOUT,
    ):
        super().__init__()
        try:
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(bert_model)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
            self.is_ready = True
        except ImportError:
            print("⚠️ Transformers tidak terinstal. Model DistilBERT tidak bisa digunakan.")
            self.is_ready = False

    def forward(self, input_ids, attention_mask):
        if not self.is_ready: raise RuntimeError("DistilBERT requires 'transformers' library.")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls_output))

# ──────────────────────────────────────────────
# HELPER: Hitung jumlah parameter
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = BiLSTMAttentionClassifier()
    print(f"Total Parameters: {count_parameters(model):,}")