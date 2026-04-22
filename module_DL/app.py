import gradio as gr
import torch
import torch.nn as nn
import json
import re

# 1. Definisi Arsitektur Model
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        # Pastikan lengths menggunakan format angka bulat (int64) agar stabil
        packed_output, (hidden, cell) = self.lstm(nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().to(torch.int64), batch_first=True, enforce_sorted=False))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        att_weights = torch.tanh(self.attention(output))
        att_weights = torch.softmax(att_weights, dim=1)
        context = torch.sum(att_weights * output, dim=1)
        
        logits = self.fc(self.dropout(context))
        return logits

# 2. Fungsi Load Model & Utils
def load_resources():
    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)
    with open("models/label_encoder.json", "r") as f:
        label_map = json.load(f)
    
    # Balik mapping label_encoder (0 -> Negative, dsb)
    # Gunakan int(v) agar dijamin bertipe angka
    id_to_label = {int(v): k for k, v in label_map.items()}
    
    # Inisialisasi model
    model = BiLSTM_Attention(vocab_size=20000, embed_dim=128, hidden_dim=256, 
                             num_layers=2, num_classes=2, dropout=0.3)
    
    # Load weights
    model.load_state_dict(torch.load("models/bilstm_attention.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, vocab, id_to_label

model, vocab, id_to_label = load_resources()

# 3. Fungsi Prediksi (dilengkapi Penangkap Error)
def predict(text):
    try:
        # Cleaning sederhana
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        
        if not tokens:
            return "⚠️ Teks kosong atau tidak valid."

        # Transform ke sequence
        unk_idx = vocab.get("<UNK>", 1)
        seq = [vocab.get(t, unk_idx) for t in tokens]
        
        seq = seq[:100] # Max len 100
        length = torch.tensor([len(seq)], dtype=torch.int64)
        seq_tensor = torch.zeros((1, 100), dtype=torch.long)
        seq_tensor[0, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
        with torch.no_grad():
            logits = model(seq_tensor, length)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_idx].item()
        
        label = id_to_label[pred_idx]
        emoji = "🔥" if label == "Positive" else "💀"
        return f"{emoji} {label} ({conf:.2%})"
    
    except Exception as e:
        # Jika ada error, tampilkan detailnya di layar!
        return f"❌ Error Backend: {str(e)}"

# 4. Kustomisasi CSS Gaming
custom_css = """
body { background-color: #0b0e14; }
.gradio-container { background-color: #0b0e14 !important; color: #00ff41 !important; font-family: 'Courier New', Courier, monospace; }
button.primary { background: linear-gradient(45deg, #ff00ff, #00ffff) !important; border: none !important; color: white !important; font-weight: bold !important; text-shadow: 2px 2px 4px #000; }
input, textarea { background-color: #1a1c23 !important; border: 1px solid #00ff41 !important; color: #00ff41 !important; }
label { color: #00ffff !important; text-transform: uppercase; letter-spacing: 2px; }
"""

# 5. Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🎮 STEAM REVIEW ANALYZER (BiLSTM + Attention)")
    gr.Markdown("### Masukkan review game kamu untuk dianalisis oleh AI")
    
    with gr.Row():
        input_text = gr.Textbox(placeholder="Contoh: This game is amazing, the graphics are top notch!", label="Review Teks")
    
    with gr.Row():
        # UBAH: Menggunakan Textbox agar aman dari error Pydantic/Labeling
        output_text = gr.Textbox(label="Hasil Prediksi Sentimen", interactive=False)
        
    btn = gr.Button("ANALYZE REVIEW 🚀", variant="primary")
    btn.click(fn=predict, inputs=input_text, outputs=output_text)
    
    gr.Markdown("---")
    gr.Markdown("Created for NLP Project - Checkpoint 3")

# UBAH: Parameter css dipindah ke sini sesuai peringatan Gradio 6.0
demo.launch(css=custom_css)