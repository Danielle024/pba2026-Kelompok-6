import pandas as pd
import gradio as gr
from pycaret.classification import load_model, predict_model
from preprocess import clean_text

# ==============================================================
# 🧠 1. LOAD MODEL MACHINE LEARNING
# ==============================================================
print("Memuat model NLP...")
# PENTING: Jika jalankan di laptop (lokal), gunakan "models/nlp_pipeline_final"
# Jika nanti di-upload ke Hugging Face (satu folder), ubah menjadi "nlp_pipeline_final"
model = load_model("models/nlp_pipeline_final")

def predict_sentiment(teks_review):
    if not teks_review.strip():
        return "Harap masukkan teks review."
    
    try:
        # Bersihkan teks input
        teks_bersih = clean_text(teks_review)
        
        # Masukkan ke dalam format DataFrame
        df_input = pd.DataFrame({'cleaned_text': [teks_bersih]})
        
        # Lakukan prediksi
        prediksi = predict_model(model, data=df_input)
        
        # Ekstrak hasil label
        if 'prediction_label' in prediksi.columns:
            hasil = prediksi['prediction_label'].iloc[0]
        else:
            hasil = prediksi['Label'].iloc[0] 
            
        # Format output
        if hasil == 1:
            return "👍 POSITIF (Recommended)"
        else:
            return "👎 NEGATIF (Not Recommended)"
    except Exception as e:
        return f"Terjadi kesalahan saat memproses: {str(e)}"


# ==============================================================
# 🎮 2. KONFIGURASI TEMA GAMER (STEAM STYLE)
# ==============================================================

bg_color = "#171a21"        # Latar belakang gelap khas Steam
panel_color = "#2a475e"     # Warna panel biru gelap
text_color = "#c7d5e0"      # Warna teks abu-abu terang
primary_accent = "#66c0f4"  # Biru cerah khas tombol Steam
border_color = "#1b2838"    # Garis tepi gelap

custom_css = f"""
    body, .gradio-container {{ background-color: {bg_color}; color: {text_color}; font-family: 'Trebuchet MS', Helvetica, sans-serif; }}
    .gr-box, .gr-panel {{ background-color: {panel_color}; border-color: {border_color}; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
    .gr-input, .gr-text-input, textarea {{ background-color: #101214 !important; color: #ffffff !important; border-color: {border_color} !important; }}
    .gr-input:focus, textarea:focus {{ border-color: {primary_accent} !important; box-shadow: 0 0 5px {primary_accent} !important; }}
    .gr-button-primary {{ background-color: {primary_accent} !important; color: #000000 !important; font-weight: bold; border-radius: 3px !important; transition: 0.2s; }}
    .gr-button-primary:hover {{ background-color: #ffffff !important; box-shadow: 0 0 10px {primary_accent}; }}
    h1, h2, h3 {{ color: #ffffff; text-shadow: 1px 1px 2px black; }}
"""

# primary_hue dipindah ke dalam kurung Default()
gamer_theme = gr.themes.Default(primary_hue="blue").set(
    body_background_fill=bg_color,
    body_text_color=text_color,
    input_background_fill="#101214",
    button_primary_background_fill=primary_accent,
    button_primary_text_color="#000000",
)

# ==============================================================
# 💻 3. APLIKASI GRADIO DEMO
# ==============================================================

with gr.Blocks(theme=gamer_theme, css=custom_css) as demo:
    gr.Markdown("# 🎮 Steam Review Sentiment Analyzer")
    gr.Markdown("Masukkan review game di bawah ini untuk melihat apakah model tersebut mendeteksinya sebagai **Recommended (Positif)** atau **Not Recommended (Negatif)**.")

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=5, 
                placeholder="Tulis review game di sini (contoh: lag parah, banyak cheater, nyesel beli)...", 
                label="Review Game"
            )
            predict_btn = gr.Button("Analisis Sentimen", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Hasil Analisis AI", interactive=False)
            clear_btn = gr.Button("Bersihkan", variant="secondary")

    with gr.Accordion("💡 Tips Uji Coba ", open=False):
        gr.Markdown("- **Contoh Positif:** *I never played a better first person shooter. Graphics doesn't matter for gamer, gameplay only.*")
        gr.Markdown("- **Contoh Negatif:** *Complete waste of money, the game crashes to desktop every 5 minutes.*")

    gr.Markdown("---")
    gr.Markdown("*(Model dilatih menggunakan dataset Steam Reviews asli menggunakan PyCaret)*")

    # Menghubungkan tombol dengan fungsi prediksi AI
    predict_btn.click(fn=predict_sentiment, inputs=input_text, outputs=output_text)
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[input_text, output_text])


# Jalankan Aplikasi
if __name__ == "__main__":
    demo.launch(share=False)