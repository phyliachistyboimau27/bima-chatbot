import streamlit as st
import tensorflow as tf
import pickle
import json
import numpy as np
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi Halaman
st.set_page_config(page_title="BIMA UNDANA", page_icon="🤖", layout="centered")

# Load semua artefak model (di-cache agar cepat)
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('model_bima_lstm.keras')
    with open('tokenizer_bima.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder_bima.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('config_bima.json', 'r') as f:
        config = json.load(f)
    with open('tag_to_kategori.json', 'r') as f:
        tag_to_kat = json.load(f)
    with open('responses.json', 'r') as f:
        responses = json.load(f)
    return model, tokenizer, le, config, tag_to_kat, responses

model, tokenizer, le, config, tag_to_kat, responses = load_models()
MAX_LEN = config['max_len']

# Fungsi Preprocessing (sama persis dengan training)
def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi Prediksi
def prediksi(teks):
    bersih = clean(teks)
    seq = tokenizer.texts_to_sequences([bersih])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(prob))
    tag = le.inverse_transform([idx])[0]
    kategori = tag_to_kat.get(tag, "Lainnya")
    confidence = float(prob[idx])
    
    resp_list = responses.get(tag, ["Maaf, belum ada jawaban untuk pertanyaan ini."])
    jawaban = np.random.choice(resp_list)
    
    return tag, kategori, confidence, jawaban

# --- Tampilan UI ---
st.title("🤖 Bantuan Informasi Mahasiswa (BIMA)")
st.markdown("Selamat datang! Silakan ketik pertanyaan Anda seputar layanan akademik, administrasi, beasiswa, dan organisasi di UNDANA.")

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya BIMA. Ada yang bisa saya bantu hari ini?"}]

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message:
            st.caption(message["meta"])

# Input chat dari user
if prompt := st.chat_input("Contoh: Bagaimana cara mengisi KRS?"):
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Proses prediksi
    tag, kategori, confidence, jawaban = prediksi(prompt)
    
    # Tentukan respon berdasarkan confidence
    if confidence < 0.4:
        meta_text = f"⚠️ Kategori: {kategori} | Confidence: {confidence*100:.1f}% (Rendah)"
        bot_reply = "Maaf, saya kurang yakin dengan pertanyaan Anda. Bisa diperjelas dengan kata lain?"
    else:
        meta_text = f"✅ Kategori: {kategori} | Confidence: {confidence*100:.1f}%"
        bot_reply = jawaban

    # Simpan jawaban bot
    st.session_state.messages.append({"role": "assistant", "content": bot_reply, "meta": meta_text})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        st.caption(meta_text)