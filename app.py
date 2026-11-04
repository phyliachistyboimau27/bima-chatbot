import streamlit as st
import tensorflow as tf
import pickle
import json
import numpy as np
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi Halaman
st.set_page_config(page_title="BIMA UNDANA", page_icon="🤖", layout="wide")

# ==========================================
# LOAD MODEL (Di-cache agar tidak loading terus)
# ==========================================
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

# Fungsi Preprocessing & Prediksi
def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Logo_Undana.png/200px-Logo_Undana.png", width=100)
    st.title("BIMA UNDANA")
    menu = st.radio("Navigasi Menu", ["Home", "Deskripsi", "Layanan", "Chatbot", "About Us"])
    st.markdown("---")
    st.caption("© 2026 Phylia Chisty Boimau")

# ==========================================
# HALAMAN 1: HOME
# ==========================================
if menu == "Home":
    st.title("🤖 Bantuan Informasi Mahasiswa (BIMA)")
    st.markdown("""
    ### Selamat Datang di BIMA UNDANA!
    
    BIMA (Bantuan Informasi Mahasiswa) Universitas Nusa Cendana hadir sebagai **asisten virtual pintar** berbasis *Long Short-Term Memory* (LSTM) untuk membantu mahasiswa mendapatkan informasi secara cepat dan akurat.
    
    Silakan gunakan menu di sebelah kiri untuk menjelajahi aplikasi, atau langsung menuju menu **Chatbot** untuk mulai bertanya.
    """)
    st.info("💡 **Tips:** Buka menu 'Chatbot' di sebelah kiri, lalu ketik pertanyaanmu seputar KRS, UKT, Beasiswa, atau Organisasi kampus.")

# ==========================================
# HALAMAN 2: DESKRIPSI
# ==========================================
elif menu == "Deskripsi":
    st.title("ℹ️ Sekilas Tentang BIMA")
    st.markdown("""
    **BIMA (Bantuan Informasi Mahasiswa)** dikembangkan untuk mengatasi tantangan penyebaran informasi di lingkungan kampus. Informasi seringkali tersebar di berbagai kanal (website, grup WhatsApp, papan pengumuman), membuat mahasiswa kesulitan mencari data yang valid dan terpusat.
    
    BIMA dikembangkan menggunakan teknologi *Natural Language Processing* (NLP) dengan arsitektur **Long Short-Term Memory (LSTM)**. Pendekatan ini memungkinkan sistem untuk memahami variasi bahasa alami (seperti sinonim atau singkatan) yang sering digunakan mahasiswa dalam mengajukan pertanyaan.
    
    Tujuan utama sistem ini adalah menyediakan layanan informasi 24 jam yang konsisten, cepat, dan mudah diakses oleh seluruh mahasiswa Universitas Nusa Cendana.
    """)

# ==========================================
# HALAMAN 3: LAYANAN
# ==========================================
elif menu == "Layanan":
    st.title("🛠️ Layanan yang Kami Sediakan")
    st.markdown("BIMA siap menjawab pertanyaanmu seputar empat kategori utama layanan kampus:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("🎓 **Akademik**\n\nInformasi KRS, jadwal UTS/UAS, kalender akademik, KKN, Skripsi, dan layanan akademik lainnya.")
        st.warning("🏆 **Beasiswa**\n\nInfo KIP Kuliah, Djarum, KSE, Bank Indonesia, persyaratan, dan jadwal pendaftaran.")
    with col2:
        st.info("💼 **Administrasi**\n\nProsedur UKT, surat aktif kuliah, legalisir, transkrip nilai, dan layanan administrasi umum.")
        st.error("🏢 **Organisasi**\n\nInfo UKM, BEM, HMP, serta kegiatan dan perlombaan mahasiswa di lingkungan kampus.")

# ==========================================
# HALAMAN 4: CHATBOT (Inti Aplikasi)
# ==========================================
elif menu == "Chatbot":
    st.title("💬 Chatbot BIMA")
    st.markdown("Silakan ketik pertanyaan Anda seputar layanan kampus UNDANA.")

    # Panggil model
    model, tokenizer, le, config, tag_to_kat, responses = load_models()
    MAX_LEN = config['max_len']

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

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya BIMA. Ada yang bisa saya bantu hari ini?"}]

    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "meta" in message:
                st.caption(message["meta"])

    # Input chat
    if prompt := st.chat_input("Contoh: Bagaimana cara mengisi KRS?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        tag, kategori, confidence, jawaban = prediksi(prompt)
        
        if confidence < 0.4:
            meta_text = f"⚠️ Kategori: {kategori} | Confidence: {confidence*100:.1f}% (Rendah)"
            bot_reply = "Maaf, saya kurang yakin dengan pertanyaan Anda. Bisa diperjelas dengan kata lain?"
        else:
            meta_text = f"✅ Kategori: {kategori} | Confidence: {confidence*100:.1f}%"
            bot_reply = jawaban

        st.session_state.messages.append({"role": "assistant", "content": bot_reply, "meta": meta_text})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            st.caption(meta_text)

# ==========================================
# HALAMAN 5: ABOUT US
# ==========================================
elif menu == "About Us":
    st.title("👨‍🎓 Tentang Kami")
    st.markdown("""
    **BIMA (Bantuan Informasi Mahasiswa)** adalah proyek penelitian yang dikembangkan oleh:
    
    *   **Nama:** Phylia Chisty Boimau
    *   **NIM:** 2206030048
    *   **Program Studi:** Teknik Elektro
    *   **Fakultas:** Sains dan Teknik, Universitas Nusa Cendana
    
    *© 2026 BIMA - Universitas Nusa Cendana. All rights reserved.*
    """)
