# -*- coding: utf-8 -*-
import streamlit as st
import os
import zipfile
import gdown
import base64
import json
import math
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# AYARLAR
# ==========================================
API_ANAHTARIM = st.secrets["GEMINI_API_KEY"]
VERITABANI_YOLU = "./veritabani"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

# ==========================================
# TASARIM & KATMANLI CSS (KESİN ÇÖZÜM)
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp { background-color: #000000; }

    /* 1. ÜST SABİT BÖLGE (Header) */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #000000;
        z-index: 1000;
        padding: 10px 20px;
        border-bottom: 2px solid #333;
    }

    /* 2. CHAT ALANI İÇİN BOŞLUK (Üstten ve Alttan) */
    .main-content {
        margin-top: 240px; /* Popüler soruların yüksekliğine göre ayarlandı */
        margin-bottom: 100px;
    }

    /* Chat Balonları Tasarımı */
    .stChatMessage { border-radius: 15px; background-color: #1A1A1A !important; margin-bottom: 15px; }
    
    /* Popüler Soru Butonları */
    div.stButton > button {
        width: 100%;
        border-radius: 15px;
        background-color: #262626;
        color: white !important;
        border: 1px solid #444;
        font-size: 13px;
        padding: 5px;
    }

    /* Soru Giriş Kutusunun Yerleşimi (En Üstte Kalması İçin) */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        z-index: 1001;
    }
    
    /* Ses Dosyası Görünümü */
    audio { filter: invert(100%); width: 100%; height: 35px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# FONKSİYONLAR
# ==========================================
def populer_sorulari_getir():
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    v_db = Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings) if os.path.exists(VERITABANI_YOLU) else None
    return v_db, embeddings

vector_db, embeddings_model = kaynaklari_yukle()

# ==========================================
# 1. KATMAN: SABİT POPÜLER SORULAR (ÜST)
# ==========================================
with st.container():
    # HTML ile sabit başlık alanı oluşturuyoruz
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    st.title("🌙 MUIN")
    
    populerler = populer_sorulari_getir()
    if "clicked_q" not in st.session_state: st.session_state.clicked_q = None

    if populerler:
        st.markdown("##### 🌟 Popüler Sorular")
        c1, c2 = st.columns(2)
        for i, k in enumerate(populerler[:4]):
            with (c1 if i%2==0 else c2):
                if st.button(f"🔍 {k['soru']}", key=f"v_{i}"):
                    st.session_state.clicked_q = k['soru']
        
        if len(populerler) > 4:
            with st.expander("Tümünü Gör"):
                c3, c4 = st.columns(2)
                for i, k in enumerate(populerler[4:]):
                    with (c3 if i%2==0 else c4):
                        if st.button(f"🔍 {k['soru']}", key=f"m_{i}"):
                            st.session_state.clicked_q = k['soru']
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 2. KATMAN: CHAT AKIŞI (ORTA)
# ==========================================
# Boşluk bırakmak için görünmez bir alan
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if st.button("🔊 Dinle", key=f"s_{i}"):
                # Basit seslendirme fonksiyonu (Hız için kısa tutuldu)
                tts = gTTS(text=m["content"].replace("*",""), lang='tr')
                tts.save("voice.mp3")
                with open("voice.mp3", "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    st.markdown(f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 3. KATMAN: GİRDİ YÖNETİMİ (ALT)
# ==========================================
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.clicked_q if st.session_state.clicked_q else u_input
st.session_state.clicked_q = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Burada asistan cevabını üretme ve RAG süreçleri (Önceki kodla aynı)
    with st.chat_message("assistant"):
        with st.spinner("İlmi cevap hazırlanıyor..."):
            try:
                docs = vector_db.similarity_search(prompt, k=4) if vector_db else []
                baglam = "\n".join([d.page_content for d in docs])
                res = client.models.generate_content(
                    model=GUNCEL_MODEL, 
                    contents=f"Asistan MUIN. Kaynak: {baglam}\nSoru: {prompt}"
                )
                st.markdown(res.text)
                st.session_state.messages.append({"role": "assistant", "content": res.text})
                st.rerun()
            except Exception as e:
                st.error(f"Hata: {e}")