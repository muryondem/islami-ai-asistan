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
# AYARLAR & API
# ==========================================
API_ANAHTARIM = st.secrets["GEMINI_API_KEY"]
VERITABANI_YOLU = "./veritabani"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

# ==========================================
# FONKSİYONLAR (Aynı Mantık, Zeki Hafıza Dahil)
# ==========================================
def cosine_similarity_manuel(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x, y = v1[i], v2[i]
        sumxx += x*x; sumyy += y*y; sumxy += x*y
    return sumxy / math.sqrt(sumxx*sumyy)

def populer_soru_guncelle(yeni_soru, embeddings_model):
    if not yeni_soru or len(yeni_soru) < 10: return
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f:
            soru_listesi = json.load(f)
    else: soru_listesi = []

    try:
        yeni_vektor = embeddings_model.embed_query(yeni_soru)
        bulundu = False
        for soru_obj in soru_listesi:
            if "vektor" in soru_obj:
                benzerlik = cosine_similarity_manuel(yeni_vektor, soru_obj["vektor"])
                if benzerlik > 0.88:
                    soru_obj["puan"] += 1
                    bulundu = True; break
        if not bulundu:
            soru_listesi.append({"soru": yeni_soru, "puan": 1, "vektor": yeni_vektor})
        soru_listesi = sorted(soru_listesi, key=lambda x: x["puan"], reverse=True)[:20]
        with open(POPULER_SORULAR_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(soru_listesi, f, ensure_ascii=False)
    except: pass

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

def metni_seslendir(metin):
    try:
        tts = gTTS(text=metin.replace("*", ""), lang='tr', slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return ""

# ==========================================
# TASARIM & CSS (Sabitleme Efekti)
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Üst Bölgeyi Sabitleme */
    [data-testid="stVerticalBlock"] > div:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #000000;
        padding-bottom: 10px;
        border-bottom: 1px solid #333;
    }
    
    .stChatMessage { border-radius: 15px; background-color: #1A1A1A; margin-bottom: 10px; }
    .stButton>button { border-radius: 20px; background-color: #1A1A1A; border: 1px solid #333; color: white !important; }
    .streamlit-expanderHeader { background-color: #000 !important; border: 1px solid #333 !important; }
    audio { filter: invert(100%); width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. SABİT ÜST BÖLGE (Popüler Sorular)
# ==========================================
header_container = st.container()
with header_container:
    st.title("🌙 MUIN")
    populer_listesi = populer_sorulari_getir()
    
    if "clicked_q" not in st.session_state: st.session_state.clicked_q = None

    if populer_listesi:
        st.markdown("##### 🌟 Popüler Sorular")
        c1, c2 = st.columns(2)
        for i, k in enumerate(populer_listesi[:4]):
            with (c1 if i%2==0 else c2):
                if st.button(f"🔍 {k['soru']}", key=f"v_{i}", use_container_width=True):
                    st.session_state.clicked_q = k['soru']
        
        if len(populer_listesi) > 4:
            with st.expander("Tüm Popüler Soruları Gör..."):
                c3, c4 = st.columns(2)
                for i, k in enumerate(populer_listesi[4:]):
                    with (c3 if i%2==0 else c4):
                        if st.button(f"🔍 {k['soru']}", key=f"m_{i}", use_container_width=True):
                            st.session_state.clicked_q = k['soru']

# ==========================================
# 2. KAYDIRILABİLİR ALT BÖLGE (Chat)
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []

# Mesajları göster
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if st.button("🔊 Dinle", key=f"s_{i}"):
                st.markdown(metni_seslendir(m["content"]), unsafe_allow_html=True)

# Girdi ve İşlem
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.clicked_q if st.session_state.clicked_q else u_input
st.session_state.clicked_q = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # Girdiyi hemen göstermek için

# En son mesaj asistan cevabı bekliyorsa tetikle
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_prompt = st.session_state.messages[-1]["content"]
    populer_soru_guncelle(current_prompt, embeddings_model)
    
    with st.chat_message("assistant"):
        with st.spinner("İlmi kaynaklar taranıyor..."):
            try:
                if vector_db:
                    docs = vector_db.similarity_search(current_prompt, k=4)
                    baglam = "\n\n".join([f"[{os.path.basename(d.metadata['source'])}]: {d.page_content}" for d in docs])
                else: baglam = "Veritabanı bağlantısı yok."

                res = client.models.generate_content(
                    model=GUNCEL_MODEL, 
                    contents=f"Sen bilge MUIN'sin. Kaynaklar: {baglam}\nSoru: {current_prompt}\nCevabı kaynaklara dayalı ver. Yıldız (*) kullanma."
                )
                
                full_res = res.text
                st.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
            except Exception as e:
                st.error(f"Hata: {e}")