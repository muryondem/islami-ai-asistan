# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import math
import base64
import random
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# AYARLAR & API
# ==========================================
API_ANAHTARIM = st.secrets["GEMINI_API_KEY"]
VERITABANI_YOLU = "./veritabanı"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

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

def metni_seslendir(metin):
    try:
        tts = gTTS(text=metin.replace("*", ""), lang='tr', slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return ""

# ==========================================
# TASARIM (GÖRSEL SABİTLİK VE EKRAN BÖLME)
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { overflow: hidden; background-color: #000000; }
    .stApp { background-color: #000000; color: #FFFFFF; }
    .stChatMessage { border-radius: 15px; background-color: #1A1A1A !important; margin-bottom: 10px; }
    .stButton>button { border-radius: 20px; background-color: #1A1A1A; border: 1px solid #333; color: white !important; }
    audio { filter: invert(100%); width: 100%; height: 30px; }
    </style>
    """, unsafe_allow_html=True)

# 1. BÖLGE: ÜST (SABİT POPÜLER SORULAR)
with st.container():
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
    st.divider()

# 2. BÖLGE: ALT (KAYDIRILABİLİR CHAT)
chat_container = st.container(height=520)

if "messages" not in st.session_state: st.session_state.messages = []

with chat_container:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and st.button("🔊 Dinle", key=f"s_{i}"):
                st.markdown(metni_seslendir(m["content"]), unsafe_allow_html=True)

# 3. BÖLGE: GİRDİ
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.clicked_q if st.session_state.clicked_q else u_input
st.session_state.clicked_q = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# --- ASİSTAN CEVAP ÜRETİMİ (ÖĞRETİCİ TON VE SELAMLAMA) ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_prompt = st.session_state.messages[-1]["content"]
    
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Bilge MUIN kaynakları mütalaa ediyor..."):
                try:
                    if vector_db:
                        docs = vector_db.similarity_search(current_prompt, k=6)
                        baglam = "\n\n".join([f"📚 Kaynak: {os.path.basename(d.metadata['source'])}\n{d.page_content}" for d in docs])
                    else: baglam = "Veritabanı bağlantısı yok."

                    # KARAKTER VE TONLAMA TALİMATLARI
                    system_instructions = (
                        "Sen bilge, nazik ve ilim sahibi bir İslami asistan olan MUIN'sin. "
                        "Üslubun her zaman öğretici, şefkatli ve yol gösterici olmalıdır. "
                        "Cevaplarına başlarken her seferinde farklı olacak şekilde; 'Selamünaleyküm kıymetli kardeşim', 'Aziz dostum merhaba', "
                        "'Sevgili kardeşim, hoş geldin', 'Esselamü aleyküm, seni dinliyorum' gibi samimi karşılamalar kullan. "
                        "\n\nKurallar:\n"
                        "1. Mutlaka kaynaklara atıf yap (Kaynak: Dosya Adı şeklinde).\n"
                        "2. Eğer kaynaklarda bilgi kısıtlıysa, genel İslami bilgini kullanarak konuyu derinleştir ve 'Komşuluk', 'Ahlak' gibi konularda öğretici bir ders verir gibi anlat.\n"
                        "3. Yıldız (*) karakterini asla kullanma, metni düz ve akıcı yaz.\n"
                        "4. Cevapların sonunda kısa bir dua veya güzel bir temenni ile bitir."
                    )
                    
                    full_prompt = f"{system_instructions}\n\nKAYNAKLAR:\n{baglam}\n\nSORU: {current_prompt}"
                    
                    res = client.models.generate_content(model=GUNCEL_MODEL, contents=full_prompt)
                    full_res = res.text
                    
                    st.markdown(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                    st.rerun()
                except Exception as e:
                    st.error(f"Bir sorun oluştu: {e}")