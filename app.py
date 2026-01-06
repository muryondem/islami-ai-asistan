# -*- coding: utf-8 -*-
import streamlit as st
import os
import zipfile
import gdown
import base64
import json
import math
import time
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# AYARLAR & DOSYA YOLLARI
# ==========================================
FILE_ID = '159Ttbvafpd5f51-BIeVD3N_WaFcgi14w'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
OUTPUT = 'veritabani.zip'
VERITABANI_YOLU = "./veritabani"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
API_ANAHTARIM = st.secrets["GEMINI_API_KEY"]
GUNCEL_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=API_ANAHTARIM)

# ==========================================
# FONKSİYONLAR & BENZERLİK HESABI
# ==========================================
def veritabanini_hazirla():
    if not os.path.exists(VERITABANI_YOLU):
        with st.spinner("Muin hazırlanıyor..."):
            try:
                gdown.download(URL, OUTPUT, quiet=False)
                with zipfile.ZipFile(OUTPUT, 'r') as zip_ref:
                    zip_ref.extractall(".")
                if os.path.exists(OUTPUT):
                    os.remove(OUTPUT)
                st.success("Kütüphane başarıyla yüklendi!")
            except Exception as e:
                st.error(f"Veritabanı hatası: {e}")

veritabanini_hazirla()

def cosine_similarity_manuel(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy / math.sqrt(sumxx*sumyy)

def populer_soru_guncelle(yeni_soru, embeddings_model):
    if not yeni_soru or len(yeni_soru) < 10: return
    
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f:
            soru_listesi = json.load(f)
    else:
        soru_listesi = []

    try:
        # Kotayı korumak için kısa bir mola
        time.sleep(1) 
        yeni_vektor = embeddings_model.embed_query(yeni_soru)
        bulundu = False

        for soru_obj in soru_listesi:
            # Her karşılaştırma öncesi Google kotası için mola
            time.sleep(0.5)
            mevcut_vektor = embeddings_model.embed_query(soru_obj["soru"])
            benzerlik = cosine_similarity_manuel(yeni_vektor, mevcut_vektor)
            
            if benzerlik > 0.88:
                soru_obj["puan"] += 1
                bulundu = True
                break
        
        if not bulundu:
            soru_listesi.append({"soru": yeni_soru, "puan": 1})
        
        soru_listesi = sorted(soru_listesi, key=lambda x: x["puan"], reverse=True)[:20]
        with open(POPULER_SORULAR_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(soru_listesi, f, ensure_ascii=False, indent=4)
    except Exception:
        # Kota hatası (429) gelirse sistemi kitleme, sadece güncelleme yapma
        pass

def populer_sorulari_getir():
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    if not os.path.exists(VERITABANI_YOLU): return None, embeddings
    return Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings), embeddings

vector_db, embeddings_model = kaynaklari_yukle()

def metni_seslendir(metin):
    try:
        temiz_metin = metin.replace("*", "")
        tts = gTTS(text=temiz_metin, lang='tr', slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return "Seslendirme şu an yapılamıyor."

# ==========================================
# TASARIM & CSS
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; background-color: #1A1A1A; }
    p, span, label, .stMarkdown, h1, h2, h3, h4 { color: #FFFFFF !important; }
    .stButton>button { border-radius: 20px; background-color: #1A1A1A; border: 1px solid #333; color: white !important; font-size: 14px; margin-bottom: 5px; }
    .stButton>button:hover { border-color: #🌙; background-color: #333; }
    .streamlit-expanderHeader { background-color: #000000 !important; color: #FFFFFF !important; border: 1px solid #333 !important; border-radius: 10px !important; }
    audio { filter: invert(100%); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ANA EKRAN
# ==========================================
st.title("🌙 MUIN: İslami Bilgi Asistanı")

populer_listesi = populer_sorulari_getir()

if "clicked_question" not in st.session_state:
    st.session_state.clicked_question = None

if populer_listesi:
    st.markdown("#### 🌟 Popüler Sorular")
    cols_vitrin = st.columns(2)
    for i, kalem in enumerate(populer_listesi[:4]):
        with cols_vitrin[i % 2]:
            if st.button(f"🔍 {kalem['soru']}", key=f"vitrin_{i}", use_container_width=True):
                st.session_state.clicked_question = kalem['soru']
    
    if len(populer_listesi) > 4:
        with st.expander("Daha Fazla Popüler Soru Gör..."):
            cols_more = st.columns(2)
            for i, kalem in enumerate(populer_listesi[4:]):
                with cols_more[i % 2]:
                    if st.button(f"🔍 {kalem['soru']}", key=f"more_{i}", use_container_width=True):
                        st.session_state.clicked_question = kalem['soru']
else:
    st.info("Henüz popüler soru oluşmadı.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj geçmişi
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if st.button(f"🔊 Dinle", key=f"btn_{i}"):
                st.markdown(metni_seslendir(message["content"]), unsafe_allow_html=True)

# Girdi Yönetimi
user_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.clicked_question if st.session_state.clicked_question else user_input
st.session_state.clicked_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Arka planda popülerliği güncelle (Hata korumalı)
    populer_soru_guncelle(prompt, embeddings_model)

    with st.chat_message("assistant"):
        with st.spinner("Hikmetli cevap hazırlanıyor..."):
            try:
                gecmis_diyalog = ""
                for m in st.session_state.messages[-5:-1]:
                    rol = "Kullanıcı" if m["role"] == "user" else "Asistan"
                    gecmis_diyalog += f"{rol}: {m['content']}\n"

                if vector_db is not None:
                    docs = vector_db.similarity_search(prompt, k=5)
                    baglam = "\n\n".join([f"[{os.path.basename(d.metadata['source'])}, S:{d.metadata.get('page', 0)+1}]: {d.page_content}" for d in docs])
                    kaynakca = "\n".join(set([f"- {os.path.basename(d.metadata['source']).replace('.pdf','')}" for d in docs]))
                else:
                    baglam = "Kaynaklara şu an ulaşılamıyor."; kaynakca = "Genel Bilgiler"

                asistan_prompt = f"Adın MUIN. Bilge asistansın.\nGeçmiş: {gecmis_diyalog}\nSoru: {prompt}\nKaynaklar: {baglam}\nTalimat: Yıldız kullanma. Kaynak belirt.\nKaynaklar: {kaynakca}"
                
                response = client.models.generate_content(model=GUNCEL_MODEL, contents=asistan_prompt)
                full_response = response.text
                st.markdown(full_response)
                
                if st.button("🔊 Dinle", key="current_btn"):
                    st.markdown(metni_seslendir(full_response), unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

            except Exception as e:
                st.error(f"Hata oluştu: {e}")