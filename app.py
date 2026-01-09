# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import math
import time
import base64
import gdown
import zipfile
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# .env yükle (Lokalde .env'den, Streamlit Cloud'da Secrets'tan alır)
load_dotenv()

# ==========================================
# AYARLAR & API
# ==========================================
# ÖNEMLİ: Yeni API anahtarını Streamlit Secrets veya .env'e eklediğinden emin ol!
API_ANAHTARIM = os.getenv("GEMINI_API_KEY")
VERITABANI_YOLU = "./veritabanı"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

DRIVE_DOSYA_ID = "10fOIQH0dyG0tixnNjtVyEPipTS3EcT9k"
ZIP_ADI = "veritabani.zip"

# ==========================================
# GÖRSEL AYARLAR (ADIM 1: SABİTLİK VE RENK)
# ==========================================
st.set_page_config(page_title="MUIN", layout="centered")

st.markdown("""
    <style>
    /* Arka plan siyah, genel yazılar beyaz */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Üst Bölgeyi Sabitleyen CSS */
    .stMainBlockContainer {
        padding-top: 0rem !important;
    }
    
    [data-testid="stVerticalBlock"] > div:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #000000;
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #333;
    }

    /* Popüler Sorular: Beyaz Zemin, Siyah Yazı */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        padding: 10px;
    }
    
    /* Girdi Kutusu */
    [data-testid="stChatInput"] { 
        position: fixed; bottom: 20px; z-index: 1000; background-color: #FFFFFF !important; 
    }
    [data-testid="stChatInput"] textarea { color: #000000 !important; }
    
    /* Mesaj Balonları */
    .stChatMessage { background-color: #1A1A1A !important; border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# FONKSİYONLAR
# ==========================================
@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    if not os.path.exists(VERITABANI_YOLU) or not os.listdir(VERITABANI_YOLU):
        try:
            url = f'https://drive.google.com/uc?id={DRIVE_DOSYA_ID}'
            gdown.download(url, ZIP_ADI, quiet=True)
            with zipfile.ZipFile(ZIP_ADI, 'r') as z: z.extractall(".")
            if os.path.exists(ZIP_ADI): os.remove(ZIP_ADI)
        except: pass
    v_db = Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings)
    return v_db, embeddings

vector_db, embeddings_model = kaynaklari_yukle()

def populer_soru_guncelle(yeni_soru, model):
    if not yeni_soru or len(yeni_soru) < 10: return
    try:
        if os.path.exists(POPULER_SORULAR_DOSYASI):
            with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f: s_list = json.load(f)
        else: s_list = []
        y_vekt = model.embed_query(yeni_soru)
        bulundu = False
        for s in s_list:
            dot = sum(a*b for a,b in zip(y_vekt, s["vektor"]))
            mag = math.sqrt(sum(a*a for a in y_vekt)) * math.sqrt(sum(b*b for b in s["vektor"]))
            if (dot/mag) > 0.88: s["puan"] += 1; bulundu = True; break
        if not bulundu: s_list.append({"soru": yeni_soru, "puan": 1, "vektor": y_vekt})
        s_list = sorted(s_list, key=lambda x: x["puan"], reverse=True)[:15]
        with open(POPULER_SORULAR_DOSYASI, "w", encoding="utf-8") as f: json.dump(s_list, f, ensure_ascii=False)
    except: pass

# ==========================================
# 1. BÖLGE: SABİT ÜST PANEL
# ==========================================
with st.container():
    st.title("🌙 MUIN")
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f: pop = json.load(f)
        st.markdown("##### 🌟 Popüler Sorular")
        c1, c2 = st.columns(2)
        for i, k in enumerate(pop[:6]):
            with (c1 if i % 2 == 0 else c2):
                if st.button(f"🔍 {k['soru']}", key=f"p_{i}", use_container_width=True):
                    st.session_state.active_prompt = k['soru']
    st.divider()

# ==========================================
# 2. & 3. BÖLGE: SOHBET VE ZEKA
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "active_prompt" not in st.session_state: st.session_state.active_prompt = None

# Geçmişi Yazdır
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Girdi
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.active_prompt if st.session_state.active_prompt else u_input
st.session_state.active_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    populer_soru_guncelle(prompt, embeddings_model)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Bu aşamada basit bir yazı gösteriyoruz (2. adımda animasyon olacak)
        placeholder.markdown("🔍 *MUIN mütalaa ediyor...*")
        
        try:
            # Hafıza Penceresi (Son 10 mesaj)
            gecmis = st.session_state.messages[-11:-1]
            gecmis_text = "\n".join([f"{m['role']}: {m['content']}" for m in gecmis])
            
            docs = vector_db.similarity_search(prompt, k=6)
            baglam = "\n\n".join([f"📚 Kaynak: {os.path.basename(d.metadata['source'])}\n{d.page_content}" for d in docs])

            # System Instructions (Dokunulmaz Bölge)
            system_instructions = (
                "Sen bilge, nazik ve öğretici bir muallim olan MUIN'sin. "
                "Cevaplarına başlarken her seferinde farklı olacak şekilde; 'Selamünaleyküm kıymetli kardeşim', 'Aziz dostum merhaba' gibi samimi karşılamalar kullan. "
                "Soru hangi dildeyse o dilde cevap ver. "
                "ÖNEMLİ: Aşağıdaki GEÇMİŞ bölümündeki diyaloğu çok dikkatli incele. Eğer kullanıcı 'peki ya şu?', 'o ne demek?' gibi takip soruları soruyorsa, "
                "bir önceki cevabına ve kullanıcının niyetine sadık kalarak konuyu devam ettir. "
                "Öğretici, şefkatli ve derinlemesine bilgi veren bir üslup kullan. "
                "\n\nKAYNAK KURALI: Sadece ve sadece belgelerde bilgi varsa (📚 Kaynak: Dosya Adı) şeklinde atıf yap. "
                "Eğer bilgi belgelerde yoksa kendi bilgini hikmetle anlat. "
                "\n\nYıldız (*) karakterini asla kullanma, metni düz ve akıcı yaz. "
                "Cevapların sonunda kısa bir dua veya güzel bir temenni ile bitir."
            )
            
            full_query = f"{system_instructions}\n\nGEÇMİŞ DİYALOG:\n{gecmis_text}\n\nKAYNAKLAR:\n{baglam}\n\nSORU: {prompt}"
            res = client.models.generate_content(model=GUNCEL_MODEL, contents=full_query)
            
            placeholder.empty()
            st.markdown(res.text)
            st.session_state.messages.append({"role": "assistant", "content": res.text})
            
        except Exception as e:
            placeholder.error(f"Hata: {e}")