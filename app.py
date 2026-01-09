# -*- coding: utf-8 -*-
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
import math
import base64
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# .env dosyasını yükle
load_dotenv()

# ==========================================
# AYARLAR & API
# ==========================================
API_ANAHTARIM = os.getenv("GEMINI_API_KEY")
VERITABANI_YOLU = "./veritabanı"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

# ==========================================
# FONKSİYONLAR (HAFIZA, KAYIT VE ZEKA)
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
        soru_listesi = sorted(soru_listesi, key=lambda x: x["puan"], reverse=True)[:30]
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

def metni_seslendir(metin, dil='tr'):
    try:
        tts = gTTS(text=metin.replace("*", ""), lang=dil, slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return ""

# ==========================================
# CSS (EKRAN BÖLME VE GENİŞLETİLMİŞ GİRİŞ)
# ==========================================
st.set_page_config(page_title="MUIN Test Paneli", layout="centered")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        background-color: #000000;
    }

    /* Ana metin, paragraflar ve asistan balonlarını beyaza zorla */
    .stApp, p, li, h1, h2, h3, span {
        color: #FFFFFF !important;
    }

    /* Soru Giriş Kutusu (Input Area) - Sabitleme ve Arka Plan */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;
        z-index: 10000;
        width: 94% !important;
        left: 3% !important;
        background-color: #FFFFFF !important; /* Kutu beyaz kalsın */
        border-radius: 10px;
    }

    /* Giriş kutusunun içindeki YAZI RENGİ - SİYAH */
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important; /* iPhone/Android için siyah harf zorlaması */
    }
    
    /* Sohbet Balonları */
    .stChatMessage { 
        border-radius: 15px; 
        background-color: #1A1A1A !important; 
    }
    
    /* Balon içindeki metinler beyaz kalsın */
    [data-testid="stChatMessageContent"] p {
        color: #FFFFFF !important;
    }

    .stButton>button { 
        border-radius: 15px; 
        background-color: #1A1A1A; 
        border: 1px solid #444; 
        color: #FFFFFF !important; 
        font-size: 13px; 
    }
    
    /* Kenar çubuğu (Sidebar) metinleri */
    [data-testid="stSidebar"] section {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. BÖLGE: ÜST (POPLER SORULAR - 10 SORU GÖRÜNÜR)
# ==========================================
with st.container():
    st.title("🌙 MUIN")
    populer_listesi = populer_sorulari_getir()
    if "clicked_q" not in st.session_state: st.session_state.clicked_q = None

    if populer_listesi:
        st.markdown("##### 🌟 Popüler Sorular")
        # İlk 10 soruyu 2 sütunda gösteriyoruz (Alan dinamik büyür)
        ana_sorular = populer_listesi[:10]
        c1, c2 = st.columns(2)
        for i, k in enumerate(ana_sorular):
            with (c1 if i % 2 == 0 else c2):
                if st.button(f"🔍 {k['soru']}", key=f"top_{i}", use_container_width=True):
                    st.session_state.clicked_q = k['soru']
        
        # 10'dan fazla varsa "Daha Fazla" seçeneği
        if len(populer_listesi) > 10:
            with st.expander("Daha Fazla Popüler Soru..."):
                d1, d2 = st.columns(2)
                for i, k in enumerate(populer_listesi[10:20]):
                    with (d1 if i % 2 == 0 else d2):
                        if st.button(f"🔍 {k['soru']}", key=f"extra_{i}", use_container_width=True):
                            st.session_state.clicked_q = k['soru']
    st.divider()

# ==========================================
# 2. BÖLGE: ALT (KAYDIRILABİLİR CHAT)
# ==========================================
chat_area = st.container(height=480) # Bağımsız kaydırma alanı

if "messages" not in st.session_state: st.session_state.messages = []

with chat_area:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and st.button("🔊 Dinle", key=f"voice_{i}"):
                st.markdown(metni_seslendir(m["content"]), unsafe_allow_html=True)

# ==========================================
# 3. BÖLGE: GİRDİ VE ZEKA
# ==========================================
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.clicked_q if st.session_state.clicked_q else u_input
st.session_state.clicked_q = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_prompt = st.session_state.messages[-1]["content"]
    
    # Soru kaydediliyor
    populer_soru_guncelle(current_prompt, embeddings_model)
    
    with chat_area:
        with st.chat_message("assistant"):
            with st.spinner("MUIN mütalaa ediyor..."):
                try:
                    gecmis = st.session_state.messages[-6:-1]
                    gecmis_text = "\n".join([f"{m['role']}: {m['content']}" for m in gecmis])

                    if vector_db:
                        docs = vector_db.similarity_search(current_prompt, k=6)
                        baglam = "\n\n".join([f"📚 Kaynak: {os.path.basename(d.metadata['source'])}\n{d.page_content}" for d in docs])
                    else: baglam = "Belge bulunamadı."

                    # TAM VE EKSİKSİZ PROMPT (SENİN VERDİĞİN METİN)
                    system_instructions = (
                        "Sen bilge, nazik ve öğretici bir muallim olan MUIN'sin. "
                        "Cevaplarına başlarken her seferinde farklı olacak şekilde; 'Selamünaleyküm kıymetli kardeşim', 'Aziz dostum merhaba' gibi samimi karşılamalar kullan. "
                        "Soru hangi dildeyse o dilde cevap ver. Diyalog geçmişini hatırla. "
                        "Öğretici, şefkatli ve derinlemesine bilgi veren bir üslup kullan. "
                        "\n\nKAYNAK KURALI: Sadece ve sadece belgelerde bilgi varsa (📚 Kaynak: Dosya Adı) şeklinde atıf yap. "
                        "Eğer bilgi belgelerde yoksa 'Kaynak yok' veya 'Belgelerde bulamadım' gibi bir ifade asla kullanma, doğrudan kendi bilgini hikmetle anlat. "
                        "\n\nYıldız (*) karakterini asla kullanma, metni düz ve akıcı yaz. "
                        "Cevapların sonunda kısa bir dua veya güzel bir temenni ile bitir."
                    )
                    
                    full_query = f"{system_instructions}\n\nGEÇMİŞ:\n{gecmis_text}\n\nKAYNAKLAR:\n{baglam}\n\nSORU: {current_prompt}"
                    
                    res = client.models.generate_content(model=GUNCEL_MODEL, contents=full_query)
                    st.markdown(res.text)
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Hata: {e}")