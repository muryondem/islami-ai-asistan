# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
import math
import base64
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# AYARLAR & API (DOKUNULMADI)
# ==========================================
API_ANAHTARIM = st.secrets["GEMINI_API_KEY"]
VERITABANI_YOLU = "./veritabanı"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

# ==========================================
# FONKSİYONLAR (HAFIZA VE REFERANS KORUNDU)
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

def metni_seslendir(metin, dil='tr'):
    try:
        tts = gTTS(text=metin.replace("*", ""), lang=dil, slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return ""

# ==========================================
# CSS (KESİN BÖLÜNMÜŞ EKRAN TASARIMI)
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="wide") # Geniş mod daha ferah olur

st.markdown("""
    <style>
    /* Sayfanın genel kaymasını engelle */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        background-color: #000000;
    }

    /* Üst Alan: Popüler Sorular (Dinamik Yükseklik) */
    .top-panel {
        background-color: #000000;
        border-bottom: 2px solid #333;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Alt Alan: Chat (Sabit Yükseklik ve Kaydırılabilir) */
    .chat-scroll {
        overflow-y: auto;
        height: 60vh; /* Ekranın %60'ı chat alanı */
        padding: 10px;
        border: 1px solid #222;
        border-radius: 10px;
    }

    .stChatMessage { border-radius: 15px; background-color: #1A1A1A !important; }
    .stButton>button { border-radius: 15px; background-color: #1A1A1A; border: 1px solid #444; color: white !important; font-size: 12px; }
    
    /* Soru Giriş Alanını Sabitle */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;
        z-index: 9999;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. BÖLGE: ÜST (ESNEK POPÜLER SORULAR)
# ==========================================
top_container = st.container()
with top_container:
    st.title("🌙 MUIN")
    populer_listesi = populer_sorulari_getir()
    if "clicked_q" not in st.session_state: st.session_state.clicked_q = None

    if populer_listesi:
        st.markdown("##### 🌟 Popüler Sorular")
        
        # İlk 10 soruyu göster
        ana_sorular = populer_listesi[:10]
        c1, c2, c3 = st.columns(3) # 3 sütun yaparak alanı daha iyi kullanıyoruz
        for i, k in enumerate(ana_sorular):
            col = [c1, c2, c3][i % 3]
            with col:
                if st.button(f"🔍 {k['soru']}", key=f"top_{i}", use_container_width=True):
                    st.session_state.clicked_q = k['soru']
        
        # 10'dan fazla varsa "Daha Fazla" expander içine al
        if len(populer_listesi) > 10:
            with st.expander("➕ Daha Fazla Popüler Soru Gör"):
                d1, d2, d3 = st.columns(3)
                for i, k in enumerate(populer_listesi[10:30]): # 30'a kadar destekle
                    col = [d1, d2, d3][i % 3]
                    with col:
                        if st.button(f"🔍 {k['soru']}", key=f"extra_{i}", use_container_width=True):
                            st.session_state.clicked_q = k['soru']

st.divider()

# ==========================================
# 2. BÖLGE: ALT (KAYDIRILABİLİR CHAT)
# ==========================================
# Container height kullanımı en güvenli bölünmüş ekran yöntemidir
chat_area = st.container(height=450) # Burası kendi içinde kayar

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
    
    with chat_area:
        with st.chat_message("assistant"):
            with st.spinner("MUIN mütalaa ediyor..."):
                try:
                    # HAFIZA: Son konuşmaları hatırla
                    gecmis = st.session_state.messages[-6:-1]
                    gecmis_text = "\n".join([f"{m['role']}: {m['content']}" for m in gecmis])

                    # RAG: Kaynaklardan bul
                    if vector_db:
                        docs = vector_db.similarity_search(current_prompt, k=6)
                        baglam = "\n\n".join([f"📚 Kaynak: {os.path.basename(d.metadata['source'])}\n{d.page_content}" for d in docs])
                    else: baglam = "Belge bulunamadı."

                    # MUIN KİMLİĞİ VE PROMPT
                    system_msg = (
                        "Sen bilge, nazik ve öğretici bir muallim olan MUIN'sin. "
                        "Cevaplarına başlarken her seferinde farklı olacak şekilde; 'Selamünaleyküm kıymetli kardeşim', 'Aziz dostum merhaba', "
                        "Soru hangi dildeyse o dilde cevap ver. "
                        "Diyalog geçmişini hatırla ve kaynaklara mutlaka (📚 Kaynak: Dosya Adı) şeklinde atıf yap. "
                        "Öğretici, şefkatli ve derinlemesine bilgi veren bir üslup kullan. "
                        "Mutlaka kaynaklara atıf yap (Kaynak: Dosya Adı şeklinde).\n"
                        "Eğer kaynaklarda bilgi kısıtlıysa, genel İslami bilgini kullanarak konuyu derinleştir ve 'Komşuluk', 'Ahlak' gibi konularda öğretici bir ders verir gibi anlat.\n"
                        "Yıldız (*) karakterini asla kullanma, metni düz ve akıcı yaz.\n"
                        "Cevapların sonunda kısa bir dua veya güzel bir temenni ile bitir."
                    )
                    
                    full_query = f"{system_msg}\n\nGEÇMİŞ:\n{gecmis_text}\n\nKAYNAKLAR:\n{baglam}\n\nSORU: {current_prompt}"
                    
                    res = client.models.generate_content(model=GUNCEL_MODEL, contents=full_query)
                    st.markdown(res.text)
                    st.session_state.messages.append({"role": "assistant", "content": res.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Hata: {e}")