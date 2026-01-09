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
from gtts import gTTS

# .env yükle
load_dotenv()

# ==========================================
# AYARLAR & API
# ==========================================
API_ANAHTARIM = os.getenv("GEMINI_API_KEY")
VERITABANI_YOLU = "./veritabanı"
POPULER_SORULAR_DOSYASI = "populer_sorular.json"
GUNCEL_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=API_ANAHTARIM)

DRIVE_DOSYA_ID = "10fOIQH0dyG0tixnNjtVyEPipTS3EcT9k"
ZIP_ADI = "veritabani.zip"

# ==========================================
# GÖRSEL AYARLAR (RENKLER HEP BEYAZ & SABİT ÜST)
# ==========================================
st.set_page_config(page_title="MUIN", layout="centered")

st.markdown("""
    <style>
    /* Arka plan siyah, yazılar hep beyaz */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    p, li, h1, h2, h3, span, div, label { color: #FFFFFF !important; }
    
    /* Mesaj Balonları */
    .stChatMessage { background-color: #1A1A1A !important; border-radius: 15px; margin-bottom: 10px; }
    
    /* Girdi Kutusu (Yazı Siyah, Arka Plan Beyaz) */
    [data-testid="stChatInput"] { 
        position: fixed; bottom: 20px; z-index: 1000; background-color: #FFFFFF !important; border-radius: 10px;
    }
    [data-testid="stChatInput"] textarea { color: #000000 !important; }
    
    /* Üst Bölge Sabitleme Stili */
    .stMainBlockContainer { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# FONKSİYONLAR
# ==========================================
@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    if not os.path.exists(VERITABANI_YOLU) or not os.listdir(VERITABANI_YOLU):
        url = f'https://drive.google.com/uc?id={DRIVE_DOSYA_ID}'
        try:
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
            similarity = dot/mag if mag != 0 else 0
            if similarity > 0.88: s["puan"] += 1; bulundu = True; break
        if not bulundu: s_list.append({"soru": yeni_soru, "puan": 1, "vektor": y_vekt})
        s_list = sorted(s_list, key=lambda x: x["puan"], reverse=True)[:15]
        with open(POPULER_SORULAR_DOSYASI, "w", encoding="utf-8") as f: json.dump(s_list, f, ensure_ascii=False)
    except: pass

# ==========================================
# 1. BÖLGE: SABİT ÜST PANEL
# ==========================================
# Bu container sohbet akışından bağımsız olarak en üstte kalır
with st.container():
    st.title("🌙 MUIN")
    if os.path.exists(POPULER_SORULAR_DOSYASI):
        try:
            with open(POPULER_SORULAR_DOSYASI, "r", encoding="utf-8") as f: pop = json.load(f)
            st.markdown("##### 🌟 Popüler Sorular")
            cols = st.columns(2)
            for i, k in enumerate(pop[:6]):
                if cols[i%2].button(f"🔍 {k['soru']}", key=f"p_{i}", use_container_width=True):
                    st.session_state.active_prompt = k['soru']
        except: pass
    st.divider()

# ==========================================
# 2. BÖLGE: SOHBET VE HAFIZA
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "active_prompt" not in st.session_state: st.session_state.active_prompt = None

# Mevcut Sohbeti Yazdır
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Girdi Yönetimi
u_input = st.chat_input("Sorunuzu buraya yazın...")
prompt = st.session_state.active_prompt if st.session_state.active_prompt else u_input
st.session_state.active_prompt = None

if prompt:
    # 1. Kullanıcı sorusunu hafızaya ekle ve anında göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    populer_soru_guncelle(prompt, embeddings_model)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. MUIN Mütalaa ve Cevap Süreci
    with st.chat_message("assistant"):
        mütalaa_alani = st.empty()
        
        # 12 NOKTA ANİMASYONU (İşlem sürerken kullanıcıyı bilgilendirir)
        for n in range(1, 13):
            mütalaa_alani.markdown(f"🔍 *MUIN mütalaa ediyor{'.' * n}*")
            time.sleep(0.1)
        
        try:
            # GEÇMİŞ HAFIZA: Son 10 mesaj (Vazgeçilmez Standart)
            # (Listenin sonundaki kullanıcı sorusunu dahil etmemek için -11:-1)
            gecmis = st.session_state.messages[-11:-1]
            gecmis_text = "\n".join([f"{m['role']}: {m['content']}" for m in gecmis])
            
            # Kaynak Arama (RAG)
            docs = vector_db.similarity_search(prompt, k=6)
            baglam = "\n\n".join([f"📚 Kaynak: {os.path.basename(d.metadata['source'])}\n{d.page_content}" for d in docs])

            # SYSTEM INSTRUCTIONS (Hassas Bölge: Tek harf değişmedi)
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
            
            # Sorguyu Gönder
            full_query = f"{system_instructions}\n\nGEÇMİŞ DİYALOG:\n{gecmis_text}\n\nKAYNAKLAR:\n{baglam}\n\nSORU: {prompt}"
            res = client.models.generate_content(model=GUNCEL_MODEL, contents=full_query)
            
            # Animasyonu temizle ve cevabı yaz (Donma ve renk hatası engellendi)
            mütalaa_alani.empty()
            st.markdown(res.text)
            
            # Cevabı hafızaya kaydet
            st.session_state.messages.append({"role": "assistant", "content": res.text})
            
        except Exception as e:
            mütalaa_alani.error(f"Bir hata oluştu: {e}")