import streamlit as st
import os
import zipfile
import gdown
import base64
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# VERİTABANI İNDİRME SİSTEMİ (DRIVE)
# ==========================================
FILE_ID = '159Ttbvafpd5f51-BIeVD3N_WaFcgi14w'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
OUTPUT = 'veritabani.zip'
VERITABANI_YOLU = "./veritabani"

def veritabani_hazirla():
    # Eğer veritabanı klasörü yoksa Google Drive'dan indir
    if not os.path.exists(VERITABANI_YOLU):
        with st.spinner("Muin hazırlanıyor, kütüphane ilk kez indiriliyor... (Bu işlem bir defaya mahsustur)"):
            try:
                gdown.download(URL, OUTPUT, quiet=False)
                with zipfile.ZipFile(OUTPUT, 'r') as zip_ref:
                    zip_ref.extractall(".")
                if os.path.exists(OUTPUT):
                    os.remove(OUTPUT) # Zip dosyasını temizle
                st.success("Kütüphane başarıyla yüklendi!")
            except Exception as e:
                st.error(f"Veritabanı indirilirken hata oluştu: {e}")

# Uygulama başlar başlamaz kontrol et
veritabani_hazirla()

# ==========================================
# SAYFA AYARLARI & KARANLIK TEMA CSS
# ==========================================
st.set_page_config(page_title="MUIN", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; background-color: #1A1A1A; }
    p, span, label, .stMarkdown, h1, h2, h3 { color: #FFFFFF !important; }
    audio { width: 100%; height: 40px; margin-top: 10px; filter: invert(100%); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# AYARLAR & MODELLER
# ==========================================
API_ANAHTARIM = "AIzaSyDL6_xWVgg3EYeQmHm_wWoyBHfSSFl75HI"
GUNCEL_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=API_ANAHTARIM)

@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    # Veritabanı yolu yoksa hata verme, boş dön (yukarıdaki fonksiyonun bitmesini bekler)
    if not os.path.exists(VERITABANI_YOLU):
        return None
    return Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings)

vector_db = kaynaklari_yukle()

def metni_seslendir(metin):
    try:
        temiz_metin = metin.replace("*", "")
        tts = gTTS(text=temiz_metin, lang='tr', slow=False)
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        return f"Seslendirme hatası: {e}"

# ==========================================
# SOHBET ARAYÜZÜ & HAFIZA YÖNETİMİ
# ==========================================
st.title("🌙 MUIN: İslami Bilgi Asistanı")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj geçmişini göster
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if st.button(f"🔊 Dinle", key=f"btn_{i}"):
                st.markdown(metni_seslendir(message["content"]), unsafe_allow_html=True)

# Kullanıcı girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Hikmetli cevap hazırlanıyor..."):
            try:
                # Diyalog Geçmişi Oluşturma
                gecmis_diyalog = ""
                for m in st.session_state.messages[-5:-1]:
                    rol = "Kullanıcı" if m["role"] == "user" else "Asistan"
                    gecmis_diyalog += f"{rol}: {m['content']}\n"

                # Kaynak Çekme (Vector DB kontrolü ile)
                if vector_db is not None:
                    docs = vector_db.similarity_search(prompt, k=5)
                    baglam = "\n\n".join([f"[{os.path.basename(d.metadata['source'])}, S:{d.metadata.get('page', 0)+1}]: {d.page_content}" for d in docs])
                    kaynakca = "\n".join(set([f"- {os.path.basename(d.metadata['source']).replace('.pdf','')}" for d in docs]))
                else:
                    baglam = "Kaynak veritabanına şu an ulaşılamıyor."
                    kaynakca = "Genel İslami Bilgiler"

                asistan_prompt = f"""Sen derin ilmi bilgiye sahip bilge bir İslami asistansın (Adın: MUIN).
                
                DİYALOG GEÇMİŞİ:
                {gecmis_diyalog}

                YENİ SORU: {prompt}

                KAYNAK VERİLER:
                {baglam}

                TALİMATLAR:
                - Eğer 'Yeni Soru' önceki diyalogla bağlantılıysa (örneğin 'o olay', 'peki o zaman' gibi), diyalog geçmişini kullanarak cevap ver.
                - Cevabında asla yıldız (*) kullanma.
                - Kaynaklardaki ayet, hadis ve tefsiri analiz et.
                - İbadetin veya hükmün ruhani hikmetini (hikmet-i teşrii) açıkla.
                - Kaynaklara atıf yap.
                - Kaynakta yoksa genel prensiplerle mantıklı bir çıkarım yap.

                YARARLANILAN KAYNAKLAR:
                {kaynakca}"""

                response = client.models.generate_content(model=GUNCEL_MODEL, contents=asistan_prompt)
                full_response = response.text
                
                st.markdown(full_response)
                
                if st.button("🔊 Bu Cevabı Sesli Dinle", key="current_btn"):
                    st.markdown(metni_seslendir(full_response), unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Hata: {e}")