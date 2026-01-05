import streamlit as st
import os
import base64
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from gtts import gTTS

# ==========================================
# SAYFA AYARLARI & KARANLIK TEMA CSS (Aynen Korundu)
# ==========================================
st.set_page_config(page_title="İslami Asistan", page_icon="🌙", layout="centered")

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
VERITABANI_YOLU = "./veritabani"
GUNCEL_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=API_ANAHTARIM)

@st.cache_resource
def kaynaklari_yukle():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    if not os.path.exists(VERITABANI_YOLU): return None
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
st.title("🌙 İslami Bilge Asistan")

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
    # 1. Kullanıcı mesajını kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Hikmetli cevap hazırlanıyor..."):
            try:
                # 2. BAĞLAM OLUŞTURMA: Son 4 mesajı geçmiş olarak al
                # Bu, kullanıcının "peki bu olay..." dediğinde neyi kastettiğini anlamasını sağlar.
                gecmis_diyalog = ""
                for m in st.session_state.messages[-5:-1]: # Son soruyu dahil etme, öncekileri al
                    rol = "Kullanıcı" if m["role"] == "user" else "Asistan"
                    gecmis_diyalog += f"{rol}: {m['content']}\n"

                # 3. Kaynak Çekme
                docs = vector_db.similarity_search(prompt, k=5)
                baglam = "\n\n".join([f"[{os.path.basename(d.metadata['source'])}, S:{d.metadata['page']+1}]: {d.page_content}" for d in docs])
                kaynakca = "\n".join(set([f"- {os.path.basename(d.metadata['source']).replace('.pdf','')} (S: {d.metadata['page']+1})" for d in docs]))

                # 4. GÜNCELLENMİŞ HİKMETLİ PROMPT (Diyalog Geçmişi Eklendi)
                asistan_prompt = f"""Sen derin ilmi bilgiye sahip bilge bir İslami asistansın.
                
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
                if "429" in str(e): st.error("⚠️ Kota doldu, 1 dk bekleyin.")
                else: st.error(f"Hata: {e}")