import os
import warnings
from google import genai 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from thefuzz import process, fuzz

warnings.filterwarnings("ignore")

# ==========================================
# AYARLAR
# ==========================================
API_ANAHTARIM = "AIzaSyDL6_xWVgg3EYeQmHm_wWoyBHfSSFl75HI"
VERITABANI_YOLU = "./veritabani"

client = genai.Client(api_key=API_ANAHTARIM)
GUNCEL_MODEL = "gemini-2.0-flash"

ANAHTAR_TERIMLER = [
    "Namaz", "Zekat", "Oruç", "Hac", "Kurban", "Farz", "Sünnet", 
    "Vacip", "Müstehap", "Mekruh", "Haram", "Helal", "Tefsir", 
    "Hadis", "Ayet", "Sure", "Fıkıh", "Akide", "Sahih", "Rivayet"
]
# ==========================================

def asistan_baslat():
    try:
        print(f"Sistem hafızası yükleniyor... ({GUNCEL_MODEL} aktif)")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_ANAHTARIM
        )
        
        if not os.path.exists(VERITABANI_YOLU):
            print(f"HATA: {VERITABANI_YOLU} bulunamadı!")
            return

        vector_db = Chroma(
            persist_directory=VERITABANI_YOLU, 
            embedding_function=embeddings
        )

        print("\n" + "="*40)
        print("ÇOK DİLLİ İSLAMİ ASİSTAN (Hikmet Odaklı) HAZIR")
        print("="*40)

        while True:
            ham_soru = input("\nSorunuz / Question / Sual (Çıkış için 'exit'): ")
            if ham_soru.lower() == 'exit': break
            if not ham_soru.strip(): continue

            # 1. ADIM: Fuzzy Matching
            kelimeler = ham_soru.split()
            fuzzy_oneriler = []
            for kelime in kelimeler:
                match, score = process.extractOne(kelime, ANAHTAR_TERIMLER, scorer=fuzz.WRatio)
                if 80 <= score < 100:
                    fuzzy_oneriler.append(f"'{kelime}' -> '{match}'")

            if fuzzy_oneriler:
                print(f"[Fuzzy Match]: {', '.join(fuzzy_oneriler)}")

            # 2. ADIM: Çok Dilli Yazım Denetimi
            print("Kontrol ediliyor...")
            # Bu promptu dili otomatik algılayıp düzeltecek şekilde güncelledik.
            denetim_prompt = f"""Şu soruyu analiz et: "{ham_soru}"
            1. Eğer yazım hatası varsa, sorunun sorulduğu orijinal dilde düzelt.
            2. Hata yoksa sadece 'TAMAM' yaz.
            Düzeltme yaparsan sadece düzeltilmiş halini yaz, başka açıklama ekleme."""
            
            denetim_yaniti = client.models.generate_content(
                model=GUNCEL_MODEL,
                contents=denetim_prompt
            )
            denetim_sonucu = denetim_yaniti.text.strip()

            final_soru = ham_soru
            if "TAMAM" not in denetim_sonucu.upper() and denetim_sonucu.upper() != ham_soru.upper():
                print(f"\nDüzeltme Önerisi: {denetim_sonucu}")
                onay = input("(Onayla: 'e', Atla: 'h'): ")
                if onay.lower() == 'e':
                    final_soru = denetim_sonucu

            # 3. ADIM: Kaynaktan Bilgi Çekme
            print("Kaynaklar taranıyor...")
            docs = vector_db.similarity_search(final_soru, k=5)
            baglam = "\n\n".join([f"[{doc.metadata.get('kategori', 'genel').upper()}]: {doc.page_content}" for doc in docs])

            # 4. ADIM: Final Cevap Üretimi (Mantık + Çok Dilli Katman)
            asistan_prompt = f"""Sen derin ilmi bilgiye sahip, bilge bir İslami asistansın. 
            
            ÖNEMLİ DİL KURALI: Kullanıcı soruyu hangi dilde sorduysa ({final_soru}), kesinlikle o dilde cevap ver. 
            Kaynaklar Türkçe olsa bile, cevabını sorunun diline (İngilizce, Arapça vb.) çevirerek sentezle.

            Görevin sadece kaynakları kopyalamak değil, bu kaynaklardaki bilgileri kullanarak 
            kullanıcının sorusuna hikmetli, doyurucu ve bütünsel bir cevap üretmektir.

            KAYNAK VERİLER:
            {baglam}

            SORU: {final_soru}

            CEVAP STRATEJİN:
            1. ÖNCE ANALİZ ET: Kaynaklardaki ayet, hadis ve tefsir notlarını birleştir. 
            2. HİKMETİ AÇIKLA: Soru "neden" diye soruyorsa, ibadetin sadece şeklini değil, ruhani ve ahlaki gerekçelerini (hikmet-i teşrii) anlat.
            3. KAYNAK GÖSTER: "X ayetinde buyurulduğu üzere..." gibi ifadelerle argümanlarını temellendir.
            4. YORUMLA: Akıl ve kalp dengesini kuran bir yorum ekle.
            5. KAYNAKTA YOKSA: "Kaynaklarımda doğrudan geçmemekle birlikte..." notunu düşerek genel İslami prensiplerle çıkarım yap.

            Lütfen soruyu bir öğretici samimiyetiyle cevapla:"""

            cevap = client.models.generate_content(
                model=GUNCEL_MODEL,
                contents=asistan_prompt
            )
            
            print(f"\nCEVAP:\n{cevap.text}")

    except Exception as e:
        print(f"\nBİR HATA OLUŞTU: {e}")

if __name__ == "__main__":
    asistan_baslat()