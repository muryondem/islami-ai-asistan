# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from models import KullaniciProfil, SoruIstegi

# 1. Ayarlar ve Çevresel Değişkenler
load_dotenv()
API_ANAHTARIM = os.getenv("GEMINI_API_KEY")
VERITABANI_YOLU = "./veritabani"

app = FastAPI(title="MUIN AI Backend")

# Gemini Client Başlatma
if API_ANAHTARIM:
    client = genai.Client(api_key=API_ANAHTARIM)
else:
    print("⚠️ HATA: .env dosyasında GEMINI_API_KEY bulunamadı!")

# Örnek Kullanıcı Profili (Test amaçlı sabitlenmiştir)
fake_db_users = {
    "test_user": KullaniciProfil(
        id="test_user", 
        isim="Ahmet", 
        din="Islam", 
        mezhep="Hanefi", 
        ilgi_alanlari=["Maneviyat"],
        derinlik_seviyesi="Öğretici"
    )
}

def kaynak_getir_katmanli(soru, profil: KullaniciProfil):
    """Veritabanından kullanıcının din ve mezhebine göre kaynak getirir."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    
    if not os.path.exists(VERITABANI_YOLU):
        print("⚠️ Veritabanı klasörü bulunamadı!")
        return ""
    
    v_db = Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings)
    
    # Filtreleri hazırlarken büyük/küçük harf duyarlılığına dikkat
    din_filtre = profil.din.capitalize() # Örn: Islam
    mezhep_filtre = profil.mezhep.capitalize() # Örn: Hanefi
    
    # Katmanlı Mezhep Listesi
    mezhep_listesi = [mezhep_filtre, "Genel", "Islam ahlaki", "Islam Ahlaki"]
    
    # Arama Filtresi (Metadata bazlı)
    search_filter = {
        "$and": [
            {"din": din_filtre},
            {"alt_mezhep": {"$in": mezhep_listesi}}
        ]
    }
    
    print(f"🔍 Arama yapılıyor... Filtre: {search_filter}")
    
    try:
        docs = v_db.similarity_search(soru, k=5, filter=search_filter)
        if not docs:
            print("❗ Eşleşen belge bulunamadı. Filtresiz genel arama deneniyor...")
            docs = v_db.similarity_search(soru, k=3)
            
        print(f"📚 Bulunan Kaynak Sayısı: {len(docs)}")
        
        kaynak_metni = ""
        for d in docs:
            dosya_adi = os.path.basename(d.metadata.get('source', 'Bilinmeyen Dosya'))
            kaynak_metni += f"\n--- (📚 Kaynak: {dosya_adi}) ---\n{d.page_content}\n"
        
        return kaynak_metni
    except Exception as e:
        print(f"❌ Arama sırasında hata: {e}")
        return ""

@app.post("/muin_sor")
async def muin_sor(istek: SoruIstegi):
    profil = fake_db_users.get(istek.user_id)
    if not profil:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")

    try:
        # Kaynakları getir
        baglam = kaynak_getir_katmanli(istek.soru, profil)
        
        # Debug: Terminale bağlamın durumunu yazdır
        if baglam:
            print(f"✅ Bağlam hazır (İlk 100 karakter): {baglam[:100]}...")
        else:
            print("⚠️ Bağlam boş! Gemini kendi bilgisiyle cevap verecek.")

        # Diyalog geçmişini sınırla
        gecmis_metni = "\n".join([f"{m['role']}: {m['content']}" for m in istek.gecmis[-5:]])
        
        # SYSTEM PROMPT (Senin talimatlarınla optimize edildi)
        system_instructions = (
            f"Sen bilge, nazik ve öğretici bir muallim olan MUIN'sin. Kullanıcının adı {profil.isim}. "
            f"Öğretici üslubu ile cevapla. Cevaplarına samimi bir selamla başla. "
            "ÖNEMLİ: Aşağıda sana 'KAYNAKLAR' başlığı altında teknik metinler verilecek.\n"
            "KURAL 1: Cevabını oluştururken EĞER KAYNAKLARDA BİLGİ VARSA mutlaka o bilgiyi kullan.\n"
            "KURAL 2: Bilgiyi kullandığın cümlenin sonuna (📚 Kaynak: Dosya Adı) ekle. Bu senin en büyük önceliğindir.\n"
            "KURAL 3: Yıldız (*) karakterini ASLA ama asla kullanma. Metni dümdüz yaz.\n"
            "KURAL 4: Eğer kaynaklarda bilgi yoksa, kendi ilminle cevap ver ama asla sahte kaynak uydurma."
            "KESİN YASAK: Yıldız (*) karakterini asla kullanma. Metni dümdüz yaz. Kalınlaştırma (bold) yapma. "
            "NUMARALANDIRMA: Liste yapacaksan 1. 2. 3. şeklinde rakam kullan. "
            "DİL: Soru hangi dildeyse o dilde cevap ver. "
            f"\n\nKAYNAKLAR:\n{baglam}\n\n"
            "ATIF KURALI (KRİTİK): Aşağıdaki KAYNAKLAR bölümünden aldığın her bilginin sonuna, "
            "o bilginin ait olduğu dosya adını (📚 Kaynak: Dosya Adı) şeklinde ekle. "
            "Bilgi kaynağını belirtmek senin en önemli görevindir. Kaynakları görmezden gelme.\n\n"
            f"KAYNAKLAR:\n{baglam}\n\n"
            "\n\nCevabı her zaman güzel bir dua veya temenni ile bitir."
        )
        
        # Soru ve Kaynakları birbirinden çok net ayırıyoruz
        user_input = (
            f"AŞAĞIDAKİ KAYNAKLARI KULLANARAK SORUYU CEVAPLA:\n\n"
            f"KAYNAKLAR:\n{baglam}\n\n"
            f"SORU: {istek.soru}"
        )
        
        res = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[system_instructions, user_input], # İkisini ayrı parçalar olarak gönderiyoruz
            config={
                "temperature": 0.0, # Sıfır yaratıcılık, tam sadakat
                "top_p": 1.0,
            }
        )
        
        return {"cevap": res.text, "isim": profil.isim}

    except Exception as e:
        print(f"❌ HATA DETAYI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 sayesinde ağdaki diğer cihazlar (Windows/Telefon) erişebilir
    uvicorn.run(app, host="0.0.0.0", port=8000)