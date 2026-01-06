# -*- coding: utf-8 -*-
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

KAYNAK_ANA_DIZIN = "./kaynaklar"
VERITABANI_YOLU = "./veritabani"

def hafiza_olustur():
    # 1. API Anahtarını hazırla
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"DEBUG: Anahtar bulundu mu?: {api_key is not None}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # 2. Mevcut Veritabanını yükle
    vector_db = Chroma(
        persist_directory=VERITABANI_YOLU,
        embedding_function=embeddings
    )

    # 3. Zaten işlenmiş dosyaları listele
    existing_data = vector_db.get()
    processed_files = set()
    if existing_data['metadatas']:
        for m in existing_data['metadatas']:
            if 'source' in m:
                processed_files.add(os.path.basename(m['source']))

    new_documents = []
    
    # 4. Kaynak klasörlerini tara
    for kategori in ["meal", "tefsir", "hadis"]:
        yol = os.path.join(KAYNAK_ANA_DIZIN, kategori)
        if not os.path.exists(yol): continue
        
        for dosya in os.listdir(yol):
            if dosya.endswith(".pdf"):
                if dosya in processed_files:
                    print(f"Atlanıyor (Zaten Hafızada): {dosya}")
                    continue
                
                print(f"YENİ DOSYA İşleniyor: {dosya}")
                try:
                    loader = PyPDFLoader(os.path.join(yol, dosya))
                    docs = loader.load()
                    for d in docs:
                        d.metadata["kategori"] = kategori
                    new_documents.extend(docs)
                except Exception as e:
                    print(f"HATA: {dosya} okunurken sorun çıktı: {e}")

    # 5. Yeni dosyaları küçük paketler halinde ve bekleyerek ekle
    if new_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        parcalar = text_splitter.split_documents(new_documents)
        
        total_parts = len(parcalar)
        print(f"Toplam {total_parts} parça 50'şerli paketler halinde ekleniyor...")
        
        batch_size = 50 
        for i in range(0, total_parts, batch_size):
            batch = parcalar[i:i + batch_size]
            print(f"İşleniyor: {i} - {min(i + batch_size, total_parts)} / {total_parts}")
            
            basarili = False
            deneme = 0
            while not basarili and deneme < 5:
                try:
                    vector_db.add_documents(documents=batch)
                    basarili = True
                    # Her paketten sonra 10 saniye mola (Kota koruması)
                    time.sleep(10) 
                except Exception as e:
                    deneme += 1
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        bekleme_suresi = 65 * deneme
                        print(f"Kota doldu, {bekleme_suresi} saniye mola veriliyor (Deneme {deneme})...")
                        time.sleep(bekleme_suresi)
                    else:
                        print(f"Hata oluştu: {e}")
                        break
        
        print("Hafıza başarıyla güncellendi!")
    else:
        print("Yeni dosya yok, hafıza zaten güncel.")

if __name__ == "__main__":
    hafiza_olustur()