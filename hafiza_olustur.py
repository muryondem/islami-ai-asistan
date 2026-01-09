# -*- coding: utf-8 -*-
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
API_ANAHTARIM = os.getenv("GEMINI_API_KEY")

KAYNAK_ANA_DIZIN = "/Users/alem/Desktop/DiniAsistan/Kaynaklar"
VERITABANI_YOLU = "./veritabani"

def metadata_olustur(tam_yol):
    bagil_yol = os.path.relpath(tam_yol, KAYNAK_ANA_DIZIN)
    parcalar = bagil_yol.split(os.sep)
    meta = {"din": "Genel", "ana_mezhep": "Genel", "kategori": "Genel", "alt_mezhep": "Genel"}
    
    if len(parcalar) >= 1: meta["din"] = parcalar[0]
    if len(parcalar) >= 2: meta["ana_mezhep"] = parcalar[1]
    if len(parcalar) >= 3: meta["kategori"] = parcalar[2]
    if len(parcalar) >= 4: meta["alt_mezhep"] = parcalar[3]
    
    for key in meta:
        meta[key] = meta[key].replace("_", " ").capitalize()
    return meta

def hafiza_olustur():
    if not API_ANAHTARIM:
        print("❌ HATA: GEMINI_API_KEY bulunamadı!")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_ANAHTARIM)
    vector_db = Chroma(persist_directory=VERITABANI_YOLU, embedding_function=embeddings)

    existing_data = vector_db.get()
    processed_files = {os.path.basename(m['source']) for m in existing_data['metadatas'] if 'source' in m}

    new_documents = []
    for root, dirs, files in os.walk(KAYNAK_ANA_DIZIN):
        for dosya in files:
            if dosya.endswith(".pdf") and dosya not in processed_files:
                tam_yol = os.path.join(root, dosya)
                meta = metadata_olustur(tam_yol)
                try:
                    loader = PyPDFLoader(tam_yol)
                    docs = loader.load()
                    for d in docs:
                        d.metadata.update(meta)
                    new_documents.extend(docs)
                    print(f"📄 İşlendi: {dosya}")
                except Exception as e:
                    print(f"⚠️ Hata: {dosya} -> {e}")

    if new_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        parcalar = text_splitter.split_documents(new_documents)
        batch_size = 200
        for i in range(0, len(parcalar), batch_size):
            batch = parcalar[i:i + batch_size]
            try:
                vector_db.add_documents(documents=batch)
                print(f"✅ Paket: {i+len(batch)}/{len(parcalar)}")
            except Exception:
                time.sleep(30)
                vector_db.add_documents(documents=batch)
        print("✨ Hafıza güncellendi!")
    else:
        print("✔ Yeni döküman yok.")

if __name__ == "__main__":
    hafiza_olustur()