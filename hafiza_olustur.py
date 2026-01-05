import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

API_KEY = "AIzaSyDL6_xWVgg3EYeQmHm_wWoyBHfSSFl75HI"
KAYNAK_ANA_DIZIN = "./kaynaklar"
VERITABANI_YOLU = "./veritabani"

def hafiza_olustur():
    all_documents = []
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    
    # Klasörleri tara (meal, tefsir, hadis)
    for kategori in ["meal", "tefsir", "hadis"]:
        yol = os.path.join(KAYNAK_ANA_DIZIN, kategori)
        if not os.path.exists(yol): continue
        
        for dosya in os.listdir(yol):
            if dosya.endswith(".pdf"):
                print(f"İşleniyor: {kategori} -> {dosya}")
                loader = PyPDFLoader(os.path.join(yol, dosya))
                # Her parçaya hangi kategoriden geldiğini etiket olarak ekle
                docs = loader.load()
                for d in docs:
                    d.metadata["kategori"] = kategori
                all_documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parcalar = text_splitter.split_documents(all_documents)

    vector_db = Chroma.from_documents(
        documents=parcalar,
        embedding=embeddings,
        persist_directory=VERITABANI_YOLU
    )
    print("Üçlü kaynak hafızası başarıyla oluşturuldu!")

if __name__ == "__main__":
    hafiza_olustur()