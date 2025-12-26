import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings


# Configuration
DOCS_PATH = "Data"  # Folder jahan saari AI/ML PDFs hain
DB_PATH = "full_nexus_ai_index"

def create_vector_db():
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"📁 '{DOCS_PATH}' folder ban gaya hai. Isme apni PDFs daalein!")
        return

    print("📄 Loading all AI/ML documents from folder...")
    # DirectoryLoader saare PDFs ko ek sath uthayega
    loader = DirectoryLoader(DOCS_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # AI/ML concepts ke liye 1000 chunk size best hai
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    print(f"🧩 Creating Mega Index for {len(chunks)} chunks... Thoda time lag sakta hai.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # FAISS index banana
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print(f"🎉 Nexus AI Database Saved at '{DB_PATH}'!")

if __name__ == "__main__":
    create_vector_db()