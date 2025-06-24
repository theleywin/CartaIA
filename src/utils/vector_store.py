import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
from utils import embedding_loader
from utils.chunking.chunking import chunk_docs

DEFAULT_PATH = "./data/faiss_vectorstore"

def load_vector_store(embeddings: HuggingFaceEmbeddings, path=DEFAULT_PATH): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        # Cargar con configuración de dispositivo
        if device == "cuda":
            vector_store = FAISS.load_local(
                path, 
                embeddings,
                allow_dangerous_deserialization=True,
                device="cuda"
            )
        else:
            vector_store = FAISS.load_local(
                path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
    else:
        raise FileNotFoundError(f"Vectorstore no encontrado en {path}. Ejecuta ingest.py primero.")
    return vector_store

def init_vector_store(embeddings: HuggingFaceEmbeddings, documents=None, path=DEFAULT_PATH):
    try:
        if documents is None or len(documents) == 0:
            documents = [Document(page_content="init")]
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(path)
        print(f"Ingesta completada. {len(documents)} fragmentos almacenados.")
    except Exception as e:
        print(f"Error al guardar vectorstore: {e}")
        raise
    

def update_vector_store(vector_store: FAISS, docs_str: List[str]):
    docs = [Document(page_content=doc) for doc in docs_str]   
    documents = chunk_docs(docs, 1000, ["\n\n", "\n", " ", ""], overlap_ratio=0.1)
    vector_store.add_documents(documents)
    vector_store.save_local("./data/faiss_vectorstore")