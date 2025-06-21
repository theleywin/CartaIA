import os
from langchain_community.vectorstores import FAISS


def load_vector_store(device, embeddings): 
    vector_store_path = "./data/faiss_vectorstore"
    if os.path.exists(vector_store_path):
        # Cargar con configuración de dispositivo
        if device == "cuda":
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings,
                allow_dangerous_deserialization=True,
                device="cuda"
            )
        else:
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
    else:
        raise FileNotFoundError(f"Vectorstore no encontrado en {vector_store_path}. Ejecuta ingest.py primero.")
    
    return vector_store