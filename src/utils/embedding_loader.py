import os
from dotenv import dotenv_values
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def embedding_loader():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print("Cargando embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": 32,
                "convert_to_numpy": True,
                "normalize_embeddings": True
            }
        )
        print("Embeddings configurados correctamente")
        return embeddings
    except Exception as e:
        print(f"Error al configurar embeddings: {e}")
        return
    
def llm_loader() -> ChatGoogleGenerativeAI:
    try:
        print("Cargando LLM...")
        config = dotenv_values(".env")
        os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY_1"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        print("LLM configurado correctamente")
        return llm
    except Exception as e:
        print(f"Error al configurar LLM: {e}")
        return None 