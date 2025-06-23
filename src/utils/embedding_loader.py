from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def embedding_loader():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
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