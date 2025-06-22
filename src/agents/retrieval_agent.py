# agents/retrieval_agent.py
from schemas.estado import EstadoConversacion
from langchain_community.vectorstores import FAISS
from src.utils.crawler import Crawler

THRESHOLD = 0.5

def crear_agente_retrieval(db: FAISS):
    crawler = Crawler()
    def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        documentos = db.similarity_search_with_score(estado.tema, k=10)
        filtered_docs = [doc.page_content for doc, score in documentos if score > THRESHOLD]
        
        if len(filtered_docs) < 4:
            raise ValueError(f"No se encontraron documentos relevantes para el tema: {estado.tema}")
        
        estado.docs_relevantes = filtered_docs

        print(f"[RETRIEVAL AGENT] Recuperados {len(documentos)} documentos para: {estado.tema}")
        return estado

    return manejar_retrieval
