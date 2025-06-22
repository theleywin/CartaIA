# agents/retrieval_agent.py
from schemas.estado import EstadoConversacion
from langchain_community.vectorstores import FAISS
from src.utils.crawler import Crawler
from utils.retrieval_utils import update_db

THRESHOLD = 1.2

def crear_agente_retrieval(db: FAISS):
    crawler = Crawler()
    def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        documentos = db.similarity_search_with_score(estado.tema, k=10)
        filtered_docs = [doc.page_content for doc, score in documentos if score < THRESHOLD]
        
        if len(filtered_docs) < 4:
            print(f"[RETRIEVAL AGENT] Buscando nuevos documentos para: {estado.tema}")
            crawl_response = crawler.crawl(estado.tema, num_results=10)
            urls = [result.get("url", "") for result in crawl_response.data]
            docs = [crawler.scrape(url) for url in urls if url]
            docs_text= [doc.markdown[:1500] for doc in docs if doc]
            update_db(db, docs_text)
            documentos = db.similarity_search_with_score(estado.tema, k=10)
            filtered_docs = [doc.page_content for doc, score in documentos if score < THRESHOLD]            
            if len(filtered_docs) == 0:
                print(f"[RETRIEVAL AGENT] No se encontraron suficientes documentos relevantes para: {estado.tema}")
                estado.docs_relevantes = []
                return estado
        
        estado.docs_relevantes = filtered_docs

        print(f"[RETRIEVAL AGENT] Recuperados {len(documentos)} documentos para: {estado.tema}")
        return estado

    return manejar_retrieval
