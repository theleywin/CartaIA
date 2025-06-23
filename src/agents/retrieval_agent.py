# agents/retrieval_agent.py
from schemas.estado import EstadoConversacion
from langchain_community.vectorstores import FAISS
from src.utils.crawler import Crawler
from utils.vector_store import update_vector_store

THRESHOLD = 1.2

def crear_agente_retrieval(vector_store: FAISS, llm):
    crawler = Crawler()
    async def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        prompt = f"""
            Traduce al inglés el siguiente texto: "{estado.tema}". 
            Devuelve solo el texto traducido, sin comillas ni explicaciones adicionales, resume el contenido traducido si excede las 40 palabras.
            No incluyas ningún otro contenido o comentario.
        """
        result = await llm.ainvoke(prompt)
        db_query = result.content.strip()
        documentos = vector_store.similarity_search_with_score(db_query, k=10)
        filtered_docs = [doc.page_content for doc, score in documentos if score < THRESHOLD]
        
        if len(filtered_docs) < 3:
            print(f"[RETRIEVAL AGENT] Buscando nuevos documentos para: {estado.tema}")
            crawl_response = crawler.crawl(estado.tema, num_results=10)
            urls = [result.get("url", "") for result in crawl_response]
            docs = [crawler.scrape(url) for url in urls if url]
            docs_text= [doc.markdown[:1500] for doc in docs if doc]
            update_vector_store(vector_store, docs_text)
            documentos = vector_store.similarity_search_with_score(estado.tema, k=10)
            filtered_docs = [doc.page_content for doc, score in documentos if score < THRESHOLD]            
            if len(filtered_docs) == 0:
                print(f"[RETRIEVAL AGENT] No se encontraron suficientes documentos relevantes para: {estado.tema}")
                estado.docs_relevantes = []
                return estado
        
        estado.docs_relevantes = filtered_docs

        print(f"[RETRIEVAL AGENT] Recuperados {len(documentos)} documentos para: {estado.tema}")
        return estado

    return manejar_retrieval
