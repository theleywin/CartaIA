# agents/retrieval_agent.py
from schemas.estado import EstadoConversacion
from langchain_community.vectorstores import FAISS
from utils.crawler import search_web
from utils.vector_store import update_vector_store

THRESHOLD = 1.2

def crear_agente_retrieval(vector_store: FAISS, llm):
    async def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        print("Buscando información ...")
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
            print("Buscando en la web ...")
            docs = search_web(db_query, num_results=20)
            docs_text = [doc['text'] for doc in docs if 'text' in doc]
            if len(docs_text) == 0:
                estado.docs_relevantes = []
                return estado
            update_vector_store(vector_store, docs_text)
            documentos = vector_store.similarity_search_with_score(db_query, k=10)
            filtered_docs = [doc.page_content for doc, score in documentos if score < THRESHOLD]            
            if len(filtered_docs) == 0:
                estado.docs_relevantes = []
                return estado
        
        estado.docs_relevantes = filtered_docs

        # print(f"[RETRIEVAL AGENT] Recuperados {len(documentos)} documentos para: {estado.tema}")
        return estado

    return manejar_retrieval
