# agents/retrieval_agent.py
from schemas.estado import EstadoConversacion
from langchain_core.vectorstores import VectorStoreRetriever

def crear_agente_retrieval(retriever: VectorStoreRetriever):
    def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        documentos = retriever.get_relevant_documents(estado.tema)
        estado.docs_relevantes = [doc.page_content for doc in documentos]

        print(f"[RETRIEVAL AGENT] Recuperados {len(documentos)} documentos para: {estado.tema}")
        print(f"[RETRIEVAL AGENT] Documentos: {estado.docs_relevantes}")
        return estado

    return manejar_retrieval
