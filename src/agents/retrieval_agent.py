from langchain_google_genai import ChatGoogleGenerativeAI
from rag.rag import Rag
from rag.vector_store import VectorDB
from schemas.estado import EstadoConversacion

def crear_agente_retrieval(vector_store: VectorDB, llm: ChatGoogleGenerativeAI):
    rag = Rag(vector_store, llm)
    async def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        estado.docs_relevantes = await rag.get_context(estado.tema)
        return estado

    return manejar_retrieval
