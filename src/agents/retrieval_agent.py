from rag.rag import Rag
from rag.vector_store import VectorDB
from schemas.estado import EstadoConversacion
from langchain_core.language_models.chat_models import BaseChatModel

def crear_agente_retrieval(vector_store: VectorDB, llm: BaseChatModel):
    rag = Rag(vector_store, llm)
    async def manejar_retrieval(estado: EstadoConversacion) -> EstadoConversacion:
        estado.docs_relevantes = await rag.get_context(estado.tema)
        return estado

    return manejar_retrieval
