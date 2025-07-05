from schemas.estado import EstadoConversacion
from langchain_core.language_models.chat_models import BaseChatModel

from utils.document_load import load_topics

def crear_agente_planificacion(llm: BaseChatModel):
    async def manejar_planificacion(estado: EstadoConversacion) -> EstadoConversacion:
        print("Planificando el estudio ...")
        return estado

    return manejar_planificacion