from utils.problema_generador import generar_problema_personalizado
from utils.simulador_estudiante import simular_respuesta_estudiante
from schemas.estado import EstadoConversacion
from langchain_core.language_models.chat_models import BaseChatModel


def crear_agente_practica(llm: BaseChatModel):
    async def manejar_practica(estado: EstadoConversacion) -> EstadoConversacion:
        # 1. Generar un problema personalizado
        problema = await generar_problema_personalizado(estado, llm)
        estado.problema_actual = problema

        # 2. Simular respuesta del estudiante usando el nuevo módulo
        respuesta = await simular_respuesta_estudiante(
            llm=llm,
            tema=estado.tema,
            pregunta=problema.enunciado,
            contexto=estado.docs_relevantes
        )
        estado.solucion_estudiante = respuesta


        return estado

    return manejar_practica