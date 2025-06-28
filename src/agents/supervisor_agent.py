from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from schemas.estado import TipoAyuda, EstadoConversacion
from schemas.bdi import BDIState
import re

class SupervisorDecision(BaseModel):
    decision: TipoAyuda
    razonamiento: str

def obtener_ultima_consulta(estado: EstadoConversacion) -> str:
    for mensaje in reversed(estado.historial):
        if mensaje.get("rol") == "user":
            return mensaje.get("contenido", "")
    return "No se encontró una consulta reciente."

def crear_supervisor(llm):
    llm = llm.with_structured_output(SupervisorDecision)
    parser = JsonOutputParser(pydantic_object=SupervisorDecision)

    prompt = ChatPromptTemplate.from_template(
        """
        Eres un tutor especializado en estructuras de datos y algoritmos.
        Basado en el plan BDI actual: {bdi_plan}
        Estado del estudiante: {student_state}

        Decide el tipo de ayuda más adecuada:
        - teoria: Solicita explicación conceptual
        - ejemplo: Necesita ejemplos de código
        - practica: Quiere resolver ejercicios
        - finalizar: La consulta está resuelta

        Consulta actual: {consulta}

        {format_instructions}
        """
    )

    async def supervisor_chain(estado: EstadoConversacion):
        print("Pensando ...")
        bdi_plan = (
            estado.bdi_state.intentions.action_plan
            if estado.bdi_state and estado.bdi_state.intentions
            else "No disponible"
        )
        student_state = estado.estado_estudiante.model_dump_json()
        consulta = obtener_ultima_consulta(estado)

        formatted_prompt = prompt.format_prompt(
            bdi_plan=bdi_plan,
            student_state=student_state,
            consulta=consulta,
            format_instructions=parser.get_format_instructions(),
        )
        respuesta = await llm.ainvoke(formatted_prompt)
        estado.tipo_ayuda_necesaria = respuesta.decision
        return estado
    return supervisor_chain