from langchain_google_genai import ChatGoogleGenerativeAI
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

    def supervisor_chain(estado: EstadoConversacion):
        bdi_plan = (
            estado.bdi_state.intentions.action_plan
            if estado.bdi_state and estado.bdi_state.intentions
            else "No disponible"
        )
        student_state = estado.estado_estudiante.json()
        consulta = obtener_ultima_consulta(estado)

        formatted_prompt = prompt.format_prompt(
            bdi_plan=bdi_plan,
            student_state=student_state,
            consulta=consulta,
            format_instructions=parser.get_format_instructions(),
        )

        output = llm.invoke(formatted_prompt)

        # Limpieza del contenido JSON en caso de que venga envuelto en ```json ... ```
        raw_content = output.content
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        json_string = match.group(1) if match else raw_content.strip()

        # Parsear y actualizar el estado
        decision = SupervisorDecision.parse_raw(json_string)
        estado.tipo_ayuda_necesaria = decision.decision
        return estado
    print("entre al agente supervisor")
    return supervisor_chain