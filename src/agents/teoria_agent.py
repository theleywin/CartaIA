from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas.contenido import ExplicacionTeorica
from schemas.estado import EstadoConversacion

def limpiar_json(raw_response: str) -> str:
    if raw_response.startswith("```json"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    if raw_response.startswith("```") and raw_response.endswith("```"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    return raw_response

def crear_agente_teoria(llm):
    prompt = ChatPromptTemplate.from_template(
        """
        Eres un experto en estructuras de datos. Proporciona una explicación clara y concisa.

        Contexto BDI:
        - Paso actual del plan: {bdi_step}
        - Conceptos erróneos: {misconceptions}
        - Preferencias: {preferences}

        Tema: {tema}
        Nivel: {nivel}

         Estructura requerida (en formato JSON válido):
    {{
      "concepto": "Definición breve del concepto",
      "definicion": "Explicación detallada",
      "caracteristicas": ["característica 1", "característica 2", "..."],
      "complejidad": {{"mejor_caso": "O(1)", "peor_caso": "O(n)"}},
      "casos_uso": ["caso 1", "caso 2", "..."],
      "analogia": "Analogía sencilla para entender el concepto"
    }}

        """
    )

    def obtener_teoria(estado: EstadoConversacion):
        bdi_step = ""
        if estado.bdi_state and estado.bdi_state.intentions.action_plan:
            current_step = estado.bdi_state.intentions.current_step
            if current_step < len(estado.bdi_state.intentions.action_plan):
                bdi_step = estado.bdi_state.intentions.action_plan[current_step]

        response = llm.invoke(prompt.format(
            tema=estado.tema,
            nivel=estado.estado_estudiante.nivel,
            bdi_step=bdi_step,
            misconceptions=", ".join(estado.estado_estudiante.errores_comunes),
            preferences=", ".join(estado.bdi_state.beliefs.learning_preferences) if estado.bdi_state else ""
        ))

        contenido_limpio = limpiar_json(response.content)
        explicacion = ExplicacionTeorica.parse_raw(contenido_limpio)
        return {"material": explicacion.dict()}
    print("entre al agente teoria")
    return obtener_teoria