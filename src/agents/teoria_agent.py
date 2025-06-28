from langchain_core.prompts import ChatPromptTemplate
from schemas.contenido import ExplicacionTeorica
from schemas.estado import EstadoConversacion
from utils.test_generador import generar_test_teoria
from utils.simulador_estudiante import simular_respuesta_estudiante

def crear_agente_teoria(llm):
    llm_teoria = llm.with_structured_output(ExplicacionTeorica)

    prompt = ChatPromptTemplate.from_template(
        """
        {context}
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

    async def obtener_teoria(estado: EstadoConversacion) -> EstadoConversacion:
        print("Buscando teoría ...")
        # 1. Paso BDI actual (opcional)
        bdi_step = ""
        if estado.bdi_state and estado.bdi_state.intentions.action_plan:
            current_step = estado.bdi_state.intentions.current_step
            if current_step < len(estado.bdi_state.intentions.action_plan):
                bdi_step = estado.bdi_state.intentions.action_plan[current_step]

        # 2. Construir contexto y generar material teórico
        context = f"Contexto:\n{'\n\n'.join(estado.docs_relevantes) or ''}"
        explicacion = await llm_teoria.ainvoke(prompt.format(
            context=context,
            tema=estado.tema,
            nivel=estado.estado_estudiante.nivel,
            bdi_step=bdi_step,
            misconceptions=", ".join(estado.estado_estudiante.errores_comunes),
            preferences=", ".join(estado.bdi_state.beliefs.learning_preferences) if estado.bdi_state else ""
        ))

        estado.material = explicacion  # ← estructura ExplicacionTeorica

        # 3. Generar una pregunta teórica
        problema = await generar_test_teoria(estado, llm)

        # 4. Simular respuesta del estudiante usando la explicación como contexto
        contexto_simulacion = explicacion.concepto + "\n" +  explicacion.definicion  + "\n" + \
            "\n".join(explicacion.caracteristicas) + "\n" + \
            "\n".join(explicacion.casos_uso) + "\n" + \
             explicacion.analogia + "\n" + explicacion.complejidad
             
        respuesta = await simular_respuesta_estudiante(
            llm=llm,
            tema=estado.tema,
            pregunta=problema.enunciado,
            contexto=contexto_simulacion
        )
        estado.solucion_estudiante = respuesta
        estado.problema_actual = problema 

        return estado

    return obtener_teoria