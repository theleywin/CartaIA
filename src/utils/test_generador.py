from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.estado import EstadoConversacion
from schemas.contenido import ProblemaPractico

async def generar_test_teoria(estado: EstadoConversacion, llm):
    prompt = f"""
    Diseña una evaluación teórica relacionada con el tema "{estado.tema}" para un estudiante de nivel {estado.estado_estudiante.nivel}.

    Considera:
    - Temas vistos: {', '.join(estado.estado_estudiante.temas_vistos)}
    - Errores comunes: {', '.join(estado.estado_estudiante.errores_comunes)}
    - Objetivo BDI: {estado.bdi_state.desires.primary_goal if estado.bdi_state else 'N/A'}

    Requisitos:
    - La pregunta debe evaluar conceptos teóricos clave.
    - Debe ser clara, concisa y de dificultad adecuada.
    - Incluir una solución esperada razonada.
    - Agregar casos de prueba si aplica (pueden ser ejemplos de respuestas válidas).
    - Formato JSON estricto con los siguientes campos:
        {{
            "id": "string",
            "dificultad": "baja/media/alta",
            "enunciado": "enunciado claro y completo",
            "solucion_referencia": "respuesta correcta o esperada",
            "casos_prueba": [{{"input": "respuesta posible", "output": "evaluación o nota"}}],
            "temas_relacionados": ["tema1", "tema2"]
        }}

    No incluyas ningún texto fuera del bloque JSON. Responde solo con el JSON.
    """
    llm = llm.with_structured_output(ProblemaPractico)
    response = await llm.ainvoke(prompt)
    return response

async def generar_test_ejemplo(estado: EstadoConversacion, llm):
    prompt = f"""
    Diseña una pregunta de evaluación basada en un ejemplo que el estudiante ha visto sobre el tema "{estado.tema}".
    
    Considera:
    - Nivel del estudiante: {estado.estado_estudiante.nivel}
    - Temas previos: {', '.join(estado.estado_estudiante.temas_vistos)}
    - Errores frecuentes: {', '.join(estado.estado_estudiante.errores_comunes)}
    - Objetivo BDI: {estado.bdi_state.desires.primary_goal if estado.bdi_state else 'N/A'}
    
    Requisitos:
    - Enunciado claro basado en interpretación de código o comportamiento esperado.
    - Dificultad adecuada.
    - Solución de referencia precisa.
    - Casos de prueba que reflejen respuestas del estudiante y su evaluación (por ejemplo, posibles interpretaciones).
    - Lista de temas relacionados.

    Formato JSON estricto:
    {{
        "id": "string",
        "dificultad": "baja/media/alta",
        "enunciado": "enunciado del ejercicio",
        "solucion_referencia": "respuesta correcta esperada",
        "casos_prueba": [{{"input": "respuesta del estudiante", "output": "evaluación"}}],
        "temas_relacionados": ["tema1", "tema2"]
    }}

    Responde solo con el JSON. No agregues explicaciones externas ni bloques de código adicionales.
    """
    llm = llm.with_structured_output(ProblemaPractico)
    response = await llm.ainvoke(prompt)
    return response