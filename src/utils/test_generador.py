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
    Basado en el tema "{estado.tema}" y el ejemplo que se le ha mostrado al estudiante, genera una pregunta que evalúe su comprensión.
    Considera:
    - Nivel del estudiante: {estado.estado_estudiante.nivel}
    - Temas previos: {', '.join(estado.estado_estudiante.temas_vistos)}
    - Errores frecuentes: {', '.join(estado.estado_estudiante.errores_comunes)}
    - Objetivo pedagógico: {estado.bdi_state.desires.primary_goal if estado.bdi_state else 'N/A'}
        
    Requisitos:
    - Formato de pregunta breve (por ejemplo, ¿qué haría este código? ¿Cuál sería la salida?)
    - Enfocada en interpretar el ejemplo dado
    - Nivel adecuado

    Devuelve solo la pregunta en texto plano.
    """
    response = await llm.ainvoke(prompt)
    return {
        "tipo": "ejemplo",
        "pregunta": response.content.strip()
    }