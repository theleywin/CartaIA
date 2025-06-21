from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.estado import EstadoConversacion

async def generar_test_teoria(estado: EstadoConversacion, llm):
    prompt = f"""
    Genera una pregunta teórica corta sobre el tema "{estado.tema}" para un estudiante de nivel {estado.estado_estudiante.nivel}.
    Considera:
    - Temas vistos: {', '.join(estado.estado_estudiante.temas_vistos)}
    - Errores comunes: {', '.join(estado.estado_estudiante.errores_comunes)}
    - Objetivo BDI: {estado.bdi_state.desires.primary_goal if estado.bdi_state else 'N/A'}
    
    Requisitos:
    - Formato: pregunta abierta o verdadero/falso
    - Breve y clara
    - Nivel apropiado

    Responde solo con la pregunta, sin explicaciones ni formato adicional.
    """
    response = await llm.ainvoke(prompt)
    return {
        "tipo": "teoría",
        "pregunta": response.content.strip()
    }
    
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