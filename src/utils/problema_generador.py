from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.contenido import ProblemaPractico
from schemas.estado import EstadoConversacion

def limpiar_json(raw_response: str) -> str:
    if raw_response.startswith("```json"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    if raw_response.startswith("```") and raw_response.endswith("```"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    return raw_response

def generar_problema_personalizado(estado: EstadoConversacion):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
    prompt = f"""
    Genera un problema de {estado.tema} para un estudiante de nivel {estado.estado_estudiante.nivel}.
    Considera:
    - Temas vistos: {', '.join(estado.estado_estudiante.temas_vistos)}
    - Errores comunes: {', '.join(estado.estado_estudiante.errores_comunes)}
    - Objetivo BDI: {estado.bdi_state.desires.primary_goal if estado.bdi_state else 'N/A'}
    
    Requisitos:
    - Dificultad apropiada
    - Enunciado claro
    - 3 casos de prueba con inputs/salidas
    - Solución óptima
    
    Formato JSON con:
    - id (string)
    - dificultad
    - enunciado
    - solucion_referencia
    - casos_prueba
    - temas_relacionados
    
    No incluyas explicaciones, solo responde con un bloque JSON puro y bien formateado.
    """
    
    respuesta_raw = llm.invoke(prompt).content
    respuesta_json = limpiar_json(respuesta_raw)
    return ProblemaPractico.model_validate_json(respuesta_json)