from schemas.contenido import ProblemaPractico
from schemas.estado import EstadoConversacion

async def generar_problema_personalizado(estado: EstadoConversacion, llm):    
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
    llm = llm.with_structured_output(ProblemaPractico)
    response = await llm.ainvoke(prompt)
    return response