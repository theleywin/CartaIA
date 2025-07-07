from pydantic import BaseModel
from agents.bdi_agent import BDIAgent
from schemas.estado import EstadoConversacion
from langchain_core.language_models.chat_models import BaseChatModel

async def evaluar_y_actualizar_bdi(estado: EstadoConversacion, bdi_agent: BDIAgent, llm):
    solucion = estado.solucion_estudiante or ""
    comprension = await evaluar_dimension_llm("comprensión", estado.problema_actual.enunciado, solucion, llm)
    precision = await evaluar_dimension_llm("precisión",estado.problema_actual.enunciado, solucion, llm)
    profundidad = await evaluar_dimension_llm("profundidad", estado.problema_actual.enunciado, solucion, llm)

    evaluacion = {
        "comprension": comprension,
        "precision": precision,
        "profundidad": profundidad
    }
    
    estado.ultima_evaluacion = evaluacion

    await bdi_agent.update_beliefs(solucion, evaluacion)

    if not bdi_agent.evaluate_progress(evaluacion):
        await bdi_agent.handle_failure()

    estado.bdi_state = bdi_agent.state
    return estado

class EvaluacionDimension(BaseModel):
    puntuacion: float

async def evaluar_dimension_llm(dimension: str, tema: str, solucion: str, llm: BaseChatModel) -> float:
    prompt = f"""
    Eres un asistente educativo. Evalúa la siguiente solución de un estudiante en la dimensión de **{dimension}** 
    respecto a la siguiente pregunta:
    
        "{tema}" 
        
        
    Asigna una puntuación entre 0.00 (muy pobre) y 1.00 (excelente) basada en:

    - Comprensión: ¿el estudiante entendió el concepto?
    - Precisión: ¿su código o explicación tiene errores?
    - Profundidad: ¿la solución muestra razonamiento suficientemente avanzado?

    Solución del estudiante:
    {solucion}

    Devuelve únicamente un número decimal entre 0.0 y 1.0 sin texto adicional.
    """
    
    llm = llm.with_structured_output(EvaluacionDimension)
    llm_response = await llm.ainvoke(prompt)
    respuesta = llm_response.puntuacion
    return respuesta