from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.estado import EstadoConversacion
import os
import re
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY_2"]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

async def evaluar_y_actualizar_bdi(estado: EstadoConversacion, bdi_agent):
    solucion = estado.solucion_estudiante or ""
    print("[Debug] Tema actual:", repr(estado.tema))

    comprension = await evaluar_dimension_llm("comprensión", estado.tema, solucion)
    precision = await evaluar_dimension_llm("precisión", estado.tema, solucion)
    profundidad = await evaluar_dimension_llm("profundidad", estado.tema, solucion)

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

async def evaluar_dimension_llm(dimension: str, tema: str, solucion: str) -> float:
    prompt = f"""
    Eres un asistente educativo. Evalúa la siguiente solución de un estudiante en la dimensión de **{dimension}** 
    respecto al tema "{tema}". Asigna una puntuación entre 0.0 (muy pobre) y 1.0 (excelente) basada en:

    - Comprensión: ¿el estudiante entendió el concepto?
    - Precisión: ¿su código o explicación tiene errores?
    - Profundidad: ¿la solución muestra razonamiento avanzado o solo básico?

    Solución del estudiante:
    {solucion}

    Devuelve únicamente un número decimal entre 0.0 y 1.0 sin texto adicional.
    """

    print(f"[Debug] Prompt ({dimension}):\n{prompt}")

    llm_response = await llm.ainvoke(prompt)
    respuesta = llm_response.content.strip()
    print(f"[Debug] Respuesta LLM ({dimension}):", repr(respuesta))

    # Extraer número decimal con regex
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", respuesta)
    if match:
        return float(match.group(1))
    return 0.0