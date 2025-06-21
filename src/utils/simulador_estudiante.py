from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

async def simular_respuesta_estudiante(
    llm: BaseChatModel,
    tema: str,
    pregunta: str,
    contexto: Optional[str] = None
) -> str:
    context_section = f"Contexto:\n{contexto}\n\n" if contexto else ""

    prompt = f"""
{context_section}
Simula la respuesta de un estudiante que intenta resolver el siguiente ejercicio o pregunta sobre el tema \"{tema}\".
El objetivo es evaluar su comprensión, precisión y profundidad. No des explicaciones, solo muestra el código o la respuesta que daría un estudiante.

Pregunta:
{pregunta}

Respuesta del estudiante:
"""

    result = await llm.ainvoke(prompt)
    return result.content.strip()
