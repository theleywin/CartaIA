import re
from schemas.estado import EstadoConversacion, EstadoConversacionResponse
from langchain_core.language_models.chat_models import BaseChatModel
from utils.document_load import load_topics

async def get_initial_state(input, llm: BaseChatModel) -> EstadoConversacion:
    topics = load_topics()
    prompt = f"""
    Eres un tutor de estructuras de datos y algoritmos.
    Tu tarea es ayudar a los estudiantes a comprender temas relacionados con estructuras de datos y algoritmos
    (como listas, árboles, grafos, complejidad, etc.) a través de una conversación interactiva.
    Dado el siguiente tema: "{input}",
    crea un estado inicial de conversación que incluya el tema y un estado del estudiante (principiante, intermedio, avanzado),
    con temas vistos y errores comunes.
    Responde únicamente con el estado inicial en el siguiente formato
    {{
        "tema": string,
        "nivel": string,
        "temas_vistos": List[string],
        "errores_comunes": List[string]
    }}
    De ser posible selecciona el tema y los temas vistos de la siguiente lista:
    {", ".join(topics)}
    En caso de no existir el tema en la lista, puedes escribir el tema en un formato similar al de la lista.
    En caso de no existir un tema en la lista visto utiliza un tema cercano a los de la lista.
    Los errores comunes deben ser temas de la lista en los que el estudiante suele tener dificultades.
    No incluyas ningún otro texto o explicación, solo el JSON.
    """
    
    llm = llm.with_structured_output(EstadoConversacionResponse)
    response = await llm.ainvoke(prompt)
    estado_inicial = EstadoConversacion(
        tema=response.tema,
        estado_estudiante={
            "nivel": response.nivel,
            "temas_vistos": response.temas_vistos,
            "errores_comunes": response.errores_comunes
        }
    )
    return estado_inicial
    
def extract_json_block(text: str) -> str:
    """
    Extrae el bloque JSON de una respuesta tipo ```json ... ``` o lo devuelve tal cual si ya es JSON plano.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()