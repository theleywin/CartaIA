from schemas.estado import EstadoConversacion, EstadoConversacionResponse


async def get_initial_state(input, llm):
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
    
    