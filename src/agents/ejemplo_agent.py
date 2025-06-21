from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.contenido import EjemploCodigo
from schemas.estado import EstadoConversacion

def crear_agente_ejemplos(llm):
    llm = llm.with_structured_output(EjemploCodigo)
    prompt = """
    {context}
    
    Eres un experto en programación y estructuras de datos.

    Genera un ejemplo de código para ilustrar el tema: "{tema}" considerando:
    - Nivel del estudiante: {nivel}
    - Lenguaje preferido: {lenguaje}
    - Incluir comentarios explicativos
    - Mostrar errores comunes y cómo evitarlos

    Devuelve la respuesta en formato JSON **válido**, con la siguiente estructura:
    {{
    "problema": "Descripción del problema",
    "lenguaje": "{lenguaje}",
    "codigo": "Bloque de código completo con comentarios",
    "explicacion": "Explicación de las decisiones clave en el código",
    "variantes": ["Variante 1", "Variante 2"]
    }}
    """

    async def generar_ejemplo(estado: EstadoConversacion):
        context = f"Contexto: \n {"\n\n".join(estado.docs_relevantes) or []}"
        respuesta = await llm.ainvoke(prompt.format(
            context=context,
            tema=estado.tema,
            nivel=estado.estado_estudiante.nivel,
            lenguaje="python"
        ))
        return respuesta
    return generar_ejemplo