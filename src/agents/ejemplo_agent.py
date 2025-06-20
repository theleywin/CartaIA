from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.contenido import EjemploCodigo
from schemas.estado import EstadoConversacion

def limpiar_json(raw_response: str) -> str:
    if raw_response.startswith("```json"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    if raw_response.startswith("```") and raw_response.endswith("```"):
        lines = raw_response.splitlines()
        return "\n".join(lines[1:-1])
    return raw_response

def crear_agente_ejemplos():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
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

    def generar_ejemplo(estado: EstadoConversacion):
        context = f"Contexto: \n {"\n\n".join(estado.docs_relevantes) or []}"
        respuesta = llm.invoke(prompt.format(
            context=context,
            tema=estado.tema,
            nivel=estado.estado_estudiante.nivel,
            lenguaje="python"
        ))
        json_limpio = limpiar_json(respuesta.content)
        return EjemploCodigo.parse_raw(json_limpio)
    print("entre al agente ejemplos")
    return generar_ejemplo