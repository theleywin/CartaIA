from langchain_core.prompts import ChatPromptTemplate
from schemas.contenido import EjemploCodigo
from schemas.estado import EstadoConversacion
from utils.test_generador import generar_test_ejemplo
from utils.simulador_estudiante import simular_respuesta_estudiante

def crear_agente_ejemplos(llm):
    llm_ejemplo = llm.with_structured_output(EjemploCodigo)

    prompt = ChatPromptTemplate.from_template(
        """
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
    )

    async def obtener_ejemplo(estado: EstadoConversacion) -> EstadoConversacion:
        print("Formulando ejemplos ...")
        # 1. Contexto
        context = f"Contexto:\n{'\n\n'.join(estado.docs_relevantes) or ''}"

        # 2. Generar ejemplo de código
        ejemplo = await llm_ejemplo.ainvoke(prompt.format(
            context=context,
            tema=estado.tema,
            nivel=estado.estado_estudiante.nivel,
            lenguaje="python"
        ))

        estado.material_ejemplo = ejemplo  # ← estructura EjemploCodigo

        # 3. Generar pregunta de comprensión basada en el ejemplo
        problema = await generar_test_ejemplo(estado, llm)

        # 4. Simular respuesta del estudiante usando el ejemplo como contexto
        contexto_simulacion = ejemplo.problema + "\n" + ejemplo.codigo + "\n" + ejemplo.explicacion
        respuesta = await simular_respuesta_estudiante(
            llm=llm,
            tema=estado.tema,
            pregunta=problema.enunciado,
            contexto=contexto_simulacion
        )

        estado.problema_actual = problema  # ← estructura ProblemaPractico
        estado.solucion_estudiante = respuesta

        return estado

    return obtener_ejemplo