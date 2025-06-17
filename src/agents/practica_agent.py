from utils.problema_generador import generar_problema_personalizado
from schemas.estado import EstadoConversacion
from langchain_core.language_models.chat_models import BaseChatModel

def crear_agente_practica(llm: BaseChatModel):
    def manejar_practica(estado: EstadoConversacion) -> EstadoConversacion:
        # 1. Generar un problema personalizado
        problema = generar_problema_personalizado(estado)
        estado.problema_actual = problema.model_dump()

        # 2. Construir un prompt para que el LLM simule ser el estudiante
        prompt = f"""
        Simula la respuesta de un estudiante que intenta resolver el siguiente ejercicio de programación sobre "{estado.problema_actual}". 
        El objetivo es evaluar su comprensión, precisión y profundidad. No des explicaciones, solo muestra el código del estudiante.

        Ejercicio:
        {problema.enunciado}

        Respuesta del estudiante:
        """
        respuesta = llm.invoke(prompt).content.strip()
        print(f"[DEBUG IMPORTANT] Respuesta del estudiante: {respuesta}")
        estado.solucion_estudiante = respuesta

        # 3. Guardar en material visible
        estado.material = {
            "problema": problema.enunciado,
            "respuesta_estudiante": respuesta
        }

        return estado

    return manejar_practica