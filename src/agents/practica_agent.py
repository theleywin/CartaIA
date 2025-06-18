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
        estado.solucion_estudiante = respuesta
        
        print(problema.enunciado)

        # 3. Guardar en material visible
        estado.problema_actual = {
            "enunciado": problema.enunciado,
            "respuesta_estudiante": respuesta
        }

        return estado
    print("estoy en Agente de práctica")
    return manejar_practica