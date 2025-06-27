import os
from graphs.tutor_workflow import crear_workflow_tutor
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.vector_store import VectorDB
from utils.input import get_initial_state
from utils.embedding_loader import embedding_loader
from utils.prettty_print import print_output
from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel

async def es_tema_valido(llm: BaseChatModel, tema: str) -> bool:
    prompt = f"""
    Dado el siguiente tema: "{tema}"

    Responde únicamente con "Sí" si el tema pertenece al dominio de estructuras de datos y algoritmos (como listas, árboles, grafos, complejidad, etc.), 
    o con "No" si no pertenece. No des ninguna explicación adicional.
    """
    respuesta = await llm.ainvoke(prompt)
    return respuesta.content.strip().lower().startswith("sí")

async def run(tutor, llm):
    user_input = input("Introduce un tema para comenzar la tutoría (presiona Enter para continuar o escribe 'q' para salir)...\n")
    if (user_input.lower() == 'q'):
        return False
    estado_inicial = await get_initial_state(user_input, llm)
    tema_usuario = estado_inicial.tema
    if not await es_tema_valido(llm, tema_usuario):
        print(f"\n⚠️ El tema \"{tema_usuario}\" no pertenece al dominio de mis conocimientos. Yo solo fui entrenado para ayudarte en temas relacionados con estructuras de datos y algoritmos, lo siento")
    else:
        estado_final = await tutor.ainvoke(estado_inicial)
        print_output(estado_final)
    return True

async def main():
    print("Cargando el tutor...")
    config = dotenv_values(".env")
    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY_1"]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    embeddings = embedding_loader()
    vector_store = VectorDB.load_existing(embeddings)
    tutor_workflow = crear_workflow_tutor(llm, vector_store)
    
    is_running = True
    while(is_running):
        is_running = await run(tutor_workflow, llm)
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())