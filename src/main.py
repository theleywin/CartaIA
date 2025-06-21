import os
from graphs.tutor_workflow import crear_workflow_tutor
from schemas.estado import EstadoConversacion, EstadoEstudiante, TipoAyuda
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from utils.load_vector_store import load_vector_store
from utils.prettty_print import print_output
from dotenv import dotenv_values

async def main():
    config = dotenv_values(".env")
    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY_1"]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": 32,
            "convert_to_numpy": False,
            "normalize_embeddings": True
        }
    )

    vector_store = load_vector_store(device, embeddings)
    tutor_workflow = crear_workflow_tutor(llm, vector_store)
    
    estado_inicial = EstadoConversacion(
        tema="Arbol binario de búsqueda practica",
        estado_estudiante=EstadoEstudiante(
            nivel="intermedio",
            temas_vistos=["Arbol binario de busqueda", "disjoint sets", "heap binarios"],
            errores_comunes=["grafos", "manejo de punteros"]
        )
    )

    estado_final = await tutor_workflow.ainvoke(estado_inicial)
    print_output(estado_final)
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())