
import os
from graphs.tutor_workflow import crear_workflow_tutor
from schemas.estado import EstadoConversacion, EstadoEstudiante, TipoAyuda
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


# Configurar API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyBUfbau3yhJPAhJAb8EPONCcAlUVUs3v3E"

# Inicializar modelos
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Configurar embeddings consistentes con la ingesta
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={
        "batch_size": 32,
        "convert_to_numpy": False,
        "normalize_embeddings": True
    }
)

# Cargar vectorstore FAISS
vector_store_path = "./data/faiss_vectorstore"
if os.path.exists(vector_store_path):
    # Cargar con configuración de dispositivo
    if device == "cuda":
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True,
            device="cuda"
        )
    else:
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
else:
    raise FileNotFoundError(f"Vectorstore no encontrado en {vector_store_path}. Ejecuta ingest.py primero.")

# Crear workflow
tutor_workflow = crear_workflow_tutor(llm, vector_store)

# Estado inicial del estudiante
estado_inicial = EstadoConversacion(
    tema="Árboles binarios de búsqueda",
    estado_estudiante=EstadoEstudiante(
        nivel="intermedio",
        temas_vistos=["listas enlazadas", "pilas", "colas"],
        errores_comunes=["recursión infinita", "manejo de punteros"]
    )
)

# Ejecutar tutor
resultado = tutor_workflow.invoke(estado_inicial)

# Mostrar resultado
if resultado.tipo_ayuda_necesaria == TipoAyuda.TEORIA:
    print("Explicación teórica:")
    print(resultado.material.definicion)
elif resultado.tipo_ayuda_necesaria == TipoAyuda.EJEMPLO:
    print("Ejemplo de código:")
    print(resultado.material.codigo)
elif resultado.tipo_ayuda_necesaria == TipoAyuda.PRACTICA:
    print("Problema para resolver:")
    print(resultado.problema_actual.enunciado)

# Guardar estado BDI para sesiones futuras
with open("estado_bdi.json", "w") as f:
    f.write(resultado.bdi_state.json())