
import os
from graphs.tutor_workflow import crear_workflow_tutor
from schemas.estado import EstadoConversacion, EstadoEstudiante, TipoAyuda
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


# Configurar API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyC811OsHFVXC_olcgWrZVHQaCgPaF4s7Uo"

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
# Ejecutar tutor
estado_final = tutor_workflow.invoke(estado_inicial)

# Mostrar resultado según el tipo de ayuda proporcionada
print("\n=== RESULTADO DEL TUTOR ===")
tipo = estado_final["tipo_ayuda_necesaria"]
if tipo == TipoAyuda.TEORIA and estado_final["material"]:
    print("🔍 Explicación teórica:")
    print(estado_final["material"].get("definicion", "[Definición no disponible]"))
elif tipo == TipoAyuda.EJEMPLO and estado_final["material"]:
    print("💡 Ejemplo de código:")
    print(estado_final["material"].get("codigo", "[Código no disponible]"))
elif tipo == TipoAyuda.PRACTICA and estado_final["problema_actual"]:
    print("📝 Problema para resolver:")
    print(estado_final["problema_actual"].get("enunciado", "[Enunciado no disponible]"))
elif tipo == TipoAyuda.FINALIZAR:
    print("✅ Progreso suficiente alcanzado. Sesión finalizada.")
else:
    print("⚠️ Tipo de ayuda no reconocido o información incompleta.")
    print(f"Tipo de ayuda recibido: {tipo}")

# Mostrar evaluación final si está disponible
if estado_final["ultima_evaluacion"] is not None:
    print(f"[Debug] ultima evaluacion {estado_final['ultima_evaluacion']}")
    print("\n📊 Evaluación final del estudiante:")
    for criterio, valor in estado_final["ultima_evaluacion"].items():
        print(f" - {criterio}: {valor:.2f}")
else:
    print("\nℹ️ No hay evaluación final disponible.")

# Guardar estado BDI si existe
if estado_final["bdi_state"] is not None:
    try:
        with open("estado_bdi.json", "w") as f:
            f.write(estado_final["bdi_state"].json())
        print("\n💾 Estado BDI guardado en 'estado_bdi.json'")
    except Exception as e:
        print(f"\n❌ Error al guardar estado BDI: {e}")
else:
    print("\n⚠️ No hay estado BDI para guardar.")