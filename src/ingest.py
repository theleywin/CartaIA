import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

def run_ingestion():
    # Configuración de dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Cargar documentos PDF
    loader = DirectoryLoader(
        './data/algoritmos',
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
        max_concurrency=4
    )
    
    try:
        docs = loader.load()
        print(f"Documentos PDF cargados: {len(docs)}")
    except Exception as e:
        print(f"Error al cargar documentos PDF: {e}")
        return

    # Dividir documentos en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    try:
        documents = text_splitter.split_documents(docs)
        print(f"Fragmentos creados: {len(documents)}")
    except Exception as e:
        print(f"Error al dividir documentos: {e}")
        return

    # Configurar embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )
        print("Embeddings configurados correctamente")
    except Exception as e:
        print(f"Error al configurar embeddings: {e}")
        return

    # Crear vectorstore FAISS y guardar
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("./data/faiss_vectorstore")
        print(f"Ingesta completada. {len(documents)} fragmentos almacenados.")
    except Exception as e:
        print(f"Error al guardar vectorstore: {e}")
        raise

if __name__ == "__main__":
    run_ingestion()