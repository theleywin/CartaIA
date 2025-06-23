from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.document_load import load_documents
from utils.embedding_loader import embedding_loader
from utils.vector_store import init_vector_store

def run_ingestion():
    docs = load_documents("./data/algoritmos")

    # split documents
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

    embeddings = embedding_loader()
    init_vector_store(embeddings, documents)
    
if __name__ == "__main__":
    run_ingestion()