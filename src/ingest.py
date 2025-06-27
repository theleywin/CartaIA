from rag.vector_store import VectorDB
from utils.chunking import chunk_docs
from utils.document_load import load_documents
from utils.embedding_loader import embedding_loader

def run_ingestion():
    docs = load_documents("./data/algoritmos")
    # aqui usamos un chunk size optimizado por nuestros experimentos
    documents = chunk_docs(docs, 137, ["\n\n", "\n", " ", ""], overlap_ratio=0.1)
    embeddings = embedding_loader()
    VectorDB.create_new(embeddings, documents)
    
if __name__ == "__main__":
    run_ingestion()