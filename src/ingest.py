from utils.chunking.chunking import chunk_optimized
from utils.document_load import load_documents
from utils.embedding_loader import embedding_loader
from utils.vector_store import init_vector_store

def run_ingestion():
    docs = load_documents("./data/algoritmos")
    documents = chunk_optimized(docs)
    embeddings = embedding_loader()
    init_vector_store(embeddings, documents)
    
if __name__ == "__main__":
    run_ingestion()