import os
import re
from langchain_core.documents import Document
from experiments.chunk_size_optimization.initial_data import get_testing_chunk_sizes, db_topics
from utils.chunking import chunk_docs
from utils.document_load import load_documents
from utils.embedding_loader import embedding_loader
from langchain_community.vectorstores import FAISS
from utils.vector_store import WORST_L2_SCORE

def chunk_with_different_sizes(docs: list[Document], chunk_sizes: list[int], overlap_ratio=0.1):
    all_chunks = {}
    for size in chunk_sizes:
        chunks = chunk_docs(docs, size, ["\n\n", "\n", " ", ""], overlap_ratio)
        all_chunks[size] = chunks
    return all_chunks

def create_vector_stores(chunked_docs: dict[int, list[Document]]) -> dict[int, any]:
    embeddings = embedding_loader()

    for size, chunks in chunked_docs.items():
        print(f"[DEBUG] Creating vector store for chunk size {size}...")
        if not chunks or len(chunks) == 0:
            chunks = [Document(page_content="init")]
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        vector_store.save_local(f"./data/experiments/size_{size}_chunks")
        print(f"[DEBUG] Vector store created with {len(chunks)} chunks")


def get_query_scores(vector_store: FAISS, query: str, k: int = 10) -> list[float]:
    if not vector_store or not query:
        return [WORST_L2_SCORE] * k
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        if results and len(results) > 0:
            scores =  [float(score) for _, score in results]
            scores.extend([WORST_L2_SCORE] * (k - len(scores)))
            return scores
        else:
            return [WORST_L2_SCORE] * k
    except Exception as e:
        print(f"[ERROR] Error during similarity search: {e}")
        return [WORST_L2_SCORE] * k
    
def get_stats(sizes: list[int], queries: list[str]) -> list[dict]:
    results = []
    embeddings = embedding_loader()
    for size in sizes:
        result = { "size": size, "results": [] }
        vector_store = FAISS.load_local(
            f"./data/experiments/size_{size}_chunks",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
            )
        for query in queries:
            score = get_query_scores(vector_store, query, 30)
            result["results"].append({"query": query, "scores": score})
        results.append(result)
    return results

def get_created_sizes_from_folders(base_path: str) -> list[int]:
    """
    Busca carpetas en base_path con el patrón size_{size}_chunks y devuelve la lista de tamaños (int).
    """
    sizes = []
    if not os.path.isdir(base_path):
        return sizes
    for folder in os.listdir(base_path):
        match = re.match(r"size_(\d+)_chunks", folder)
        if match:
            sizes.append(int(match.group(1)))
    return sorted(sizes)

def run_chunking_experiment(overlap_ratio=0.1) -> dict[int, list[dict]]:
    docs = load_documents("./data/algoritmos")
    chunk_sizes = get_testing_chunk_sizes(10)
    print("[DEBUG] Starting chunking experiment...")
    chunked_docs = chunk_with_different_sizes(docs, chunk_sizes, overlap_ratio)
    create_vector_stores(chunked_docs)
    created_sizes = get_created_sizes_from_folders("./data/experiments")
    stats = get_stats(created_sizes, db_topics)
    print("[DEBUG] Experiment completed.")
    return stats
