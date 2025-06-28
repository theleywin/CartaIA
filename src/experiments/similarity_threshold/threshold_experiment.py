import json
import numpy as np
from typing import List, Dict, Any
from rag.vector_store import VectorDB
from utils.embedding_loader import embedding_loader
from utils.thresholds import is_relevant_l2
from experiments.initial_data import db_topics

DEFAULT_THRESHOLDS = np.round(np.arange(0.0, 4.01, 0.1), 2).tolist()

def load_vector_db() -> VectorDB:
    embeddings = embedding_loader()
    return VectorDB.load_existing(embeddings)

def evaluate_query(
    vector_db: VectorDB,
    query: str,
    thresholds: List[float]
) -> Dict[str, Any]:
    """Evalúa una query para todos los umbrales dados."""
    doc_scores = vector_db.get_docs_with_relevance(query, k=10)

    chunk_scores = [
        {"text": doc.page_content, "score": score}
        for doc, score in doc_scores
    ]

    per_threshold_docs = {
        str(t): [
            chunk["text"]
            for chunk in chunk_scores
            if is_relevant_l2(chunk["score"], threshold=t)
        ]
        for t in thresholds
    }

    return {
        "query": query,
        "chunk_scores": chunk_scores,
        "retrieved_by_threshold": per_threshold_docs
    }

def run_experiments(
    queries: List[str],
    thresholds: List[float] = DEFAULT_THRESHOLDS,
) -> List[Dict[str, Any]]:
    vector_db = load_vector_db()
    results = []

    for query in queries:
        print(f"[INFO] Procesando query: {query}")
        result = evaluate_query(vector_db, query, thresholds)
        results.append(result)

    return results

def save_results(
    results: List[Dict[str, Any]],
    filepath: str
):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[✅] Resultados guardados en {filepath}")

def run_threshold_experiment(
    queries: List[str] = db_topics,
    output_path: str = "./src/experiments/similarity_threshold/results.json",
    thresholds: List[float] = DEFAULT_THRESHOLDS,
):
    results = run_experiments(queries, thresholds)
    save_results(results, output_path)
