import json
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel
from rag.vector_store import VectorDB
from utils.embedding_loader import embedding_loader, llm_loader
from utils.thresholds import is_relevant_l2
from langchain_google_genai import ChatGoogleGenerativeAI
from experiments.initial_data import db_topics

class ChunkLabel(BaseModel):
    is_relevant: bool

def label_chunk_with_llm(query: str, chunk_text: str, llm: ChatGoogleGenerativeAI) -> bool:
    llm = llm.with_structured_output(ChunkLabel)

    prompt = f"""
    Pregunta: "{query}"

    ¿Es el siguiente texto relevante para responderla, en un contexto de estructuras de datos y algoritmos?

    Texto: "{chunk_text}"

    Responde solo con un JSON que contenga el campo booleano `is_relevant`.
    """
    result: ChunkLabel = llm.invoke(prompt)
    return result.is_relevant

def load_vector_db() -> VectorDB:
    embeddings = embedding_loader()
    return VectorDB.load_existing(embeddings)

def evaluate_query(
    vector_db: VectorDB,
    query: str,
    llm: ChatGoogleGenerativeAI
) -> Dict[str, Any]:
    doc_scores = vector_db.get_docs_with_relevance(query, k=10)

    chunk_scores = []
    for doc, score in doc_scores:
        chunk_text = doc.page_content
        is_relevant = label_chunk_with_llm(query, chunk_text, llm)
        chunk_scores.append({
            "text": chunk_text,
            "score": score,
            "is_relevant": is_relevant
        })

    return {
        "query": query,
        "chunk_scores": chunk_scores
    }

def run_experiments(
    queries: List[str],
) -> List[Dict[str, Any]]:
    vector_db = load_vector_db()
    llm = llm_loader()
    results = []

    for query in queries:
        print(f"[INFO] Procesando query: {query}")
        result = evaluate_query(vector_db, query, llm)
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
):
    results = run_experiments(queries)
    save_results(results, output_path)
