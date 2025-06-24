import os
from random import sample
from dotenv import dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.embedding_loader import embedding_loader
from utils.chunking.globals import test_queries, test_chunk_sizes
from utils.chunking.chunk_optimization import calculate_optimization_score, evaluate_answer_quality, evaluate_chunk_size_performance
from langchain_community.vectorstores import FAISS

def chunk_docs(docs: list[Document], size: int, separators: list[str], overlap_ratio=0.1) -> list[Document]:
    overlap = int(size * overlap_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators,
    )
    chunks = splitter.split_documents(docs)
    return chunks

def chunk_with_different_sizes(docs: list[Document], chunk_sizes: list[int], overlap_ratio=0.1):
    all_chunks = {}
    for size in chunk_sizes:
        chunks = chunk_docs(docs, size, ["\n\n", "\n", " ", ""], overlap_ratio)
        all_chunks[size] = chunks
    return all_chunks

def create_vector_stores(chunked_docs: dict[int, list[Document]]) -> dict[int, any]:
    embeddings = embedding_loader()
    vector_stores = {}

    for size, chunks in chunked_docs.items():
        print(f"[DEBUG] Creating vector store for chunk size {size}...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        vector_stores[size] = vector_store
        print(f"[DEBUG] Vector store created with {len(chunks)} chunks")

    return vector_stores

def chunk_optimized(docs: list[Document], computed_chunk_size=None) -> list[Document]:
    if computed_chunk_size is not None:
        print("[DEBUG] Using precomputed chunk size:", computed_chunk_size)
        return chunk_docs(docs, computed_chunk_size, ["\n\n", "\n", " ", ""], overlap_ratio=0.1)
    
    config = dotenv_values(".env")
    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY_1"]
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    queries = sample(test_queries, 3)
    
    chunked_docs = chunk_with_different_sizes(docs, test_chunk_sizes, overlap_ratio=0.1)
    vector_stores = create_vector_stores(chunked_docs)
    performance_df = evaluate_chunk_size_performance(vector_stores, queries)
    # qa_results = evaluate_answer_quality(vector_stores, queries, llm)
    optimization_score = calculate_optimization_score(performance_df, None, None)
    best_chunk_size = int(optimization_score.iloc[0]['chunk_size'])
    print("[DEBUG] Best chunk size determined:", best_chunk_size)
    return chunked_docs[best_chunk_size]