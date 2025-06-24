import time
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

def analyze_chunk_stats(chunked_docs: dict[int, list[Document]]) -> dict:
    stats_data = []

    for size, chunks in chunked_docs.items():
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        stats_data.append({
            'chunk_size': size,
            'num_chunks': len(chunks),
            'avg_length': np.mean(chunk_lengths),
            'min_length': np.min(chunk_lengths),
            'max_length': np.max(chunk_lengths),
            'std_length': np.std(chunk_lengths)
        })

    return pd.DataFrame(stats_data)

def evaluate_chunk_size_performance(vector_stores: dict[int, FAISS], test_queries: list[str], k=3):
    results = []

    for query in test_queries:
        print(f"[DEBUG] Testing query: '{query[:50]}...'")

        for chunk_size, vector_store in vector_stores.items():
            start_time = time.time()
            docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
            retrieval_time = time.time() - start_time
            docs = [doc for doc, _ in docs_with_scores]
            total_chars = sum(len(doc.page_content) for doc in docs)
            avg_relevance_score = (
                sum(score for _, score in docs_with_scores) / len(docs_with_scores)
                if docs_with_scores else 0
            )

            results.append({
                'query': query,
                'chunk_size': chunk_size,
                'retrieval_time': retrieval_time,
                'total_retrieved_chars': total_chars,
                'avg_chunk_length': total_chars / len(docs) if docs else 0,
                'num_chunks_retrieved': len(docs),
                'avg_similarity_score': avg_relevance_score
            })

    return pd.DataFrame(results)

def calculate_optimization_score(performance_df: pd.DataFrame, qa_df: pd.DataFrame, weights=None):
    if weights is None:
        weights = {
            'retrieval_speed': 0.2,
            'answer_speed': 0.2,
            'similarity_score': 0.3,
            'context_efficiency': 0.3
        }

    perf_agg = performance_df.groupby('chunk_size').agg({
        'retrieval_time': 'mean',
        'avg_similarity_score': 'mean'
    }).reset_index()

    qa_agg = qa_df.groupby('chunk_size').agg({
        'answer_time': 'mean',
        'context_length': 'mean'
    }).reset_index()

    merged = perf_agg.merge(qa_agg, on='chunk_size')

    merged['retrieval_speed_norm'] = 1 - (merged['retrieval_time'] - merged['retrieval_time'].min()) / (merged['retrieval_time'].max() - merged['retrieval_time'].min())
    merged['answer_speed_norm'] = 1 - (merged['answer_time'] - merged['answer_time'].min()) / (merged['answer_time'].max() - merged['answer_time'].min())
    merged['similarity_norm'] = (merged['avg_similarity_score'] - merged['avg_similarity_score'].min()) / (merged['avg_similarity_score'].max() - merged['avg_similarity_score'].min())
    merged['context_eff_norm'] = 1 - (merged['context_length'] - merged['context_length'].min()) / (merged['context_length'].max() - merged['context_length'].min())

    merged['optimization_score'] = (
        merged['retrieval_speed_norm'] * weights['retrieval_speed'] +
        merged['answer_speed_norm'] * weights['answer_speed'] +
        merged['similarity_norm'] * weights['similarity_score'] +
        merged['context_eff_norm'] * weights['context_efficiency']
    )

    return merged.sort_values('optimization_score', ascending=False)

def evaluate_answer_quality(vector_stores: dict[int, FAISS], test_questions, llm):
    prompt_template = """
        Based on the following context, answer the question concisely and accurately:
        Context:
        {context}

        Question: {question}

        Answer:
        """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
        )

    results = []

    timeout = 10 # para no exceder el rate limit de la API :(
    for question in test_questions:
        print(f"\nEvaluating: {question}")

        for chunk_size, vector_store in vector_stores.items():
            docs = vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            start_time = time.time()
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": question})
            answer_time = time.time() - start_time
            time.sleep(timeout)

            results.append({
                'question': question,
                'chunk_size': chunk_size,
                'answer': response.content,
                'context_length': len(context),
                'answer_time': answer_time,
                'num_chunks_used': len(docs)
            })

    return pd.DataFrame(results)

def test_different_chunking_strategies(documents, chunk_size=512):
    strategies = {}

    # default
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        separators=["\n\n", "\n", " ", ""]
    )
    strategies['recursive'] = recursive_splitter.split_documents(documents)

    # basado en tokens
    token_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1)
    )
    strategies['token'] = token_splitter.split_documents(documents)

    # aproximacion de oraciones
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15),
        separators=[". ", "! ", "? ", "\n\n", "\n", " "]
    )
    strategies['sentence'] = sentence_splitter.split_documents(documents)

    return strategies

def create_performance_visualizations(performance_df: pd.DataFrame, qa_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Rendimiento por Tamaño de Chunk', fontsize=16, fontweight='bold')

    avg_times = performance_df.groupby('chunk_size')['retrieval_time'].mean()
    axes[0, 0].bar(avg_times.index, avg_times.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Tiempo Promedio de Recuperación por Tamaño de Chunk')
    axes[0, 0].set_xlabel('Tamaño de Chunk')
    axes[0, 0].set_ylabel('Tiempo (segundos)')

    avg_scores = performance_df.groupby('chunk_size')['avg_similarity_score'].mean()
    axes[0, 1].bar(avg_scores.index, avg_scores.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Puntaje Promedio de Similitud por Tamaño de Chunk')
    axes[0, 1].set_xlabel('Tamaño de Chunk')
    axes[0, 1].set_ylabel('Puntaje de Similitud')

    qa_df.boxplot(column='context_length', by='chunk_size', ax=axes[1, 0])
    axes[1, 0].set_title('Distribución de Longitud del Contexto por Tamaño de Chunk')
    axes[1, 0].set_xlabel('Tamaño de Chunk')
    axes[1, 0].set_ylabel('Longitud del Contexto (caracteres)')

    avg_answer_times = qa_df.groupby('chunk_size')['answer_time'].mean()
    axes[1, 1].bar(avg_answer_times.index, avg_answer_times.values, color='salmon', alpha=0.7)
    axes[1, 1].set_title('Tiempo Promedio de Generación de Respuesta')
    axes[1, 1].set_xlabel('Tamaño de Chunk')
    axes[1, 1].set_ylabel('Tiempo (segundos)')

    plt.tight_layout()
    plt.show()