import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from utils.chunking import chunk_docs

DEFAULT_PATH = "./data/faiss_vectorstore"
WORST_L2_SCORE = 4.

class VectorDB:
    def __init__(self, vector_store: FAISS, path=DEFAULT_PATH):
        self.vector_store = vector_store
        self.path = path
        
    def get_docs_with_relevance(self, query: str, k: int = 10) -> List[tuple[Document, float]]:
        if not self.vector_store or not query:
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k)
            return [(doc, float(score)) for doc, score in results]
        except Exception as e:
            print(f"[ERROR] Error during similarity search: {e}")
            return []
    
    def add_docs(self, docs_str: List[str]):
        docs = [Document(page_content=doc) for doc in docs_str]   
        documents = chunk_docs(docs, 137, ["\n\n", "\n", " ", ""], overlap_ratio=0.1)
        self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.path)
        
    @staticmethod
    def load_existing(embeddings: HuggingFaceEmbeddings, path=DEFAULT_PATH) -> 'VectorDB':
        if os.path.exists(path):
            vector_store = FAISS.load_local(
                path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return VectorDB(vector_store, path)
        else:
            raise FileNotFoundError(f"Vectorstore not found at {path}. Run ingest.py first.")
        
    @staticmethod
    def create_new(embeddings: HuggingFaceEmbeddings, documents: List[Document], path=DEFAULT_PATH) -> 'VectorDB':
        try:
            if documents is None or len(documents) == 0:
                documents = [Document(page_content="init")]
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(path)
            print(f"Ingesta completada. {len(documents)} fragmentos almacenados.")
            return VectorDB(vectorstore, path)
        except Exception as e:
            print(f"Error al guardar vectorstore: {e}")
            raise