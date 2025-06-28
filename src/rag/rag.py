from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.crawler import search_web
from utils.thresholds import is_relevant_l2
from rag.vector_store import VectorDB

class Rag:
    def __init__(self, vector_db: VectorDB, llm: ChatGoogleGenerativeAI):
        self.vector_db = vector_db
        self.llm = llm

    def get_relevant_docs(self, query: str) -> list[str]:
        documents: list[tuple[Document, float]] = self.vector_db.get_docs_with_relevance(query, k=10)
        filtered_docs = [doc.page_content for doc, score in documents if is_relevant_l2(score)]
        return filtered_docs    
    
    async def get_context(self, query: str) -> list[str]:
        print("Buscando información ...")
        prompt = f"""
            Traduce al inglés el siguiente texto: "{query}". 
            Devuelve solo el texto traducido, sin comillas ni explicaciones adicionales, resume el contenido traducido si excede las 40 palabras.
            No incluyas ningún otro contenido o comentario.
        """
        result = await self.llm.ainvoke(prompt)
        db_query = result.content.strip()
        filtered_docs = self.get_relevant_docs(db_query)
        
        if len(filtered_docs) == 0:
            print("Buscando en la web ...")
            docs = search_web(db_query, num_results=20)
            docs_text = [doc['text'] for doc in docs if 'text' in doc]
            if len(docs_text) == 0:
                return []
            self.vector_db.add_docs(docs_text)
            filtered_docs = self.get_relevant_docs(db_query)
        return filtered_docs