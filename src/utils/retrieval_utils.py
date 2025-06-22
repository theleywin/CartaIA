from typing import List
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def update_db(db: FAISS, docs_str: List[str]):
    docs = [Document(page_content=doc) for doc in docs_str]   
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
    db.add_documents(documents)