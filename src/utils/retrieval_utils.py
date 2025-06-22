from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def update_db(db: FAISS, docs):   
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