from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def cargar_retriever():
    persist_directory = "./data/faiss_vectorstore"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(persist_directory, embedding, index_name="index", allow_dangerous_deserialization=True)
    return vectordb.as_retriever()