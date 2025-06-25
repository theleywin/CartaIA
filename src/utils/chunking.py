from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.embedding_loader import embedding_loader
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
        if not chunks or len(chunks) == 0:
            chunks = [Document(page_content="init")]  # Ensure at least one chunk exists
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        vector_stores[size] = vector_store
        print(f"[DEBUG] Vector store created with {len(chunks)} chunks")

    return vector_stores