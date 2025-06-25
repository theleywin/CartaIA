from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_docs(docs: list[Document], size: int, separators: list[str], overlap_ratio=0.1) -> list[Document]:
    overlap = int(size * overlap_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators,
    )
    chunks = splitter.split_documents(docs)
    return chunks
