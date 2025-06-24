from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader

def load_pdfs(directory: str):
    pdf_loader = DirectoryLoader(
        path=directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
        max_concurrency=4
    )
    
    try:
        docs = pdf_loader.load()
        print(f"Documentos PDF cargados: {len(docs)}")
    except Exception as e:
        print(f"Error al cargar documentos PDF: {e}")
        return []
    return docs
    
def load_mds(directory: str):
    md_loader = DirectoryLoader(
        path=directory,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        use_multithreading=True,
        max_concurrency=4
    )
    
    try:
        docs = md_loader.load()
        print(f"Documentos Markdown cargados: {len(docs)}")
    except Exception as e:
        print(f"Error al cargar documentos Markdown: {e}")
        return []
    
    return docs

def load_documents(directory: str):
    """
    El directorio debe contener subdirectorios 'pdf' y 'md' con los archivos correspondientes.
    """
    docs = load_pdfs(f"{directory}/pdf")
    docs.extend(load_mds(f"{directory}/md"))
    return docs