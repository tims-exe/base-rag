import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

def get_retriever(docs_folder = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db")

    if docs_folder:
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        all_docs = []

        for fname in os.listdir(docs_folder):
            file_path = os.path.join(docs_folder, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in {".doc", ".docx"}:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                loader = TextLoader(file_path)

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
            all_docs += splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

        db = Chroma.from_documents(
            all_docs,
            embeddings,
            persist_directory=db_dir,
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    return db.as_retriever(search_type = "similarity", search_kwargs={"k": 3})


def db_exists():
    current_dir = os.path.dirname(os.path.abspath((__file__)))
    db_dir = os.path.join(current_dir, "db", "chroma_db")

    return os.path.exists(db_dir) and os.listdir(db_dir)

