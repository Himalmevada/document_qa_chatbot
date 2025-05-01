from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader
import os

def load_document(file_obj):
    print(file_obj)
    ext = os.path.splitext(file_obj.name)[1]
    if ext == ".pdf":
        return PyPDFLoader(file_path=file_obj.path).load()
    if ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path=file_obj.path).load()
    if ext == ".txt":
        return TextLoader(file_path=file_obj.path).load()
    return None


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def add_metadata(document_chunks):
    pass