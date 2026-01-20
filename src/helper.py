
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



def load_pdf_files(data):
    loader =  DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader  
    )

    documents = loader.load()
    return documents

def fetch_page_content(docs: List[Document]) -> List[Document]:
    this_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        this_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return this_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def load_embedding_model():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )
    return embedding