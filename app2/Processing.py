from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from datetime import datetime, timezone
import faiss


def split_documents(docs, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)



def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



def create_vectorstore(chunks, embedding_model=None, index_name=None):

    if not chunks:
        raise ValueError("Chunks list is empty!")

    texts = [c.page_content for c in chunks if c.page_content.strip()]

    if not texts:
        raise ValueError("All chunks are empty!")

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    if index_name:
        vectorstore.save_local(index_name)

    return vectorstore