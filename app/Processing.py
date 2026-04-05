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



def create_vectorstore(chunks, embedding_model=None):

    if not chunks:
        raise ValueError("Chunks list is empty!")

    # Ensure the first chunk has content
    texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
    if not texts:
        raise ValueError("All chunks are empty strings!")

    # Try generating embeddings
    sample = embedding_model.embed_documents([texts[0]])
    if not sample:
        raise ValueError("Embedding model returned empty vector list!")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
    vectorstore.save_local("faiss_index_constitution")

    return vectorstore