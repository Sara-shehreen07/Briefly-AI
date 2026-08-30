import os
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript:str)->QdrantVectorStore:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk,metadata = {'chunk_index':i})
        for i,chunk in enumerate(chunks)
    ]
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        location=":memory:",
    )
    return vector_store

def load_vector_store()->QdrantVectorStore:
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore(
        collection_name= COLLECTION_NAME,
        embedding_function = embeddings
    )
    return vector_store

def get_retreiver(vector_store:QdrantVectorStore,k:int=4):
    return vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":k}
    )
