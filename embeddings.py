# embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(texts):
    # Generate embeddings using a multilingual model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_texts(texts, embeddings)

def save_vector_store(store, path):
    store.save_local(path)

def load_vector_store(path):
    return FAISS.load_local(path, HuggingFaceEmbeddings())