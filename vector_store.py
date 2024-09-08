import os
from langchain_community.vectorstores import FAISS

def store_vector(chunks, embeddings_model, index_name="faiss_index"):
    """
    Create or load a FAISS vector store from text chunks.
    Args:
        chunks (list): List of text chunks.
        embeddings_model: Embedding model to convert text to vectors.
        index_name (str): Name of the directory to store the FAISS index.
    Returns:
        FAISS: Vector store containing text chunk embeddings.
    """
    if os.path.exists(index_name):
        vector_store = FAISS.load_local(index_name, embeddings_model)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings_model)
        vector_store.save_local(index_name)
    return vector_store