from langchain_community.vectorstores import FAISS

def store_vector(chunks, embeddings_model):
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    vector_store.save_local("faiss_index")
    return vector_store