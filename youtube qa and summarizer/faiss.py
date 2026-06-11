from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search


def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)


def perform_similarity_search(faiss_index, query, k=3):
    results = faiss_index.similarity_search(query, k=k)
    return results