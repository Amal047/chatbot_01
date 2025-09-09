# vectorstore.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # model output dimension

# Initialize FAISS index and storage
index = faiss.IndexFlatL2(dimension)
stored_chunks = []

def add_to_index(chunks):
    """
    chunks: list of dicts, each dict = {"text": ..., "embedding": ...}
    """
    global index, stored_chunks
    if not chunks:
        return

    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
    stored_chunks.extend(chunks)

    dim = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    print(f"Added {len(chunks)} chunks to FAISS index")


def search_index(query_text, top_k=5):
    """
    query_text: string
    returns: list of top_k most similar chunks
    """
    if len(stored_chunks) == 0:
        return []

    query_embedding = model.encode(query_text)
    query_vec = np.array([query_embedding]).astype("float32")
    D, I = index.search(query_vec, top_k)

    results = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
    return results
