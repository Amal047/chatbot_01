import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Embedding Model & FAISS Setup
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # embedding size for this model

# FAISS index and storage
index = faiss.IndexFlatL2(dimension)
stored_chunks = []  # list of dicts: {"text": str, "embedding": np.array}


# -----------------------------
# Add chunks to index
# -----------------------------
def add_to_index(chunks: list[dict]):
    """
    Add chunks of text + embeddings to FAISS index.
    Each chunk must have 'text' and 'embedding'.
    """
    global index, stored_chunks
    if not chunks:
        return

    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    stored_chunks.extend(chunks)
    index.add(embeddings)
    print(f"✅ Added {len(chunks)} chunks to FAISS index. Total stored: {len(stored_chunks)}")


# -----------------------------
# Search index
# -----------------------------
def search_index(query_text: str, top_k=5) -> list[dict]:
    """
    Search FAISS index for most similar chunks to query.
    Returns top_k chunks (dicts with 'text' and 'embedding').
    """
    if len(stored_chunks) == 0:
        return []

    query_vec = model.encode([query_text]).astype("float32")  # shape: (1, dim)
    D, I = index.search(query_vec, top_k)
    results = [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]
    return results