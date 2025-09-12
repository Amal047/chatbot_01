# vectorstore.py
import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Optional, Any

from app.utils import embed_text

# Config
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_DIM = embed_text("test").shape[0]  # infer dimension at runtime
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

# we'll use an IVFFlat/HNSW style index for scalability — for simplicity use HNSW
def _create_index(dim: int):
    # Use inner product if vectors are normalized (we will normalize to do cosine search)
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200
    return index

# In-memory structures
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print("Loaded FAISS index and metadata from disk.")
    except Exception:
        index = _create_index(EMBED_DIM)
        metadata_store = []  # list of dicts
else:
    index = _create_index(EMBED_DIM)
    metadata_store = []


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_index():
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)


def add_to_index(chunks: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
    """
    chunks: list of {"text": str, "metadata": dict}
    namespace: optional grouping key; we store namespace inside metadata
    Returns number of added chunks.
    """
    if not chunks:
        return 0
    vecs = []
    for c in chunks:
        txt = c.get("text") or ""
        meta = dict(c.get("metadata") or {})
        if namespace:
            meta["namespace"] = namespace
        # create embedding (utils.embed_text caches)
        emb = embed_text(txt)
        vecs.append(emb)
        metadata_store.append({"text": txt, "metadata": meta})
    arr = np.array(vecs, dtype="float32")
    arr = _normalize_vectors(arr)
    index.add(arr)
    save_index()
    return len(chunks)


def search_index(query_text: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search index, return list of dicts {"text":..., "metadata": {...}}.
    If namespace is provided, filter metadata results to that namespace (post-filtering).
    """
    if index.ntotal == 0:
        return []
    q_vec = embed_text(query_text).astype("float32")
    q_vec = q_vec.reshape(1, -1)
    q_vec = _normalize_vectors(q_vec)
    distances, indices = index.search(q_vec, top_k * 3)  # overfetch to allow namespace filtering
    results = []
    seen = set()
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata_store):
            continue
        entry = metadata_store[idx]
        if namespace and entry["metadata"].get("namespace") != namespace:
            continue
        txt = entry["text"]
        if txt in seen:
            continue
        seen.add(txt)
        results.append({"text": txt, "metadata": entry["metadata"]})
        if len(results) >= top_k:
            break
    return results


def clear_index(namespace: Optional[str] = None):
    """
    Remove embeddings and metadata optionally filtered by namespace.
    This is implemented as rebuilding index when deletions are needed (simple and safe).
    """
    global index, metadata_store
    if namespace is None:
        index = _create_index(EMBED_DIM)
        metadata_store = []
        save_index()
        return True
    # rebuild without the namespace entries
    new_meta = [m for m in metadata_store if m["metadata"].get("namespace") != namespace]
    vecs = [embed_text(m["text"]).astype("float32") for m in new_meta]
    index = _create_index(EMBED_DIM)
    if vecs:
        arr = np.array(vecs, dtype="float32")
        arr = _normalize_vectors(arr)
        index.add(arr)
    metadata_store = new_meta
    save_index()
    return True

def reset_index():
    """
    Clear FAISS index and stored chunks (for fresh start).
    """
    global index, stored_chunks
    index.reset()  # clear FAISS index
    stored_chunks = []
    print("✅ FAISS index and stored chunks cleared.")
