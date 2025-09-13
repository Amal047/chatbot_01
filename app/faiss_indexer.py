# faiss_indexer.py
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


# --- Helper: Create FAISS index ---
def _create_index(dim: int):
    # Using cosine similarity (via normalized inner product)
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200
    return index


# --- Load existing index or initialize ---
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print("✅ Loaded FAISS index and metadata from disk.")
    except Exception:
        index = _create_index(EMBED_DIM)
        metadata_store = []  # list of dicts
else:
    index = _create_index(EMBED_DIM)
    metadata_store = []


# --- Vector utilities ---
def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_index():
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)


# --- Core API ---
def add_to_index(chunks: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
    """
    chunks: list of {"text": str, "metadata": dict}
    namespace: user/session identifier (required for isolation!)
    Returns number of added chunks.
    """
    if not chunks:
        return 0
    if not namespace:
        raise ValueError("❌ Namespace (e.g. entered_by/user_id) is required when adding to index.")

    vecs = []
    for c in chunks:
        txt = c.get("text") or ""
        meta = dict(c.get("metadata") or {})
        meta["namespace"] = namespace  # enforce namespace always
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
    Namespace is REQUIRED for user-specific isolation.
    """
    if index.ntotal == 0:
        return []
    if not namespace:
        raise ValueError("❌ Namespace (e.g. entered_by/user_id) is required when searching the index.")

    q_vec = embed_text(query_text).astype("float32").reshape(1, -1)
    q_vec = _normalize_vectors(q_vec)

    distances, indices = index.search(q_vec, top_k * 3)  # overfetch to allow filtering
    results, seen = [], set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata_store):
            continue
        entry = metadata_store[idx]
        if entry["metadata"].get("namespace") != namespace:
            continue  # only allow same user’s data
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
    Remove embeddings/metadata by namespace.
    If namespace=None → reset everything.
    """
    global index, metadata_store
    if namespace is None:
        index = _create_index(EMBED_DIM)
        metadata_store = []
        save_index()
        return True

    # rebuild without that namespace
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
    """Clear everything (all users)."""
    global index, metadata_store
    index = _create_index(EMBED_DIM)
    metadata_store = []
    save_index()
    print("✅ FAISS index and metadata fully reset.")
