# main.py
import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from app import db, utils
from app import faiss_indexer
from app.models import ChatHistory
from sqlalchemy.orm import Session

# --- App & uploads ---
app = FastAPI(title="Document Q&A Chatbot")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Limits & allowed ---
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))  # 50MB default
ALLOWED_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTENSIONS", "csv,pptx,pdf,xlsx,docx,txt").split(",")
}

# --- Simple in-memory cache ---
# Key: hash(entered_by + query), Value: dict(answer, entered_by, namespace)
query_cache = {}


def _cache_key(entered_by: str, query: str) -> str:
    """Generate a unique key for caching based on user and query."""
    raw = f"{entered_by}:{query}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# --- DB dependency ---
def get_db():
    """FastAPI dependency to get DB session and close it properly."""
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


# --- Upload endpoint ---
@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    entered_by: str = Form("user"),
    namespace: Optional[str] = Form(None),
    db_session: Session = Depends(get_db),
):
    """
    Upload a document, extract text, create chunks, embed, and store in FAISS.
    Automatically uses namespace=entered_by if not provided.
    """
    # --- Basic validation ---
    filename = os.path.basename(file.filename or "")
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # --- Read & size check ---
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # --- Save file ---
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # --- Extract text ---
    try:
        raw_text = utils.extract_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="No readable text in uploaded file")

    # --- Smart chunking & metadata ---
    chunks = utils.smart_split_text(
        raw_text, max_chars=int(os.getenv("CHUNK_MAX_CHARS", 800))
    )
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source_file": unique_name,
            "source_ext": ext,
            "chunk_index": i,
            "entered_by": entered_by,
            "namespace": namespace or entered_by,
        }
        chunk_dicts.append({"text": chunk, "metadata": meta})

    # --- Determine namespace ---
    ns = namespace or entered_by

    # --- Add to vectorstore ---
    try:
        added = faiss_indexer.add_to_index(chunk_dicts, namespace=ns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

    # --- Optional: clear per-user chat history ---
    if os.getenv("CLEAR_HISTORY_ON_UPLOAD", "false").lower() in ("1", "true", "yes"):
        utils.clear_chat_history(user=entered_by, db_session=db_session)

    # --- Invalidate cache for this user/namespace ---
    keys_to_remove = [
        k
        for k, v in query_cache.items()
        if v.get("entered_by") == entered_by or v.get("namespace") == ns
    ]
    for k in keys_to_remove:
        query_cache.pop(k, None)

    return JSONResponse(
        {"message": f"{filename} uploaded and indexed", "added_chunks": added}
    )


# --- Chat endpoint with caching ---
@app.post("/chat/")
async def chat(
    query: str = Form(...),
    entered_by: str = Form("user"),
    namespace: Optional[str] = Form(None),
    top_k: int = Form(5),
    db_session: Session = Depends(get_db),
):
    """
    Query endpoint with caching.
    Searches personal + global documents, asks LLM, and caches results.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    ns = namespace or entered_by
    cache_id = _cache_key(entered_by, query)

    # --- Check cache ---
    if cache_id in query_cache:
        return query_cache[cache_id]

    # --- Search vectorstore ---
    try:
        doc_context = faiss_indexer.search_index(query, top_k=top_k, namespace=ns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")

    # --- Fetch recent answered chats ---
    prev_chats_q = (
        db_session.query(ChatHistory)
        .filter(ChatHistory.enteredby == entered_by, ChatHistory.isanswered == True)
        .order_by(ChatHistory.enteredon.desc())
        .limit(3)
        .all()
    )
    chat_context = [c.answer for c in prev_chats_q if c.answer]

    # --- Ask local LLM pipeline ---
    response_text, used_source = utils.ask_local_llm_with_context(
        query,
        doc_context_texts=[d["text"] for d in doc_context],
        chat_context_texts=chat_context,
        top_k_docs=top_k,
        doc_threshold=float(os.getenv("DOC_THRESHOLD", 0.3)),
        chat_threshold=float(os.getenv("CHAT_THRESHOLD", 0.6)),
    )

    # --- Fallback to Groq ---
    if not response_text:
        response_text = utils.ask_groq(
            query, context=list({d["text"] for d in doc_context} | set(chat_context))
        )
        used_source = "groq" if response_text else "none"

    # --- Save history ---
    entry = ChatHistory(
        question=query,
        answer=response_text or "",
        enteredon=datetime.utcnow(),
        enteredby=entered_by,
        fetchedfrom=used_source,
        isanswered=bool(response_text),
    )
    db_session.add(entry)
    db_session.commit()

    # --- Store in cache ---
    result = {
        "answer": response_text or "Sorry — I couldn't find an answer.",
        "entered_by": entered_by,
        "namespace": ns,
    }
    query_cache[cache_id] = result

    return result


# --- Clear FAISS index ---
@app.post("/clear-index/")
async def clear_index(
    user: Optional[str] = Form(None), db_session: Session = Depends(get_db)
):
    try:
        if user:
            faiss_indexer.clear_index(namespace=user)
            utils.clear_chat_history(user=user, db_session=db_session)
            return {"message": f"Cleared FAISS index and chat history for user: {user}"}
        else:
            faiss_indexer.reset_index()
            utils.clear_chat_history(user=None, db_session=db_session)
            return {"message": "Cleared FAISS index and chat history for all users"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e}")


# --- Clear chat history ---
@app.post("/clear-history/")
async def clear_history(
    user: Optional[str] = Form(None), db_session: Session = Depends(get_db)
):
    try:
        utils.clear_chat_history(user=user, db_session=db_session)
        return {"message": f"Cleared chat history for {'all users' if not user else user}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
