# main.py
import os
import uuid
from datetime import datetime
from typing import Optional


from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from app import db, utils
from app import faiss_indexer

from app.faiss_indexer import add_to_index, search_index, reset_index

from app import db, utils
from app.models import ChatHistory
from sqlalchemy.orm import Session

# App & uploads
app = FastAPI(title="Document Q&A Chatbot")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Limits & allowed
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))  # 50MB default
ALLOWED_EXTENSIONS = {ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", "csv,pptx,pdf,xlsx,docx,txt").split(",")}


def get_db():
    """FastAPI dependency to get DB session and close it properly."""
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    entered_by: str = Form("user"),
    namespace: Optional[str] = Form(None),  # optional namespace (e.g., user id or doc id)
    db_session: Session = Depends(get_db),
):
    """
    Upload a document, extract text, create chunks, embed, and store in FAISS.
    `namespace` allows isolating documents (per-user or per-project).
    """
    # Basic validation
    filename = os.path.basename(file.filename or "")
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Read and size-check
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # Create safe unique filename
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # Extract text (utils handles filetypes)
    try:
        raw_text = utils.extract_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="No readable text in uploaded file")

    # Smart chunking & normalization
    chunks = utils.smart_split_text(raw_text, max_chars=int(os.getenv("CHUNK_MAX_CHARS", 800)))
    # map to desired chunk dict with metadata
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

    # Add to vectorstore (it will embed & persist)
    try:
        added = faiss_indexer.add_to_index(chunk_dicts, namespace=namespace or entered_by)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

    # Optionally clear per-user history if requested (keep default behavior conservative)
    if os.getenv("CLEAR_HISTORY_ON_UPLOAD", "false").lower() in ("1", "true", "yes"):
        utils.clear_chat_history(user=entered_by, db_session=db_session)

    return JSONResponse({"message": f"{filename} uploaded and indexed", "added_chunks": added})


@app.post("/chat/")
async def chat(
    query: str = Form(...),
    entered_by: str = Form("user"),
    namespace: Optional[str] = Form(None),
    top_k: int = Form(5),
    db_session: Session = Depends(get_db),
):
    """
    Query endpoint. We search documents (namespace first), include recent chat context,
    ask the local LLM pipeline and fallback to Groq if needed.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    ns = namespace or entered_by

    # 1) Search vectorstore for doc context
    doc_context = faiss_indexer.search_index(query, top_k=top_k, namespace=ns)

    # 2) Fetch recent answered chats for this user (as context)
    prev_chats_q = (
        db_session.query(ChatHistory)
        .filter(ChatHistory.enteredby == entered_by, ChatHistory.isanswered == True)
        .order_by(ChatHistory.enteredon.desc())
        .limit(3)
        .all()
    )
    chat_context = [c.answer for c in prev_chats_q if c.answer]

    # 3) Ask local LLM pipeline (this function will assemble prompt & call configured model/Groq)
    response_text, used_source = utils.ask_local_llm_with_context(
        query,
        doc_context_texts=[d["text"] for d in doc_context],
        chat_context_texts=chat_context,
        top_k_docs=top_k,
        doc_threshold=float(os.getenv("DOC_THRESHOLD", 0.3)),
        chat_threshold=float(os.getenv("CHAT_THRESHOLD", 0.6)),
    )

    # 4) If empty, fallback to groq with combined context
    if not response_text:
        response_text = utils.ask_groq(query, context=list({d["text"] for d in doc_context} | set(chat_context)))
        used_source = "groq" if response_text else "none"

    # 5) Save history
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

    return {"answer": response_text or "Sorry — I couldn't find an answer."}


@app.post("/clear-history/")
async def clear_history(user: Optional[str] = Form(None), db_session: Session = Depends(get_db)):
    """
    Clear chat history. If `user` is None, clears all (admin operation).
    """
    try:
        utils.clear_chat_history(user=user, db_session=db_session)
        return {"message": f"Cleared chat history for {'all users' if not user else user}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
