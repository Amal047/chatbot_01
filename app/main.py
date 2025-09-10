from fastapi import FastAPI, UploadFile, File, Form
from app import db, vectorstore, utils
from app.models import ChatHistory
from sqlalchemy.orm import Session
from datetime import datetime
import os

app = FastAPI()

# Ensure uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# 1) File Upload API
# -----------------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), entered_by: str = Form("user")):
    # Validate file type
    allowed_types = ["csv", "pptx", "pdf", "xlsx", "docx"]
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_types:
        return {"error": "Invalid file type"}

    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from file
    text = utils.extract_text(file_path)

    # Split text into smaller chunks (naive split: 500 chars each)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Prepare chunks for FAISS (each chunk is a dict with text and embedding)
    chunk_dicts = [
        {"text": chunk, "embedding": utils.embed_text(chunk)}
        for chunk in chunks if chunk.strip()
    ]
    
    # Add to FAISS index
    vectorstore.add_to_index(chunk_dicts)

    return {"message": f"{file.filename} uploaded & indexed"}

# -----------------------------
# 2) Chat API
# -----------------------------
@app.post("/chat/")
async def chat(query: str = Form(...), entered_by: str = Form("user")):
    # 1) Search uploaded documents
    doc_context = vectorstore.search_index(query)  # returns list of dicts with 'text' and 'embedding'

    # 2) Fetch previous chats from DB (same user, last 3 answered)
    session: Session = db.SessionLocal()
    try:
        prev_chats = (
            session.query(ChatHistory)
            .filter(ChatHistory.enteredby == entered_by, ChatHistory.isanswered == True)
            .order_by(ChatHistory.enteredon.desc())
            .limit(3)
            .all()
        )
        # Convert to dicts with 'text' and 'embedding' for ask_local_llm
        chat_context = [
            {"text": c.answer, "embedding": utils.embed_text(c.answer)}
            for c in prev_chats
        ]
    finally:
        session.close()

    # 3) Ask local LLM (documents first, then previous chats)
    response_text = utils.ask_local_llm(
        query,
        doc_context,
        chat_context,
        doc_threshold=0.3,   # Lower threshold for documents
        chat_threshold=0.6   # Higher threshold for previous chats
    )

    if response_text:
        fetched_from = "local"
        is_answered = True
    else:
        # 4) Fallback to Groq
        response_text = utils.ask_groq(query, context=doc_context + chat_context)
        fetched_from = "groq"
        is_answered = True if response_text else False

    # 5) Save chat history
    session: Session = db.SessionLocal()
    try:
        entry = ChatHistory(
            question=query,
            answer=response_text,
            enteredon=datetime.now(),
            enteredby=entered_by,
            fetchedfrom=fetched_from,
            isanswered=is_answered
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()

    return {"answer": response_text}
