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

# 1) File Upload API
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
    chunks = utils.extract_text(file_path)

    # Prepare chunks for FAISS (each chunk is a dict with text and embedding)
    chunk_dicts = [{"text": chunk, "embedding": utils.embed_text(chunk)} for chunk in chunks]
    
    # Add to FAISS index
    vectorstore.add_to_index(chunk_dicts)

    return {"message": f"{file.filename} uploaded & indexed"}

# 2) Chat API
@app.post("/chat/")
async def chat(query: str = Form(...), entered_by: str = Form("user")):
    # Search vectorstore for relevant context
    context = vectorstore.search_index(query)

    if context:
        response_text = utils.ask_local_llm(query, context)
        fetched_from = "vectorDB"
        is_answered = True
    else:
        response_text = utils.ask_openai(query)
        fetched_from = "openai"
        is_answered = True if response_text else False

    # Save to Postgres
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
        session.close()  # Ensure session is closed

    return {"answer": response_text}
