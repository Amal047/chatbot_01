from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import os

# -----------------------------
# Environment & Groq setup
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# -----------------------------
# PDF / Text extraction
# -----------------------------
def extract_text(file_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -----------------------------
# Embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str):
    """Convert text to vector embedding."""
    return model.encode([text])[0]

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Local LLM using embeddings
# -----------------------------
def ask_local_llm(query: str, doc_context: list[dict], chat_context: list[dict],
                   doc_threshold=0.3, chat_threshold=0.6) -> str:
    query_vec = embed_text(query)

    best_doc = ("", 0)   # (text, similarity)
    best_chat = ("", 0)

    # 1) Document check
    for chunk in doc_context or []:
        chunk_vec = chunk.get("embedding")
        if chunk_vec is None:
            chunk_vec = embed_text(chunk["text"])
        sim = cosine_similarity(query_vec, chunk_vec)
        if sim > best_doc[1]:
            best_doc = (chunk["text"], sim)

    # 2) Chat check
    for chat in chat_context or []:
        chat_vec = chat.get("embedding")
        if chat_vec is None:
            chat_vec = embed_text(chat["text"])
        sim = cosine_similarity(query_vec, chat_vec)
        if sim > best_chat[1]:
            best_chat = (chat["text"], sim)

    # 3) Decide which to use
    if best_doc[1] >= doc_threshold and best_doc[1] >= best_chat[1]:
        return f"Answer from your documents:\n{best_doc[0]}"
    elif best_chat[1] >= chat_threshold:
        # strip duplicate prefix if exists
        text = best_chat[0]
        prefix = "Answer from previous chats:\n"
        if text.startswith(prefix):
            text = text[len(prefix):]
        return f"Answer from previous chats:\n{text}"

    return ""


# -----------------------------
# Fallback LLM (Groq)
# -----------------------------
def ask_groq(query: str, context: list[dict] = None) -> str:
    """Use Groq API if local embeddings don't match."""
    combined_context = "\n".join([c["text"] for c in context[:3]]) if context else ""
    prompt = f"Context: {combined_context}\n\nUser query: {query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq API error:", e)
        return "Sorry, I could not fetch an answer."
