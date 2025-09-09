from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import openai
import os

def extract_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Load your model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str):
    """Convert text to vector embedding."""
    return model.encode([text])[0]

openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is set in env

def ask_openai(query: str) -> str:
    """Send query to OpenAI GPT model and return response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print("OpenAI API error:", e)
        return "Sorry, I could not fetch an answer."
