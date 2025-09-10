from sqlalchemy import Column, Integer, String, DateTime, Boolean, func
from app.db import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    historyuno = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=True)
    enteredon = Column(DateTime(timezone=True), server_default=func.now())  # auto timestamp
    enteredby = Column(String, nullable=True)  # e.g., user ID or "system"
    fetchedfrom = Column(String, nullable=True)  # "faiss", "db", "gpt", etc.
    isanswered = Column(Boolean, default=False)

    def __repr__(self):
        return f"<ChatHistory(historyuno={self.historyuno}, question='{self.question[:30]}...', isanswered={self.isanswered})>"
