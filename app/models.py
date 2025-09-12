# models.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, func
from sqlalchemy.dialects.postgresql import UUID
import uuid
from app.db import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    historyuno = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    enteredon = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    enteredby = Column(String(128), nullable=True, index=True)  # user ID or "system"
    fetchedfrom = Column(String(64), nullable=True)  # "faiss", "db", "gpt", etc.
    isanswered = Column(Boolean, default=False)

    # Optional: group related messages into a conversation
    conversation_id = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False, index=True)

    def __repr__(self):
        short_q = (self.question[:30] + "...") if self.question and len(self.question) > 30 else self.question
        return f"<ChatHistory(historyuno={self.historyuno}, question='{short_q}', isanswered={self.isanswered}, conversation_id={self.conversation_id})>"
