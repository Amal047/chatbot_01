from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.db import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    historyuno = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(String)
    answer = Column(String)
    enteredon = Column(DateTime)
    enteredby = Column(String)
    fetchedfrom = Column(String)
    isanswered = Column(Boolean)
