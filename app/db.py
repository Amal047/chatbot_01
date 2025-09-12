# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/chatbot_db")
# create engine with sensible defaults; allow pool sizing from env
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def init_db(create_all_if_missing: bool = False):
    """
    Import models and optionally create tables. For production, use Alembic migrations.
    """
    # import models to register metadata
    # delayed import to avoid circular imports
    try:
        from app import models  # noqa: F401
    except Exception:
        pass
    if create_all_if_missing or os.getenv("AUTO_CREATE_TABLES", "false").lower() in ("1", "true", "yes"):
        Base.metadata.create_all(bind=engine)
