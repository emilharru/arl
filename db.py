import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class ProjectState(Base):
    """Tracks the current, live status of any given project run."""
    __tablename__ = 'project_state'

    project_name = Column(String, primary_key=True)
    current_cycle = Column(Integer, default=0)
    current_step = Column(String, default='Idle')
    status = Column(String, default='Stopped')
    target_status = Column(String, default='Stopped')
    pid = Column(Integer, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExecutionLog(Base):
    """Append-only log of every step execution."""
    __tablename__ = 'execution_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_name = Column(String, nullable=False)
    cycle = Column(Integer, nullable=False)
    step_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

def get_engine_and_session(db_path: str = "sqlite:///arl_status.db"):
    """Initialize the database and return a SessionLocal class."""
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal
