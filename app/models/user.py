from sqlalchemy import Column, String, DateTime
from app.services.db import Base
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    password = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)