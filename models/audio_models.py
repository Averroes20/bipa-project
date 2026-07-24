from sqlalchemy import Column, Float, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AudioAnalysis(Base):
    __tablename__ = "audio_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True))

    pitch_mean = Column(Float)
    pitch_range = Column(Float)
    energy_mean = Column(Float)
    pause_ratio = Column(Float)

    dtw_score_male = Column(Float)
    dtw_score_female = Column(Float)
    embedding_score_male = Column(Float)
    embedding_score_female = Column(Float)
    final_score = Column(Float)

    embedding = Column(JSONB)
    ai_feedback = Column(Text)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)