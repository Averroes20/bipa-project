import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Text, DateTime, Integer, Boolean
from app.core.database import Base

class AudioAnalysis(Base):
    __tablename__ = "audio_analysis"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)

    gender_label = Column(String, nullable=True)
    pitch_mean = Column(Float, default=0.0)
    pitch_range = Column(Float, default=0.0)
    energy_mean = Column(Float, default=0.0)
    pause_ratio = Column(Float, default=0.0)
    duration = Column(Float, default=0.0)

    dtw_score_male = Column(Float, default=0.0)
    dtw_score_female = Column(Float, default=0.0)
    embedding_score_male = Column(Float, default=0.0)
    embedding_score_female = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0)

    embedding = Column(Text, nullable=True)  # JSON string or vector string
    ai_feedback = Column(Text, nullable=True)
    
    # New JSONB/Text column to store detailed dimensional scores, contours, timelines
    analysis_detail = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

class GlobalStatistics(Base):
    __tablename__ = "global_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gender_label = Column(String, nullable=False)
    pitch_mean = Column(Float, default=0.0)
    energy_mean = Column(Float, default=0.0)
    pause_ratio = Column(Float, default=0.0)
    duration = Column(Float, default=0.0)

class PhonemeStatistic(Base):
    __tablename__ = "phoneme_statistics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    analysis_id = Column(String, index=True)
    
    phoneme = Column(String, nullable=False)
    expected_phoneme = Column(String, nullable=True)
    detected_phoneme = Column(String, nullable=True)
    
    is_correct = Column(Boolean, default=True)
    confidence = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPhonemeSummary(Base):
    __tablename__ = "user_phoneme_summary"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    phoneme = Column(String, nullable=False)
    
    occurrences = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    mistake_count = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)