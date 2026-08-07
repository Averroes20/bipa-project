import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base

class DatasetAudio(Base):
    __tablename__ = "dataset_audio"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    speaker_id = Column(String, nullable=True)
    filename = Column(String, nullable=False)
    gender = Column(String, nullable=True)
    language = Column(String, default="id-ID")
    transcript = Column(Text, nullable=True)
    sample_rate = Column(Integer, default=16000)
    duration = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    feature = relationship("DatasetFeature", back_populates="audio", uselist=False, cascade="all, delete-orphan")
    contour = relationship("DatasetContour", back_populates="audio", uselist=False, cascade="all, delete-orphan")
    formant = relationship("DatasetFormant", back_populates="audio", uselist=False, cascade="all, delete-orphan")


class DatasetFeature(Base):
    __tablename__ = "dataset_feature"

    audio_id = Column(String, ForeignKey("dataset_audio.id"), primary_key=True)
    
    pitch_mean = Column(Float, default=0.0)
    pitch_std = Column(Float, default=0.0)
    pitch_range = Column(Float, default=0.0)
    
    energy_mean = Column(Float, default=0.0)
    pause_ratio = Column(Float, default=0.0)
    speech_rate = Column(Float, default=0.0)
    
    mfcc = Column(JSON, nullable=True) # Usually an array of arrays
    zcr = Column(Float, default=0.0)
    spectral_centroid = Column(Float, default=0.0)
    spectral_bandwidth = Column(Float, default=0.0)
    spectral_rolloff = Column(Float, default=0.0)
    
    embedding_vector = Column(Text, nullable=True) # pgvector will cast this to vector

    audio = relationship("DatasetAudio", back_populates="feature")


class DatasetContour(Base):
    __tablename__ = "dataset_contour"

    audio_id = Column(String, ForeignKey("dataset_audio.id"), primary_key=True)
    
    # We use JSONB for Postgres optimizations as requested
    pitch_contour = Column(JSONB, nullable=True)
    energy_contour = Column(JSONB, nullable=True)
    pause_timeline = Column(JSONB, nullable=True)

    audio = relationship("DatasetAudio", back_populates="contour")


class DatasetFormant(Base):
    __tablename__ = "dataset_formant"

    audio_id = Column(String, ForeignKey("dataset_audio.id"), primary_key=True)
    
    f1 = Column(Float, nullable=True)
    f2 = Column(Float, nullable=True)
    f3 = Column(Float, nullable=True)
    vowel_profile = Column(JSONB, nullable=True)

    audio = relationship("DatasetAudio", back_populates="formant")
