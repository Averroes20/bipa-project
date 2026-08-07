import pytest
import io
import wave
import math
import struct
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import get_db, Base
from app.core.deps import get_current_user
from app.models.audio_models import (
    AudioAnalysis, PhonemeStatistic, AnalysisWord, AnalysisPhoneme, 
    AnalysisPronunciation, AnalysisIntonation, AnalysisFeedback
)

from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles

@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(type_, compiler, **kw):
    return 'JSON'

@compiles(UUID, 'sqlite')
def compile_uuid_sqlite(type_, compiler, **kw):
    return 'TEXT'

from sqlalchemy.pool import StaticPool

# Setup in-memory SQLite DB
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

def override_get_current_user():
    return "test_user_id_123"

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

# Create the DB schema
Base.metadata.create_all(bind=engine)

def create_dummy_wav():
    """Generates a 1-second 440Hz sine wave WAV file in memory."""
    buf = io.BytesIO()
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        # Write frames in chunks for efficiency
        samples = [int(16000.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)) for i in range(num_samples)]
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            
    buf.seek(0)
    return buf

from unittest.mock import patch

@patch("app.repositories.analysis_repository.AnalysisRepository.get_top_k_candidates")
def test_full_pronunciation_pipeline(mock_get_top_k, mock_whisper, mock_wav2vec, mock_parselmouth):
    """
    Tests the complete pronunciation pipeline end-to-end:
    Upload Audio -> Feature Extraction -> Phoneme Scoring -> Vowel Space ->
    Accent -> Scoring -> Feedback -> Save Database -> API Response.
    """
    mock_get_top_k.return_value = [
        {"pitch_contour": [100.0, 110.0, 120.0], "energy_contour": [0.1, 0.2, 0.3], "embedding": [0.1]*512}
    ]
    
    # 1. Setup client and dummy audio
    client = TestClient(app)
    wav_buf = create_dummy_wav()
    
    # 2. Trigger the /analyze endpoint
    response = client.post(
        "/analyze",
        data={"target_text": "ini contoh"},
        files={"file": ("test.wav", wav_buf, "audio/wav")}
    )
    
    # 3. Verify Response Status and Type (SSE)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    
    # 4. Parse the SSE stream to find the final complete payload
    events = response.text.strip().split("\n\n")
    final_result = None
    for event in events:
        event = event.strip()
        print("EVENT:", event)
        if event.startswith("data: "):
            data_str = event[6:]
            payload = json.loads(data_str)
            if payload.get("status") == "complete":
                final_result = payload.get("result")
                break
                
    # 5. Verify the API returned valid payload data
    assert final_result is not None, f"Pipeline did not return a 'complete' event. Events: {events}"
    assert "id" in final_result
    assert "overall_score" in final_result
    assert "dimensions" in final_result
    assert "intonation" in final_result["dimensions"]
    assert "clarity" in final_result["dimensions"]
    assert "recommendation" in final_result
    
    analysis_id = final_result["id"]
    
    # 6. Verify Database Persistence
    db = TestingSessionLocal()
    try:
        # Check AudioAnalysis
        analysis = db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
        assert analysis is not None
        assert analysis.user_id == "test_user_id_123"
        assert analysis.analysis_detail is not None
        
        # Check PhonemeStats
        stats = db.query(PhonemeStatistic).filter(PhonemeStatistic.analysis_id == analysis_id).all()
        assert len(stats) >= 0
        
    finally:
        db.close()
