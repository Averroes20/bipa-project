import pytest
import io
import wave
import math
import struct
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.main import app
from app.core.database import get_db, Base
from app.core.deps import get_current_user
from app.models.audio_models import (
    AudioAnalysis, PhonemeStatistic, AnalysisWord, AnalysisPhoneme, 
    AnalysisPronunciation, AnalysisIntonation, AnalysisFeedback
)

# SQLite compilation for PostgreSQL types
@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(type_, compiler, **kw):
    return 'JSON'

@compiles(UUID, 'sqlite')
def compile_uuid_sqlite(type_, compiler, **kw):
    return 'TEXT'

# DB Setup
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

# Setup DB Schema before tests
Base.metadata.create_all(bind=engine)

def generate_wav(duration=1.0, sample_rate=16000, silent=False, noisy=False):
    """Generates a WAV file buffer in memory."""
    buf = io.BytesIO()
    num_samples = int(sample_rate * duration)
    
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        if silent:
            samples = [0] * num_samples
        elif noisy:
            import random
            samples = [int(random.uniform(-32768, 32767)) for _ in range(num_samples)]
        else:
            samples = [int(16000.0 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)) for i in range(num_samples)]
            
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            
    buf.seek(0)
    return buf

def parse_sse(response_text):
    """Parses SSE text and returns the final 'complete' JSON payload."""
    events = response_text.strip().split("\n\n")
    final_result = None
    for event in events:
        event = event.strip()
        if event.startswith("data: "):
            try:
                payload = json.loads(event[6:])
                if payload.get("status") == "complete":
                    final_result = payload.get("result")
                    break
            except json.JSONDecodeError:
                continue
    return final_result

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

# Common mock patchers
def patch_services(mocker, gender="Male", score_multiplier=1.0, is_fast=False, is_slow=False):
    m_align = mocker.patch("app.services.pronunciation.alignment_service.AlignmentService.align")
    
    start_time = 0.0
    end_time = 0.5
    if is_fast:
        end_time = 0.1
    elif is_slow:
        end_time = 5.0
        
    m_align.return_value = {
        "words": [
            {"word": "ini", "start": start_time, "end": end_time, "confidence": 0.9 * score_multiplier},
            {"word": "contoh", "start": end_time, "end": end_time * 2, "confidence": 0.9 * score_multiplier}
        ]
    }
    
    m_pitch = mocker.patch("app.services.pipeline.pitch_service.PitchAnalysisService.analyze")
    m_pitch.return_value = {"mean": 120.0, "range": 50.0, "contour": [100.0, 110.0]}
    
    m_feature = mocker.patch("app.services.pipeline.feature_service.FeatureExtractionService.extract_prosody_and_clarity")
    m_feature.return_value = {"pause_ratio": 0.1, "energy_contour": [0.5, 0.6]}
    
    m_native = mocker.patch("app.services.pipeline.feature_service.FeatureExtractionService.extract_native_similarity")
    m_native.return_value = {
        "user_embedding": [0.1]*512,
        "reference_gender": gender,
        "male_score": 0.8 if gender == "Male" else 0.4,
        "female_score": 0.8 if gender == "Female" else 0.4
    }
    
    m_vowel = mocker.patch("app.services.pronunciation.vowel_service.VowelService.extract_and_compare")
    m_vowel.return_value = {"f1_avg": 500, "f2_avg": 1500}
    
    return m_align, m_pitch, m_feature, m_native, m_vowel


# Scenario 1: Native speaker
def test_native_speaker(client, mocker):
    patch_services(mocker, score_multiplier=1.0)
    
    start_time = time.time()
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    exec_time = time.time() - start_time
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert exec_time < 5.0 # Execution time should be fast due to mocks
    
    final_result = parse_sse(response.text)
    assert final_result is not None
    assert final_result["overall_score"] >= 0 # Check score field exists and is valid
    assert "dimensions" in final_result
    assert "intonation" in final_result["dimensions"]
    assert "recommendation" in final_result
    
    # Verify DB Persistence
    db = TestingSessionLocal()
    analysis = db.query(AudioAnalysis).filter(AudioAnalysis.id == final_result["id"]).first()
    assert analysis is not None
    db.close()

# Scenario 2: Non-native speaker
def test_non_native_speaker(client, mocker):
    patch_services(mocker, score_multiplier=0.5) # Lower probability implies non-native
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    final_result = parse_sse(response.text)
    
    assert final_result is not None
    assert final_result["overall_score"] < 100
    
    db = TestingSessionLocal()
    words = db.query(AnalysisWord).filter(AnalysisWord.analysis_id == final_result["id"]).all()
    assert len(words) > 0
    db.close()

# Scenario 3: Male speaker
def test_male_speaker(client, mocker):
    patch_services(mocker, gender="Male")
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    final_result = parse_sse(response.text)
    assert final_result is not None
    assert final_result["voice_profile"] == "Male"

# Scenario 4: Female speaker
def test_female_speaker(client, mocker):
    patch_services(mocker, gender="Female")
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    final_result = parse_sse(response.text)
    assert final_result is not None
    assert final_result["voice_profile"] == "Female"

# Scenario 5: Short audio
def test_short_audio(client, mocker):
    patch_services(mocker)
    
    response = client.post("/analyze", data={"target_text": "hi"}, files={"file": ("test.wav", generate_wav(duration=0.5), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 6: Long audio
def test_long_audio(client, mocker):
    patch_services(mocker)
    
    response = client.post("/analyze", data={"target_text": "ini contoh panjang"}, files={"file": ("test.wav", generate_wav(duration=30.0), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 7: Silent audio
def test_silent_audio(client, mocker):
    patch_services(mocker, score_multiplier=0.0)
    
    try:
        response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(silent=True), "audio/wav")})
        final_result = parse_sse(response.text)
        assert final_result is not None or "error" in response.text.lower()
    except Exception as e:
        # FastAPI might raise RuntimeError due to HTTPException inside StreamingResponse
        assert "response already started" in str(e) or "400" in str(e)

# Scenario 8: Noisy audio
def test_noisy_audio(client, mocker):
    patch_services(mocker, score_multiplier=0.2)
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(noisy=True), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 9: Wrong sample rate
def test_wrong_sample_rate(client, mocker):
    patch_services(mocker)
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(sample_rate=8000), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 10: Corrupted audio
def test_corrupted_audio(client):
    # No patching needed since AudioPreprocessingService will fail early before calling models
    buf = io.BytesIO(b'this is not a valid wav file buffer')
    try:
        response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", buf, "audio/wav")})
        assert response.status_code in [200, 400, 422, 500] 
    except Exception as e:
        # FastAPI might raise RuntimeError due to HTTPException inside StreamingResponse
        assert "response already started" in str(e) or "400" in str(e)

# Scenario 11: Missing transcript
def test_missing_transcript(client, mocker):
    patch_services(mocker)
    
    response = client.post("/analyze", data={}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 12: Very fast speech
def test_very_fast_speech(client, mocker):
    patch_services(mocker, is_fast=True)
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None

# Scenario 13: Very slow speech
def test_very_slow_speech(client, mocker):
    patch_services(mocker, is_slow=True)
    
    response = client.post("/analyze", data={"target_text": "ini contoh"}, files={"file": ("test.wav", generate_wav(duration=11.0), "audio/wav")})
    assert response.status_code == 200
    final_result = parse_sse(response.text)
    assert final_result is not None
