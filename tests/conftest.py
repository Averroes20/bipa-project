import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from app.main import app
import numpy as np

@pytest.fixture
def mock_db_session():
    """Returns a mocked SQLAlchemy session."""
    session = MagicMock(spec=Session)
    return session

@pytest.fixture
def test_client():
    """Returns a FastAPI TestClient."""
    return TestClient(app)

@pytest.fixture
def dummy_audio():
    """Returns a dummy 1-second audio array (16kHz)."""
    return np.zeros(16000, dtype=np.float32)

@pytest.fixture
def mock_parselmouth(mocker):
    """Mock parselmouth.Sound and its methods to prevent actual parselmouth calls."""
    try:
        mock = mocker.patch("parselmouth.Sound")
        mock.return_value.to_pitch.return_value.selected_array = {'frequency': np.array([100.0, 110.0, 105.0])}
        mock.return_value.to_intensity.return_value.values = np.array([[50.0, 55.0, 52.0]])
        mock.return_value.to_formant_burg.return_value.get_value_at_time.return_value = 500.0
        return mock
    except ImportError:
        pass

@pytest.fixture
def mock_whisper(mocker):
    """Mock Whisper model transcribe result."""
    mock = mocker.patch("app.services.pipeline.alignment_service.get_whisper_model")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "ini contoh",
        "segments": [{"words": [{"word": "ini", "start": 0.0, "end": 0.5, "probability": 0.9},
                                {"word": "contoh", "start": 0.5, "end": 1.0, "probability": 0.9}]}]
    }
    mock.return_value = mock_model
    return mock

@pytest.fixture
def mock_wav2vec(mocker):
    """Mock Wav2Vec2."""
    mock = mocker.patch("app.services.pipeline.alignment_service.get_w2v_model")
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_proc.batch_decode.return_value = ["ini contoh"]
    mock.return_value = (mock_proc, mock_model)
    return mock
