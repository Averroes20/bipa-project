import pytest
from tests.conftest import test_client
from app.models.audio_models import AnalysisWord, AnalysisPhoneme
from unittest.mock import patch, MagicMock

def test_get_analysis_words(test_client):
    mock_db = MagicMock()
    
    # Mocking the query
    mock_word1 = AnalysisWord(word="selamat", overall_score=90)
    mock_word2 = AnalysisWord(word="pagi", overall_score=85)
    
    mock_db.query().filter().all.return_value = [mock_word1, mock_word2]
    
    from app.core.database import get_db
    test_client.app.dependency_overrides[get_db] = lambda: mock_db
    
    response = test_client.get("/analysis/test_id/words")
    
    # Cleanup
    test_client.app.dependency_overrides = {}
    
    assert response.status_code == 200
    data = response.json()
    assert "words" in data
    assert len(data["words"]) == 2
    assert data["words"][0]["word"] == "selamat"

def test_get_analysis_phonemes(test_client):
    mock_db = MagicMock()
    mock_ph = AnalysisPhoneme(phoneme="a", pronunciation_score=80)
    mock_db.query().filter().all.return_value = [mock_ph]
    
    from app.core.database import get_db
    test_client.app.dependency_overrides[get_db] = lambda: mock_db
    
    response = test_client.get("/analysis/test_id/phonemes")
    
    test_client.app.dependency_overrides = {}
    
    assert response.status_code == 200
    data = response.json()
    assert "phonemes" in data
    assert len(data["phonemes"]) == 1
    assert data["phonemes"][0]["phoneme"] == "a"
