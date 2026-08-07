import pytest
from app.services.pronunciation.accent_service import AccentService

def test_analyze_empty():
    result = AccentService.analyze([])
    assert result["rhythm_score"] == 0.0
    assert result["accent_score"] == 0.0

def test_analyze_with_data():
    words_data = [
        {"duration_score": 80.0, "stress_score": 90.0},
        {"duration_score": 100.0, "stress_score": 70.0}
    ]
    result = AccentService.analyze(words_data)
    assert result["rhythm_score"] == 90.0
    assert result["accent_score"] == 80.0
