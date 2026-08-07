import pytest
from unittest.mock import patch
from app.services.pipeline.scoring_service import PronunciationService, ScoringService

def test_detect_errors():
    target_text = "halo dunia"
    transcribed_words = [{"word": "halo", "confidence": 0.9}, {"word": "salah", "confidence": 0.8}]
    phonemes_data = [{"symbol": "h", "confidence": 0.9}, {"symbol": "a", "confidence": 0.9}, {"symbol": "l", "confidence": 0.9}, {"symbol": "o", "confidence": 0.9}, {"symbol": "s", "confidence": 0.8}, {"symbol": "a", "confidence": 0.8}]
    
    result = PronunciationService.detect_errors(target_text, transcribed_words, phonemes_data)
    
    assert result["pronunciation_score"] > 0
    assert result["word_score"] == 85.0
    assert len(result["word_errors"]) == 1

def test_calculate_scores():
    words_data = [{"confidence": 0.9}, {"confidence": 0.8}]
    phonemes_data = [{"confidence": 0.85}, {"confidence": 0.95}]
    native_sim = {"similarity": 80.0, "reference_gender": "male", "best_male_candidate": {"pitch_contour": [100, 110], "energy_contour": [0.1, 0.2]}}
    features = {"wpm": 120, "pause_ratio": 0.1, "clarity_score": 85.0, "energy_contour": [0.1, 0.2]}
    pitch_stats = {"variance": 20.0, "contour": [100, 110]}
    
    with patch("app.audio_engine.comparison.dtw_distance", return_value=0.0):
        result = ScoringService.calculate_scores(words_data, phonemes_data, native_sim, features, pitch_stats)
        
        assert "overallScore" in result
        assert "pronunciation" in result
        assert "fluency" in result
        assert "intonation" in result
        assert "clarity" in result
        assert result["dtw_score"] == 100.0
