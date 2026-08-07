import pytest
from app.services.pronunciation.phoneme_service import PhonemeService

def test_evaluate_phonemes():
    evaluated_words = [
        {
            "word": "pagi",
            "start_time": 0.0,
            "end_time": 0.4,
            "confidence": 0.9,
            "overall_score": 70,  # Below 75 should trigger mispronounced
            "pitch_score": 60,
            "energy_score": 80,
            "duration_score": 80,
            "stress_score": 80
        }
    ]
    
    phonemes, mispronounced = PhonemeService.evaluate_phonemes(evaluated_words)
    
    assert len(phonemes) == 4 # p, a, g, i
    assert phonemes[0]["phoneme"] == "p"
    assert phonemes[1]["phoneme"] == "a"
    
    assert len(mispronounced) == 1
    assert mispronounced[0]["word"] == "pagi"
    assert mispronounced[0]["score"] == 70
    assert "Intonasi atau nada pada kata ini terlalu datar" in mispronounced[0]["reason"] # Since pitch is the lowest at 60
