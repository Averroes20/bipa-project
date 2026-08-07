import pytest
from app.services.pronunciation.phoneme_service import PhonemeService

def test_evaluate_phonemes():
    evaluated_words = [
        {
            "word": "buku",
            "start_time": 0.0,
            "end_time": 0.5,
            "confidence": 0.8,
            "overall_score": 85.0,
            "pitch_score": 90.0,
            "energy_score": 90.0,
            "duration_score": 90.0,
            "stress_score": 90.0
        },
        {
            "word": "ini",
            "start_time": 0.6,
            "end_time": 1.0,
            "confidence": 0.4,
            "overall_score": 60.0,
            "pitch_score": 55.0, # worst aspect
            "energy_score": 70.0,
            "duration_score": 65.0,
            "stress_score": 80.0
        }
    ]
    
    phonemes, mispronounced = PhonemeService.evaluate_phonemes(evaluated_words)
    
    assert len(phonemes) > 0 # b, u, k, u, i, n, i
    assert any(p["word_ref"] == "buku" for p in phonemes)
    
    assert len(mispronounced) == 1
    assert mispronounced[0]["word"] == "ini"
    assert "Intonasi atau nada" in mispronounced[0]["reason"]
