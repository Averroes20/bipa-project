import pytest
from app.services.pronunciation.word_service import WordService

def test_evaluate_words_valid_input():
    words = [
        {"word": "selamat", "start": 0.0, "end": 0.5, "confidence": 0.95},
        {"word": "pagi", "start": 0.5, "end": 1.0, "confidence": 0.90}
    ]
    
    # Mocking 100 frames per second (1 sec total duration)
    duration_total = 1.0
    
    # 100 points of pitch and energy
    pitch_contour = [150.0] * 100 
    energy_contour = [0.8] * 100
    
    result = WordService.evaluate_words(words, pitch_contour, energy_contour, duration_total)
    
    assert len(result) == 2
    
    # First word: 0.0 to 0.5s -> 50 frames
    w1 = result[0]
    assert w1["word"] == "selamat"
    assert w1["pitch_score"] == 100.0 # 150 Hz matches perfect heuristic
    assert w1["energy_score"] == 80.0 # (0.8/max(1, 0.8))*100 = 80.0
    
def test_evaluate_words_empty_contour():
    words = [
        {"word": "selamat", "start": 0.0, "end": 0.5, "confidence": 0.95}
    ]
    
    result = WordService.evaluate_words(words, [], [], 0.5)
    
    assert len(result) == 1
    w1 = result[0]
    assert w1["pitch_score"] == 50.0 # fallback when 0
    assert w1["energy_score"] == 50.0 # fallback
