import pytest
from app.services.pipeline.phoneme_detection_service import PhonemeDetectionService

def test_analyze_empty():
    result = PhonemeDetectionService.analyze([])
    assert result["critical_phonemes_accuracy"] == 100.0
    assert result["details"] == {}

def test_analyze_with_errors():
    errors = [
        {"expected": "r", "detected": "l", "is_correct": False, "confidence": 0.8},
        {"expected": "r", "detected": "r", "is_correct": True, "confidence": 0.9},
        {"expected": "ng", "detected": "n", "is_correct": False, "confidence": 0.7},
        {"expected": "x", "detected": "x", "is_correct": True, "confidence": 0.9} # Non-critical
    ]
    result = PhonemeDetectionService.analyze(errors)
    
    assert result["critical_phonemes_accuracy"] == 25.0  # (50% for r + 0% for ng) / 2
    assert "r" in result["details"]
    assert result["details"]["r"]["accuracy"] == 50.0
    assert result["details"]["r"]["occurrences"] == 2
    assert result["details"]["r"]["common_mistake"] == "l"
    
    assert "ng" in result["details"]
    assert result["details"]["ng"]["accuracy"] == 0.0
    assert result["details"]["ng"]["occurrences"] == 1
    assert result["details"]["ng"]["common_mistake"] == "n"
    
    assert "x" not in result["details"]
