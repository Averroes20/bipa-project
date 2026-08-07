import pytest
from unittest.mock import patch, MagicMock
from app.services.pipeline.formant_service import FormantAnalysisService, classify_vowel

def test_classify_vowel():
    assert classify_vowel(400, 1900) == "i"
    assert classify_vowel(400, 1000) == "u"
    assert classify_vowel(700, 1200) == "a"
    assert classify_vowel(500, 1700) == "e"
    assert classify_vowel(500, 1000) == "o"
    assert classify_vowel(900, 900) == "unknown"

def test_analyze_exception():
    with patch("builtins.__import__", side_effect=Exception("Mocked error")):
        result = FormantAnalysisService.analyze("dummy.wav")
        assert result == {"vowelSpace": []}

def test_analyze_success():
    mock_sound = MagicMock()
    mock_intensity = MagicMock()
    mock_formant = MagicMock()
    
    mock_sound.to_intensity.return_value = mock_intensity
    mock_sound.to_formant_burg.return_value = mock_formant
    mock_sound.get_total_duration.return_value = 0.05  # 5 frames
    
    mock_intensity.get_value.return_value = 60.0  # Above silence threshold
    
    # Return F1=400, F2=1900 -> 'i'
    def mock_get_value_at_time(formant_num, time):
        return 400.0 if formant_num == 1 else 1900.0
        
    mock_formant.get_value_at_time.side_effect = mock_get_value_at_time
    
    with patch("parselmouth.Sound", return_value=mock_sound):
        result = FormantAnalysisService.analyze("dummy.wav")
        
        # 5 frames are enough to cluster 'i' (>3)
        assert len(result["vowelSpace"]) == 1
        assert result["vowelSpace"][0]["vowel"] == "i"
        assert result["vowelSpace"][0]["f1"] == 400.0
        assert result["vowelSpace"][0]["f2"] == 1900.0
