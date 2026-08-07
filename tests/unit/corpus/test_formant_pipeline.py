import pytest
from unittest.mock import patch
from app.services.corpus.formant_pipeline import CorpusFormantPipeline

def test_extract_formants_empty():
    with patch("app.services.corpus.formant_pipeline.VowelAnalysisService.extract_vowels", return_value={}):
        result = CorpusFormantPipeline.extract_formants("dummy.wav", [])
        assert result["f1"] == 0.0
        assert result["f2"] == 0.0
        assert result["f3"] == 0.0
        assert result["vowel_profile"] == []

def test_extract_formants_with_data():
    mock_vowels = {
        "vowelSpace": [
            {"F1": 500, "F2": 1500, "F3": 2500},
            {"F1": 700, "F2": 1100, "F3": 2100}
        ]
    }
    with patch("app.services.corpus.formant_pipeline.VowelAnalysisService.extract_vowels", return_value=mock_vowels):
        result = CorpusFormantPipeline.extract_formants("dummy.wav", [])
        assert result["f1"] == 600.0
        assert result["f2"] == 1300.0
        assert result["f3"] == 2300.0
        assert len(result["vowel_profile"]) == 2
