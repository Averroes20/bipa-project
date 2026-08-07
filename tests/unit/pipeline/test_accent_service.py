import pytest
from app.services.pipeline.accent_service import AccentAnalysisService

def test_analyze_indonesian():
    features = {"wpm": 120, "pause_ratio": 0.1, "energy_contour": [0.1, 0.1, 0.1]} # Low stress density
    pitch_stats = {"variance": 20.0} # moderate variance
    phonemes_data = [{"duration": 0.1}, {"duration": 0.11}] # very low variance
    
    result = AccentAnalysisService.analyze(features, pitch_stats, phonemes_data)
    assert result["accent_classification"] == "Native Indonesia"

def test_analyze_english():
    features = {"wpm": 150, "pause_ratio": 0.1, "energy_contour": [0.1, 0.5, 0.1, 0.6]} # High stress density
    pitch_stats = {"variance": 50.0}
    phonemes_data = [{"duration": 0.05}, {"duration": 0.3}] # high rhythm variance
    
    result = AccentAnalysisService.analyze(features, pitch_stats, phonemes_data)
    assert result["accent_classification"] == "English"
    
def test_analyze_mandarin():
    features = {"wpm": 130, "pause_ratio": 0.1, "energy_contour": [0.1, 0.11, 0.1]} # Low stress density
    pitch_stats = {"variance": 80.0} # Very high variance
    phonemes_data = [{"duration": 0.1}, {"duration": 0.12}]
    
    result = AccentAnalysisService.analyze(features, pitch_stats, phonemes_data)
    assert result["accent_classification"] == "Mandarin"
