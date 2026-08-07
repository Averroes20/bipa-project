import pytest
import numpy as np
from app.services.pipeline.articulation_service import ArticulationAnalysisService

def test_analyze_empty():
    result = ArticulationAnalysisService.analyze(np.array([]), 22050)
    assert result["zcr"] == 0.0
    assert result["speech_clarity"] == 0.0

def test_analyze_with_data(dummy_audio):
    result = ArticulationAnalysisService.analyze(dummy_audio, 22050)
    assert "zcr" in result
    assert "spectral_centroid" in result
    assert "spectral_bandwidth" in result
    assert "spectral_contrast" in result
    assert "speech_clarity" in result
