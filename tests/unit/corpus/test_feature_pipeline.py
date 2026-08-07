import pytest
import numpy as np
from unittest.mock import patch
from app.services.corpus.feature_pipeline import CorpusFeaturePipeline

def test_extract_features(dummy_audio):
    mock_base_features = {
        "energy_contour": [0.1, 0.2, 0.3],
        "pause_ratio": 0.15,
        "wpm": 120.0
    }
    mock_pitch_stats = {
        "mean": 150.0,
        "range": 50.0,
        "contour": [140.0, 150.0, 160.0]
    }
    
    with patch("app.services.corpus.feature_pipeline.FeatureExtractionService.extract_prosody_and_clarity", return_value=mock_base_features), \
         patch("app.services.corpus.feature_pipeline.PitchAnalysisService.analyze", return_value=mock_pitch_stats):
         
        result = CorpusFeaturePipeline.extract_features(dummy_audio, 22050, [])
        
        assert result["pitch_mean"] == 150.0
        assert result["pitch_range"] == 50.0
        assert result["energy_mean"] == np.mean([0.1, 0.2, 0.3])
        assert result["pause_ratio"] == 0.15
        assert result["speech_rate"] == 2.0  # 120 / 60
        assert "mfcc" in result
        assert "zcr" in result
        assert "spectral_centroid" in result
        assert "spectral_bandwidth" in result
        assert "spectral_rolloff" in result
