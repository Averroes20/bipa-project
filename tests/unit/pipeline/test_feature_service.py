import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.pipeline.feature_service import FeatureExtractionService

def test_extract_prosody_and_clarity(dummy_audio):
    words_data = [{"word": "test"}, {"word": "data"}]
    audio = np.ones(16000, dtype=np.float32) * 0.1
    result = FeatureExtractionService.extract_prosody_and_clarity(audio, 22050, words_data)
    
    assert "duration" in result
    assert "wpm" in result
    assert result["snr"] >= 0.0
    assert "clarity_score" in result
    assert "energy_contour" in result

def test_extract_native_similarity():
    mock_audio = np.zeros(16000)
    mock_repo = MagicMock()
    mock_repo.get_top_k_candidates.side_effect = [
        [{"embedding_vector": "[0.1, 0.2]", "id": 1}], # male
        [{"embedding_vector": "[0.2, 0.4]", "id": 2}]  # female
    ]
    
    mock_emb = np.array([0.1, 0.2])
    with patch("app.services.pipeline.feature_service.extract_embedding", return_value=mock_emb):
        result = FeatureExtractionService.extract_native_similarity(mock_audio, 16000, mock_repo)
        
        assert "similarity" in result
        assert "reference_gender" in result
        assert result["best_male_candidate"]["id"] == 1
        assert result["best_female_candidate"]["id"] == 2
