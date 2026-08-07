import pytest
from unittest.mock import patch, MagicMock
from app.services.feedback_service import generate_rule_based_feedback, generate_natural_llm_feedback, relative_diff

def test_relative_diff():
    assert relative_diff(120, 100) == 0.2
    assert relative_diff(80, 100) == -0.2
    assert relative_diff(50, 0) == 0.0
    assert relative_diff(None, 100) == -1.0

def test_generate_rule_based_feedback():
    user_features = {"pitch_mean": 130, "energy_mean": 0.5, "pause_ratio": 0.2, "duration": 5.0}
    dataset_reference = {
        "male": {"pitch_mean": 100, "energy_mean": 0.4, "pause_ratio": 0.15, "duration": 4.5},
        "female": {"pitch_mean": 200, "energy_mean": 0.6, "pause_ratio": 0.25, "duration": 5.5}
    }
    
    # User is closer to female (magnitude 0.8 vs 0.99)
    result = generate_rule_based_feedback(user_features, dataset_reference)
    
    assert "cenderung female" in result["voice_profile"]
    assert "pitch" in result["features"]
    assert "energy" in result["features"]
    assert "pause" in result["features"]
    assert "perempuan" in result["features"]["pitch"]

def test_generate_natural_llm_feedback_no_api_key():
    with patch("os.getenv", return_value=""):
        result = generate_natural_llm_feedback({})
        assert "Pelafalan dan intonasi" in result

def test_generate_natural_llm_feedback_with_mock_genai():
    with patch("os.getenv", return_value="mock_key"), \
         patch("app.services.feedback_service.genai.GenerativeModel") as mock_model_class:
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Ini respons dari LLM"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        result = generate_natural_llm_feedback({})
        
        assert result == "Ini respons dari LLM"
        mock_model.generate_content.assert_called_once()
