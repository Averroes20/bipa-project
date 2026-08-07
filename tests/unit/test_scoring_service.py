import pytest
from app.services.pronunciation.scoring_service import ScoringService

def test_aggregate_scores():
    word_scores = [{"overall_score": 80}, {"overall_score": 90}]
    phoneme_scores = [{"pronunciation_score": 70}, {"pronunciation_score": 80}]
    intonation_data = {"similarity_score": 85.0}
    accent_data = {"accent_score": 90.0, "rhythm_score": 88.0}
    vowel_data = {
        "user_space": [
            {"distance_male": 150, "distance_female": 200}, # min dist = 150
            {"distance_male": 350, "distance_female": 400}  # min dist = 350
        ]
    }
    # Avg dist = 250. vowel_score = max(0, 100 - (250/300)*100) = 16.67
    
    fluency_ratio = 0.8 # 80.0 fluency score
    
    result = ScoringService.aggregate_scores(
        word_scores, phoneme_scores, intonation_data, accent_data, vowel_data, fluency_ratio
    )
    
    assert result["word_score"] == 85.0
    assert result["phoneme_score"] == 75.0
    assert result["pronunciation_score"] == 80.0 # (85*0.5) + (75*0.5)
    assert result["intonation_score"] == 85.0
    assert result["accent_score"] == 90.0
    assert result["rhythm_score"] == 88.0
    assert result["fluency_score"] == 80.0
    
    assert "overall_score" in result
    assert result["overall_score"] > 0
