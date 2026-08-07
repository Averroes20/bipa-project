import pytest
from app.services.pronunciation.scoring_service import ScoringService

def test_aggregate_scores_empty():
    result = ScoringService.aggregate_scores([], [], {}, {}, {}, 0.0)
    assert result["overall_score"] == 0.0

def test_aggregate_scores_with_data():
    word_scores = [{"overall_score": 80.0}]
    phoneme_scores = [{"pronunciation_score": 90.0}]
    intonation_data = {"similarity_score": 75.0}
    accent_data = {"accent_score": 70.0, "rhythm_score": 85.0}
    vowel_data = {"user_space": [{"distance_male": 150, "distance_female": 200}]}
    
    result = ScoringService.aggregate_scores(
        word_scores, phoneme_scores, intonation_data, accent_data, vowel_data, 0.8
    )
    
    assert result["word_score"] == 80.0
    assert result["phoneme_score"] == 90.0
    assert result["pronunciation_score"] == 85.0  # (80*0.5 + 90*0.5)
    assert result["vowel_score"] == 50.0  # 100 - (150/300)*100
    assert result["fluency_score"] == 80.0
    assert result["overall_score"] > 0
