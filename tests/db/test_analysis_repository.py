import pytest
from app.repositories.analysis_repository import AnalysisRepository
from app.models.audio_models import AudioAnalysis
from unittest.mock import MagicMock

def test_save_analysis():
    mock_db = MagicMock()
    repo = AnalysisRepository(mock_db)
    
    # Simulate DB add and commit
    user_features = {"duration": 1.5, "pitch_mean": 150}
    score_result = {"overall": 80, "male": {"dtw": 10}, "female": {"dtw": 15}}
    
    analysis = repo.save_analysis(
        user_id="test_user",
        user_features=user_features,
        score_result=score_result,
        embedding_scores={"male": 80, "female": 70},
        user_emb=[0.1, 0.2],
        ai_feedback="Bagus!",
        gender_label="Male",
        analysis_detail="{}"
    )
    
    assert mock_db.add.called
    assert mock_db.commit.called
    assert analysis.user_id == "test_user"
    assert analysis.final_score == 80
