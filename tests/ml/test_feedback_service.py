import pytest
from app.services.pronunciation.feedback_service import FeedbackService

def test_generate_llm_prompt():
    word_scores = []
    mispronounced = [
        {"word": "bandung", "score": 60, "reason": "vokal terlalu pendek"}
    ]
    intonation = {"dtw_distance": 150}
    vowel = {}
    scoring = {"overall_score": 85, "intonation_score": 75, "fluency_score": 90}
    
    prompt = FeedbackService.generate_llm_prompt(word_scores, mispronounced, intonation, vowel, scoring)
    
    assert "85 / 100" in prompt
    assert "bandung" in prompt
    assert "vokal terlalu pendek" in prompt
    assert "Do not expose raw numbers" in prompt
