import pytest
from app.services.pronunciation.feedback_service import FeedbackService

def test_generate_llm_prompt():
    scoring = {"overall_score": 85, "intonation_score": 80, "fluency_score": 90}
    intonation = {"dtw_distance": 1.5}
    mispronounced = [{"word": "salah", "score": 50, "reason": "Intonasi salah"}]
    
    prompt = FeedbackService.generate_llm_prompt([], mispronounced, intonation, {}, scoring)
    
    assert "85 / 100" in prompt
    assert "salah" in prompt
    assert "Intonasi salah" in prompt

def test_generate_feedback_good():
    scoring = {"overall_score": 90}
    intonation = {"similarity_score": 80}
    
    feedback = FeedbackService.generate_feedback([], [], intonation, {}, scoring)
    
    assert "90" in feedback
    assert "terdengar sangat natural" in feedback
    assert "tidak memiliki kesalahan" in feedback

def test_generate_feedback_bad():
    scoring = {"overall_score": 50}
    intonation = {"similarity_score": 40}
    mispronounced = [{"word": "salah", "score": 50, "reason": "Intonasi salah."}]
    
    feedback = FeedbackService.generate_feedback([], mispronounced, intonation, {}, scoring)
    
    assert "kurang natural" in feedback
    assert "salah" in feedback
    assert "Intonasi salah" in feedback
