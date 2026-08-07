import pytest
from app.services.pipeline.feedback_service import FeedbackService

def test_generate_feedback_native():
    pronunciation = {"pronunciation_score": 90}
    vowel = {"vsa": 15000}
    articulation = {"speech_clarity": 85}
    accent = {"accent_classification": "Native Indonesia"}
    intonation = {"pattern": "Statement", "sentence_ending": "Flat"}
    phoneme_det = {"critical_phonemes_accuracy": 95, "details": {}}
    
    insights = FeedbackService.generate(pronunciation, vowel, articulation, accent, intonation, phoneme_det)
    
    assert any("setara penutur asli" in s for s in insights)
    assert any("tegas" in s for s in insights)
    assert any("Aksen dan ritme" in s for s in insights)
    assert any("dilafalkan dengan sangat baik" in s for s in insights)

def test_generate_feedback_needs_improvement():
    pronunciation = {"pronunciation_score": 50}
    vowel = {"vsa": 5000}
    articulation = {"speech_clarity": 50}
    accent = {"accent_classification": "English"}
    intonation = {"pattern": "Statement", "sentence_ending": "Flat"}
    phoneme_det = {
        "critical_phonemes_accuracy": 50,
        "details": {"r": {"accuracy": 40, "common_mistake": "l"}}
    }
    
    insights = FeedbackService.generate(pronunciation, vowel, articulation, accent, intonation, phoneme_det)
    
    assert any("kurang jelas" in s for s in insights)
    assert any("sempit" in s for s in insights)
    assert any("stress-timed" in s for s in insights)
    assert any("'r' (terdengar seperti 'l')" in s for s in insights)
