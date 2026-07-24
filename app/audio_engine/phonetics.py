import numpy as np
import librosa
from typing import Dict, Any
from app.audio_engine.prosody import extract_formants

def safe_float(v, default=0.0):
    try:
        if v is None or np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default

def analyze_articulation(audio_path: str) -> Dict[str, Any]:
    """Analyze articulation clarity via Zero Crossing Rate & Spectral Centroid."""
    audio, sr = librosa.load(audio_path, sr=22050)

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    zcr_mean = safe_float(np.mean(zcr))
    centroid_mean = safe_float(np.mean(centroid))

    clarity_score = ((centroid_mean / 4000.0) * 0.6) + ((zcr_mean * 10.0) * 0.4)
    clarity_score = max(0.0, min(1.0, clarity_score))

    if clarity_score > 0.75:
        clarity_label = "artikulasi sangat jelas"
    elif clarity_score > 0.5:
        clarity_label = "artikulasi cukup jelas"
    else:
        clarity_label = "artikulasi masih kurang jelas"

    return {
        "zcr_mean": zcr_mean,
        "spectral_centroid": centroid_mean,
        "clarity_score": clarity_score,
        "clarity_label": clarity_label
    }

def detect_accent(
    pitch_mean: float,
    pause_ratio: float,
    duration: float,
    articulation_score: float
) -> Dict[str, Any]:
    """Heuristic accent insight generator."""
    insights = []

    if pause_ratio > 0.35:
        insights.append("ritme bicara cenderung terputus-putus seperti penutur English")
    if pitch_mean > 210:
        insights.append("intonasi terdengar cukup tinggi dan stabil")
    if duration < 3:
        insights.append("tempo bicara terdengar cukup cepat")
    if articulation_score < 0.4:
        insights.append("pengucapan beberapa kata masih kurang tegas")

    if not insights:
        insights.append("pelafalan terdengar cukup natural")

    return {"accent_insight": insights}

def classify_vowel_profile(f1: float, f2: float) -> str:
    if f1 < 400 and f2 > 2000:
        return "i"
    if f1 < 500 and f2 < 1200:
        return "u"
    if f1 > 700 and f2 > 1200:
        return "a"
    if f1 < 600 and f2 > 1700:
        return "e"
    return "o"

def analyze_phonetics(audio_path: str) -> Dict[str, Any]:
    formants = extract_formants(audio_path)
    vowel_profile = classify_vowel_profile(formants["f1"], formants["f2"])
    
    localized_vowels = []
    for sample in formants.get("samples", []):
        phoneme = classify_vowel_profile(sample["f1"], sample["f2"])
        localized_vowels.append({
            "phoneme": phoneme,
            "f1": sample["f1"],
            "f2": sample["f2"]
        })
    
    return {
        "formants": formants,
        "vowel_profile": vowel_profile,
        "localized_vowels": localized_vowels
    }

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """Whisper ASR Transcription."""
    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        print("Whisper transcription unavailable:", e)
        return ""
