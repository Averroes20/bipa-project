from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union

class DimensionsScore(BaseModel):
    intonation: float
    pronunciation: float
    fluency: float
    clarity: float
    accent: float

class SimilarityScore(BaseModel):
    male: float
    female: float

class PitchData(BaseModel):
    mean: float
    range: float
    contour: List[float]

class EnergyData(BaseModel):
    mean: float
    contour: List[float]

class PauseTimelineItem(BaseModel):
    start: float
    end: float

class PauseData(BaseModel):
    ratio: float
    timeline: List[PauseTimelineItem]

class FormantsData(BaseModel):
    F1: float
    F2: float
    F3: float

class VowelAccuracy(BaseModel):
    vowel: str
    accuracy: float

class PhoneticsData(BaseModel):
    formants: FormantsData
    vowels: List[VowelAccuracy]

class PhonemeScore(BaseModel):
    symbol: str
    start: float
    end: float
    score: float

class WordScore(BaseModel):
    word: str
    score: float
    start: Optional[float] = None
    end: Optional[float] = None
    status: Optional[str] = "ok" # "ok", "error"

class PronunciationError(BaseModel):
    word: str
    expected: str
    spoken: str
    severity: str # "low", "medium", "high"

class PronunciationData(BaseModel):
    phonemes: List[PhonemeScore]
    words: List[WordScore]
    errors: List[PronunciationError]

class RecommendationItem(BaseModel):
    type: str # "Strength", "Weakness", "Suggestion", "Practice"
    message: str

class AnalysisResultResponse(BaseModel):
    overall_score: float
    dimensions: DimensionsScore
    similarity: SimilarityScore
    voice_profile: str
    pitch: PitchData
    energy: EnergyData
    pause: PauseData
    phonetics: PhoneticsData
    pronunciation: PronunciationData
    recommendation: List[RecommendationItem]

class DashboardSummaryResponse(BaseModel):
    total_analysis: int
    avg_score: float
    avg_pitch: float
    avg_energy: float
    avg_pause: float

class UserProgressResponse(BaseModel):
    avg_score: float
    best_score: float
    sessions: int
