from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DimensionsSchema(BaseModel):
    intonation: float
    pronunciation: float
    fluency: float
    clarity: float
    accent: float

class SimilaritySchema(BaseModel):
    male: float
    female: float

class WordSchema(BaseModel):
    word: str
    start: float
    end: float
    duration: float
    confidence: float

class PhonemeSchema(BaseModel):
    symbol: str
    start: float
    end: float
    duration: float
    confidence: float

class PronunciationSchema(BaseModel):
    transcription: str
    words: List[WordSchema]
    phonemes: List[PhonemeSchema]
    pronunciation_score: float
    word_score: float
    phoneme_score: float

class PitchSchema(BaseModel):
    mean: Optional[float] = None
    range: Optional[float] = None
    contour: List[float]

class EnergySchema(BaseModel):
    mean: Optional[float] = None
    contour: List[float]

class TimelineEvent(BaseModel):
    start: float
    end: float

class PauseSchema(BaseModel):
    ratio: float
    timeline: List[TimelineEvent]

class PhoneticVowelSchema(BaseModel):
    vowel: str
    accuracy: float

class PhoneticsSchema(BaseModel):
    vowel_space: List[Dict[str, Any]] = []
    formants: Dict[str, Optional[float]] = {}
    vowels: List[PhoneticVowelSchema] = []

class ArticulationSchema(BaseModel):
    zcr: Optional[float] = None
    spectral_centroid: Optional[float] = None
    spectral_bandwidth: Optional[float] = None
    spectral_contrast: Optional[float] = None
    speech_clarity: Optional[float] = None

class AccentSchema(BaseModel):
    speaking_rate_wpm: float
    rhythm_variance: float
    stress_density: float
    pitch_variance: float
    pause_ratio: float
    accent_classification: str

class IntonationSchema(BaseModel):
    sentence_ending: str
    pattern: str
    pitch_variance: Optional[float] = None

class PhonemeDetectionDetailSchema(BaseModel):
    accuracy: float
    occurrences: int
    common_mistake: Optional[str] = None

class PhonemeDetectionSchema(BaseModel):
    critical_phonemes_accuracy: float
    details: Dict[str, PhonemeDetectionDetailSchema]

class VowelDetailSchema(BaseModel):
    vowel: str
    f1: Optional[float] = None
    f2: Optional[float] = None
    f3: Optional[float] = None
    time: Optional[float] = None

class VowelAnalysisSchema(BaseModel):
    vowels: List[VowelDetailSchema]
    f1_mean: float
    f2_mean: float
    f3_mean: float
    vsa: float

class AnalysisResponse(BaseModel):
    id: str = Field(default="")
    overall_score: float
    dimensions: DimensionsSchema
    similarity: SimilaritySchema
    voice_profile: str
    pronunciation: PronunciationSchema
    pitch: PitchSchema
    energy: EnergySchema
    pause: PauseSchema
    phonetics: PhoneticsSchema
    articulation: ArticulationSchema
    accent: AccentSchema
    intonation: IntonationSchema
    phoneme_detection: PhonemeDetectionSchema
    vowel_analysis: VowelAnalysisSchema
    errors: List[Any]
    recommendation: List[str]
    analysisMetadata: Dict[str, Any]
