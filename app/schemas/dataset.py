from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

class DatasetAudioBase(BaseModel):
    id: str
    filename: str
    gender: Optional[str]
    duration: float
    language: str
    created_at: datetime
    
class DatasetAudioList(BaseModel):
    items: List[DatasetAudioBase]
    total: int
    page: int
    size: int

class DatasetFeatureSchema(BaseModel):
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    energy_mean: float
    pause_ratio: float
    speech_rate: float
    zcr: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    # Omit embeddings and mfcc for generic responses unless needed

class DatasetFormantSchema(BaseModel):
    f1: Optional[float]
    f2: Optional[float]
    f3: Optional[float]
    vowel_profile: Optional[Any]

class DatasetDetail(DatasetAudioBase):
    feature: Optional[DatasetFeatureSchema]
    formant: Optional[DatasetFormantSchema]
    
class StatisticsResponse(BaseModel):
    total_audio: int
    male_count: int
    female_count: int
    avg_duration: float
    avg_pitch: float
    avg_energy: float
    gender_distribution: Any
    pitch_distribution: Any
    energy_distribution: Any
    duration_distribution: Any
