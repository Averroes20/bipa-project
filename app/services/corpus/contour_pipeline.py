import numpy as np
from typing import Dict, Any, List
from app.services.pipeline.pitch_service import PitchAnalysisService
from app.services.pipeline.feature_service import FeatureExtractionService

class CorpusContourPipeline:
    @staticmethod
    def extract_contours(audio_22k: np.ndarray, sr_22k: int, words_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts pitch, energy, and pause timeline as JSONB structures.
        """
        pitch_stats = PitchAnalysisService.analyze(audio_22k, sr_22k)
        base_features = FeatureExtractionService.extract_prosody_and_clarity(audio_22k, sr_22k, words_data)
        
        pitch_contour = pitch_stats.get("contour", [])
        energy_contour = base_features.get("energy_contour", [])
        
        # Simple pause timeline based on average pause. 
        # In a real implementation this would map start/end frames of silence.
        pause_timeline = [{"start": 0.0, "end": base_features.get("avg_pause", 0.0)}]
        
        return {
            "pitch_contour": pitch_contour,
            "energy_contour": energy_contour,
            "pause_timeline": pause_timeline
        }
