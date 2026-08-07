import numpy as np
import librosa
from typing import Dict, Any, List
from app.services.pipeline.feature_service import FeatureExtractionService
from app.services.pipeline.pitch_service import PitchAnalysisService

class CorpusFeaturePipeline:
    @staticmethod
    def extract_features(audio_22k: np.ndarray, sr_22k: int, words_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts pitch, energy, pause, speech_rate, mfcc, zcr, and spectral features.
        """
        # Get baseline features from existing services
        base_features = FeatureExtractionService.extract_prosody_and_clarity(audio_22k, sr_22k, words_data)
        pitch_stats = PitchAnalysisService.analyze(audio_22k, sr_22k)
        
        # Calculate additional corpus-specific spectral features
        mfcc = librosa.feature.mfcc(y=audio_22k, sr=sr_22k, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(audio_22k)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_22k, sr=sr_22k)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_22k, sr=sr_22k)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_22k, sr=sr_22k)[0]
        
        pitch_contour = pitch_stats.get("contour", [])
        pitch_std = float(np.std(pitch_contour)) if pitch_contour else 0.0
        
        return {
            "pitch_mean": pitch_stats.get("mean", 0.0),
            "pitch_std": pitch_std,
            "pitch_range": pitch_stats.get("range", 0.0),
            "energy_mean": float(np.mean(base_features["energy_contour"])) if base_features.get("energy_contour") else 0.0,
            "pause_ratio": base_features.get("pause_ratio", 0.0),
            "speech_rate": base_features.get("wpm", 0.0) / 60.0 if base_features.get("wpm") else 0.0,
            "mfcc": mfcc.mean(axis=1).tolist(),
            "zcr": float(np.mean(zcr)),
            "spectral_centroid": float(np.mean(spectral_centroid)),
            "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
            "spectral_rolloff": float(np.mean(spectral_rolloff))
        }
