import librosa
import numpy as np
from typing import Dict, Any

class ArticulationAnalysisService:
    @staticmethod
    def analyze(audio_22k: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extracts ZCR, Spectral Centroid, Bandwidth, and Contrast to determine Speech Clarity.
        """
        if len(audio_22k) == 0:
            return {
                "zcr": 0.0,
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_contrast": 0.0,
                "speech_clarity": 0.0
            }
            
        zcr = librosa.feature.zero_crossing_rate(audio_22k)[0]
        centroid = librosa.feature.spectral_centroid(y=audio_22k, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_22k, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(y=audio_22k, sr=sr)
        
        zcr_mean = float(np.mean(zcr))
        centroid_mean = float(np.mean(centroid))
        bandwidth_mean = float(np.mean(bandwidth))
        contrast_mean = float(np.mean(contrast))
        
        # Calculate Speech Clarity index heuristically based on distinctness of spectrum
        # Good articulation usually means a well-defined contrast and controlled centroid/zcr
        # ZCR for speech is typically around 0.05 to 0.15
        zcr_score = max(0, 100 - abs(zcr_mean - 0.1) * 500)
        
        # Contrast indicates peak distinctness (higher is better for clarity)
        contrast_score = min(100, contrast_mean * 5)
        
        # Bandwidth indicates frequency spread
        bw_score = min(100, (bandwidth_mean / 3000.0) * 100)
        
        clarity_score = (zcr_score * 0.3) + (contrast_score * 0.5) + (bw_score * 0.2)
        clarity_score = max(0.0, min(100.0, clarity_score))
        
        return {
            "zcr": round(zcr_mean, 4),
            "spectral_centroid": round(centroid_mean, 2),
            "spectral_bandwidth": round(bandwidth_mean, 2),
            "spectral_contrast": round(contrast_mean, 2),
            "speech_clarity": round(clarity_score, 1)
        }
