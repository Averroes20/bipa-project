import numpy as np
import librosa
from scipy.signal import medfilt
from typing import Dict, Any

class PitchAnalysisService:
    @staticmethod
    def analyze(audio: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
        """
        Extracts Fundamental Frequency (F0) using probabilistic YIN (pyin).
        Smooths contour, removes outliers, and calculates statistics.
        """
        # Define expected human pitch range (65Hz - 1046Hz, ~C2 to C6)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            sr=sr, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C6')
        )
        
        # Filter out unvoiced frames (NaNs)
        if f0 is None or len(f0) == 0:
            return {
                "contour": [],
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "range": 0.0,
                "variance": 0.0
            }

        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) == 0:
             return {
                "contour": [],
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "range": 0.0,
                "variance": 0.0
            }

        # Apply median filter to remove octave jump outliers
        kernel_size = min(5, len(f0_valid))
        if kernel_size % 2 == 0:
            kernel_size -= 1
            
        f0_smoothed = medfilt(f0_valid, kernel_size=max(3, kernel_size))
        
        # Calculate stats
        mean_pitch = float(np.mean(f0_smoothed))
        max_pitch = float(np.max(f0_smoothed))
        min_pitch = float(np.min(f0_smoothed))
        pitch_range = max_pitch - min_pitch
        variance = float(np.var(f0_smoothed))

        # Downsample contour for frontend performance (max 100 points)
        contour = f0_smoothed.tolist()
        if len(contour) > 100:
            indices = np.linspace(0, len(contour) - 1, 100, dtype=int)
            contour = [round(contour[i], 2) for i in indices]
        else:
            contour = [round(v, 2) for v in contour]

        return {
            "contour": contour,
            "mean": round(mean_pitch, 2),
            "max": round(max_pitch, 2),
            "min": round(min_pitch, 2),
            "range": round(pitch_range, 2),
            "variance": round(variance, 2)
        }
