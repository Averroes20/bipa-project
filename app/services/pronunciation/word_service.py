import numpy as np
from typing import Dict, Any, List

class WordService:
    @staticmethod
    def evaluate_words(words: List[Dict[str, Any]], pitch_contour: List[float], energy_contour: List[float], duration_total: float) -> List[Dict[str, Any]]:
        """
        Calculates acoustic scores for each word boundary.
        Args:
            words: [{"word": "selamat", "start": 0.21, "end": 0.74, "confidence": 0.98}]
            pitch_contour: Un-downsampled pitch contour matching time.
            energy_contour: Un-downsampled energy contour matching time.
            duration_total: Total duration of the audio in seconds.
        """
        evaluated_words = []
        
        # We assume contour represents frames, let's calculate frame rate roughly
        if duration_total <= 0:
            frame_rate = 100 # Fallback 100 fps (0.01 hop size)
        else:
            frame_rate = len(pitch_contour) / duration_total
            
        for w in words:
            start = w["start"]
            end = w["end"]
            
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            
            # Slice contours for this word
            p_slice = pitch_contour[start_frame:end_frame]
            e_slice = energy_contour[start_frame:end_frame]
            
            # Compute Pitch Score
            p_slice_valid = [p for p in p_slice if p > 0]
            pitch_mean = np.mean(p_slice_valid) if p_slice_valid else 0.0
            
            # Compute Energy Score
            energy_mean = np.mean(e_slice) if e_slice else 0.0
            
            # Compute Duration
            duration = end - start
            
            # Very basic scoring logic for now (baseline for further refinement by AI Teacher)
            pitch_score = min(100, max(0, 100 - abs(pitch_mean - 150) / 3.0)) if pitch_mean > 0 else 50
            energy_score = min(100, max(0, (energy_mean / max(1, np.max(energy_contour))) * 100)) if len(energy_contour) > 0 else 50
            duration_score = min(100, max(0, 100 - abs(duration - 0.4) * 100)) # Assumes 0.4s is optimal average word length
            
            # Stress score based on max energy vs mean energy
            max_e = np.max(e_slice) if e_slice else 0.0
            stress_score = min(100, max(0, (max_e / (energy_mean + 1e-6)) * 50))
            
            # Base pronunciation score relies on whisper's confidence
            pronunciation_score = w["confidence"] * 100
            
            overall_score = (pronunciation_score * 0.4) + (pitch_score * 0.2) + (energy_score * 0.2) + (duration_score * 0.1) + (stress_score * 0.1)
            
            evaluated_words.append({
                "word": w["word"],
                "start_time": start,
                "end_time": end,
                "confidence": w["confidence"],
                "pronunciation_score": round(pronunciation_score, 2),
                "pitch_score": round(pitch_score, 2),
                "energy_score": round(energy_score, 2),
                "duration_score": round(duration_score, 2),
                "stress_score": round(stress_score, 2),
                "overall_score": round(overall_score, 2)
            })
            
        return evaluated_words
