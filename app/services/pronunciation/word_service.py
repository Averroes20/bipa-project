import numpy as np
from typing import Dict, Any, List

class WordService:
    @staticmethod
    def evaluate_words(words: List[Dict[str, Any]], pitch_contour: List[float], energy_contour: List[float], duration_total: float) -> List[Dict[str, Any]]:
        """
        Calculates acoustic scores for each word boundary.
        """
        evaluated_words = []
        
        if duration_total <= 0:
            frame_rate = 100 
        else:
            frame_rate = len(pitch_contour) / duration_total
            
        for w in words:
            start = w["start"]
            end = w["end"]
            
            # Missing words due to alignment
            if start == 0.0 and end == 0.0 and w.get("missing"):
                evaluated_words.append({
                    "word": w["word"],
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "confidence": 0.0,
                    "pitch_score": 0.0,
                    "energy_score": 0.0,
                    "duration_score": 0.0,
                    "stress_score": 0.0
                })
                continue

            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            
            p_slice = pitch_contour[start_frame:end_frame]
            e_slice = energy_contour[start_frame:end_frame]
            
            p_slice_valid = [p for p in p_slice if p > 0]
            pitch_mean = np.mean(p_slice_valid) if p_slice_valid else 0.0
            energy_mean = np.mean(e_slice) if e_slice else 0.0
            
            duration = end - start
            
            pitch_score = min(100, max(0, 100 - abs(pitch_mean - 150) / 3.0)) if pitch_mean > 0 else 50
            energy_score = min(100, max(0, (energy_mean / max(1, np.max(energy_contour))) * 100)) if len(energy_contour) > 0 else 50
            duration_score = min(100, max(0, 100 - abs(duration - 0.4) * 100)) 
            
            max_e = np.max(e_slice) if e_slice else 0.0
            stress_score = min(100, max(0, (max_e / (energy_mean + 1e-6)) * 50))
            
            evaluated_words.append({
                "word": w["word"],
                "start_time": start,
                "end_time": end,
                "confidence": w["confidence"],
                "pitch_score": round(pitch_score, 2),
                "energy_score": round(energy_score, 2),
                "duration_score": round(duration_score, 2),
                "stress_score": round(stress_score, 2)
            })
            
        return evaluated_words
