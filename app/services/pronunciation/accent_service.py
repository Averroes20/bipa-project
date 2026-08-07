from typing import Dict, Any, List
import numpy as np

class AccentService:
    @staticmethod
    def analyze(words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the rhythm and accent based on word durations and stress.
        """
        if not words_data:
            return {
                "rhythm_score": 0.0,
                "accent_score": 0.0
            }
            
        durations = [w["duration_score"] for w in words_data]
        stresses = [w["stress_score"] for w in words_data]
        
        # Rhythm is consistency in duration and expected stress
        rhythm_score = float(np.mean(durations)) if durations else 0.0
        accent_score = float(np.mean(stresses)) if stresses else 0.0
        
        return {
            "rhythm_score": round(rhythm_score, 2),
            "accent_score": round(accent_score, 2)
        }
