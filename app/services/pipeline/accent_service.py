import numpy as np
from typing import Dict, Any, List

class AccentAnalysisService:
    @staticmethod
    def analyze(features: Dict[str, Any], pitch_stats: Dict[str, Any], phonemes_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyzes Speaking Rate, Rhythm, Stress, Pitch Contour, Pause Pattern,
        and classifies accent.
        """
        wpm = features.get("wpm", 0)
        pause_ratio = features.get("pause_ratio", 0.0)
        pitch_variance = pitch_stats.get("variance", 0.0)
        
        # Calculate Rhythm (syllable duration variance)
        durations = [p["duration"] for p in phonemes_data if p["duration"] > 0]
        rhythm_variance = float(np.var(durations)) if durations else 0.0
        
        # Stress Pattern (using energy spikes from features)
        energy = features.get("energy_contour", [])
        if energy:
            energy_arr = np.array(energy)
            # Find peaks (simple threshold above 1.5x mean)
            mean_e = np.mean(energy_arr)
            stress_points = np.sum(energy_arr > 1.5 * mean_e)
            stress_density = float(stress_points / max(1, len(energy_arr)))
        else:
            stress_density = 0.0
            
        # Classify accent based on heuristics
        # Indonesian: Syllable-timed (low rhythm_variance), moderate pitch variance, steady stress
        # English: Stress-timed (high rhythm_variance), high pitch variance, high stress density
        # Mandarin: Tonal (very high pitch variance within syllables - approx by overall high variance but low stress density)
        
        scores = {
            "native_indonesia": 0,
            "english": 0,
            "mandarin": 0,
            "other": 0
        }
        
        # Indonesian rule
        if rhythm_variance < 0.01 and 10 < pitch_variance < 50:
            scores["native_indonesia"] += 3
        # English rule
        if rhythm_variance > 0.015 and pitch_variance > 40 and stress_density > 0.1:
            scores["english"] += 3
        # Mandarin rule
        if pitch_variance > 60 and stress_density < 0.1:
            scores["mandarin"] += 3
            
        # Add default scores for unclassified
        scores["other"] += 1
        
        best_match = max(scores.items(), key=lambda x: x[1])[0]
        
        classified = "Native Indonesia"
        if best_match == "english":
            classified = "English"
        elif best_match == "mandarin":
            classified = "Mandarin"
        elif best_match == "other" and sum(scores.values()) == 1:
            classified = "Other/Unknown"
            
        return {
            "speaking_rate_wpm": round(wpm, 1),
            "rhythm_variance": round(rhythm_variance, 4),
            "stress_density": round(stress_density, 3),
            "pitch_variance": round(pitch_variance, 2),
            "pause_ratio": round(pause_ratio, 2),
            "accent_classification": classified
        }
