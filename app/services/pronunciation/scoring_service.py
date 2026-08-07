from typing import Dict, Any, List
import numpy as np

class ScoringService:
    @staticmethod
    def aggregate_scores(
        word_scores: List[Dict],
        phoneme_scores: List[Dict],
        intonation_data: Dict,
        accent_data: Dict,
        vowel_data: Dict,
        fluency_ratio: float
    ) -> Dict[str, float]:
        """
        Aggregates all hierarchical scores into a final holistic score profile.
        Pipeline: Word -> Pronunciation -> Phoneme -> Intonation -> Accent -> Rhythm -> Fluency -> Overall Score
        """
        
        # Word-level aggregation
        if word_scores:
            word_overall = float(np.mean([w["overall_score"] for w in word_scores]))
        else:
            word_overall = 0.0
            
        # Phoneme-level aggregation
        if phoneme_scores:
            phoneme_overall = float(np.mean([p["pronunciation_score"] for p in phoneme_scores]))
        else:
            phoneme_overall = 0.0
            
        # Vowel Space Score (lower distance -> higher score)
        vowel_dists = []
        for v in vowel_data.get("user_space", []):
            dist = min(v["distance_male"], v["distance_female"])
            vowel_dists.append(dist)
            
        if vowel_dists:
            avg_dist = float(np.mean(vowel_dists))
            # Rough heuristic: distance > 300 is bad, 0 is perfect
            vowel_score = max(0, 100 - (avg_dist / 300.0) * 100)
        else:
            vowel_score = 0.0

        # Sub-scores
        pronunciation = (word_overall * 0.5) + (phoneme_overall * 0.5)
        intonation = intonation_data.get("similarity_score", 0.0)
        accent = accent_data.get("accent_score", 0.0)
        rhythm = accent_data.get("rhythm_score", 0.0)
        
        # Fluency based on pause ratio etc.
        # fluency_ratio is between 0 and 1 representing speech time vs total time
        fluency = fluency_ratio * 100
        
        # Final weighting
        overall = (
            pronunciation * 0.4 +
            intonation * 0.2 +
            accent * 0.1 +
            rhythm * 0.1 +
            fluency * 0.1 +
            vowel_score * 0.1
        )
        
        return {
            "word_score": round(word_overall, 2),
            "phoneme_score": round(phoneme_overall, 2),
            "pronunciation_score": round(pronunciation, 2),
            "intonation_score": round(intonation, 2),
            "accent_score": round(accent, 2),
            "rhythm_score": round(rhythm, 2),
            "vowel_score": round(vowel_score, 2),
            "fluency_score": round(fluency, 2),
            "overall_score": round(overall, 2)
        }
