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
        
        if word_scores:
            word_overall = float(np.mean([w.get("overall_score", 0.0) for w in word_scores]))
            pronunciation = float(np.mean([w.get("pronunciation_score", 0.0) for w in word_scores]))
        else:
            word_overall = 0.0
            pronunciation = 0.0
            
        if phoneme_scores:
            phoneme_overall = float(np.mean([p.get("pronunciation_score", 0.0) for p in phoneme_scores]))
        else:
            phoneme_overall = 0.0
            
        vowel_dists = []
        for v in vowel_data.get("user_space", []):
            if v.get("distance_male") is not None and v.get("distance_female") is not None:
                dist = min(v["distance_male"], v["distance_female"])
                vowel_dists.append(dist)
            
        if vowel_dists:
            avg_dist = float(np.mean(vowel_dists))
            vowel_score = max(0.0, 100.0 - (avg_dist / 300.0) * 100.0)
        else:
            vowel_score = None

        intonation = intonation_data.get("similarity_score")
        if intonation is None:
            intonation = 0.0
            
        accent = accent_data.get("accent_score", 0.0)
        rhythm = accent_data.get("rhythm_score", 0.0)
        fluency = fluency_ratio * 100.0
        
        # Calculate final overall score from available components
        valid_components = []
        weights = []
        
        # Pronunciation carries heavy weight
        valid_components.append(pronunciation)
        weights.append(0.5)
        
        # Intonation
        valid_components.append(intonation)
        weights.append(0.2)
        
        # Fluency
        valid_components.append(fluency)
        weights.append(0.2)
        
        # Vowel space if available
        if vowel_score is not None:
            valid_components.append(vowel_score)
            weights.append(0.1)
        else:
            # Distribute vowel weight back to pronunciation
            weights[0] += 0.1
            
        overall = float(np.average(valid_components, weights=weights))
        
        return {
            "word_score": round(word_overall, 2),
            "phoneme_score": round(phoneme_overall, 2),
            "pronunciation_score": round(pronunciation, 2),
            "intonation_score": round(intonation, 2),
            "accent_score": round(accent, 2),
            "rhythm_score": round(rhythm, 2),
            "vowel_score": round(vowel_score, 2) if vowel_score is not None else 0.0,
            "fluency_score": round(fluency, 2),
            "overall_score": round(overall, 2)
        }
