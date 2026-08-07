import numpy as np
from typing import Dict, Any, List, Optional
import math
import parselmouth
from parselmouth.praat import call
from app.core.logger import logger

def classify_vowel(f1: float, f2: float) -> str:
    # Indonesian Vowel Approximation mapping based on F1/F2 (Hz)
    if f1 < 450 and f2 > 1800:
        return "i"
    if f1 < 500 and f2 < 1200:
        return "u"
    if f1 > 650 and f2 > 1100:
        return "a"
    if 450 <= f1 <= 650 and f2 > 1600:
        return "e"
    if 450 <= f1 <= 650 and f2 <= 1600:
        return "o"
    return "unknown"

class FormantAnalysisService:
    @staticmethod
    def analyze(audio_path: str) -> Dict[str, Any]:
        """
        Extracts F1 and F2 using Parselmouth.
        Filters out silence/consonants by only keeping clear formants.
        Clusters values by vowel and calculates representative coordinates.
        """
        try:
            import parselmouth
            sound = parselmouth.Sound(audio_path)
            
            # Extract intensity to ignore silence
            intensity = sound.to_intensity()
            formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
            
            duration = sound.get_total_duration()
            times = np.arange(0, duration, 0.01)

            vowel_clusters: Dict[str, List[Dict[str, float]]] = {
                "a": [], "i": [], "u": [], "e": [], "o": []
            }
            
            # Use intensity threshold (relative to max) to filter out silence
            int_vals = [intensity.get_value(t) for t in times]
            int_vals = [v for v in int_vals if not math.isnan(v)]
            min_intensity = np.max(int_vals) - 25 if int_vals else 50

            for t in times:
                try:
                    int_db = intensity.get_value(t)
                    if math.isnan(int_db) or int_db < min_intensity:
                        continue # Skip silence or unvoiced
                        
                    f1 = formant.get_value_at_time(1, t)
                    f2 = formant.get_value_at_time(2, t)
                    
                    if math.isnan(f1) or math.isnan(f2) or f1 <= 0 or f2 <= 0:
                        continue
                        
                    v = classify_vowel(f1, f2)
                    if v in vowel_clusters:
                        vowel_clusters[v].append({"f1": f1, "f2": f2})
                except Exception:
                    continue

            # Calculate representatives (median to avoid outlier skewing)
            vowel_space = []
            for v, points in vowel_clusters.items():
                if len(points) > 3: # Need at least a few frames to consider it a valid spoken vowel
                    f1_med = float(np.median([p["f1"] for p in points]))
                    f2_med = float(np.median([p["f2"] for p in points]))
                    vowel_space.append({
                        "vowel": v,
                        "f1": round(f1_med, 1),
                        "f2": round(f2_med, 1)
                    })

            return {
                "vowelSpace": vowel_space
            }

        except Exception as e:
            logger.error(f"Formant extraction skipped or error: {e}")
            return {"vowelSpace": []}
