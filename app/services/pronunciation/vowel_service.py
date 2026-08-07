import parselmouth
from parselmouth.praat import call
from typing import Dict, Any, List
import numpy as np
from app.core.logger import logger

class VowelService:
    VOWELS = ['a', 'i', 'u', 'e', 'o']
    
    # Static baseline formants for Indonesian vowels (Rough approximations for Male/Female)
    NATIVE_BASELINES = {
        "Male": {
            "a": {"f1": 700, "f2": 1200},
            "i": {"f1": 300, "f2": 2200},
            "u": {"f1": 300, "f2": 800},
            "e": {"f1": 400, "f2": 1900},
            "o": {"f1": 400, "f2": 900}
        },
        "Female": {
            "a": {"f1": 850, "f2": 1400},
            "i": {"f1": 350, "f2": 2700},
            "u": {"f1": 350, "f2": 900},
            "e": {"f1": 500, "f2": 2300},
            "o": {"f1": 500, "f2": 1000}
        }
    }

    @staticmethod
    def extract_and_compare(audio_path: str, phonemes_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts F1, F2, F3 for each vowel and compares to native baselines using Euclidean distance.
        """
        try:
            snd = parselmouth.Sound(audio_path)
            formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
            
            user_vowels = []
            
            for ph in phonemes_data:
                symbol = ph["phoneme"].lower()
                if symbol in VowelService.VOWELS:
                    midpoint = ph["start_time"] + ((ph["end_time"] - ph["start_time"]) / 2.0)
                    
                    f1 = call(formants, "Get value at time", 1, midpoint, 'Hertz', 'Linear')
                    f2 = call(formants, "Get value at time", 2, midpoint, 'Hertz', 'Linear')
                    
                    if not np.isnan(f1) and not np.isnan(f2):
                        # Compare distance to male and female baseline
                        b_m = VowelService.NATIVE_BASELINES["Male"][symbol]
                        b_f = VowelService.NATIVE_BASELINES["Female"][symbol]
                        
                        dist_m = np.sqrt((f1 - b_m["f1"])**2 + (f2 - b_m["f2"])**2)
                        dist_f = np.sqrt((f1 - b_f["f1"])**2 + (f2 - b_f["f2"])**2)
                        
                        dist = min(dist_m, dist_f)
                        accuracy = max(0.0, 100.0 - (dist / 400.0) * 100.0)
                        
                        user_vowels.append({
                            "vowel": symbol,
                            "f1": round(f1, 2),
                            "f2": round(f2, 2),
                            "distance_male": round(dist_m, 2),
                            "distance_female": round(dist_f, 2),
                            "accuracy": round(accuracy, 2)
                        })
            
            return {
                "user_space": user_vowels,
                "native_male_space": VowelService.NATIVE_BASELINES["Male"],
                "native_female_space": VowelService.NATIVE_BASELINES["Female"]
            }
            
        except Exception as e:
            logger.error(f"Formant extraction error: {e}")
            return {
                "user_space": [],
                "native_male_space": VowelService.NATIVE_BASELINES["Male"],
                "native_female_space": VowelService.NATIVE_BASELINES["Female"]
            }
