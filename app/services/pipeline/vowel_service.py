import parselmouth
from parselmouth.praat import call
from typing import Dict, Any, List
import numpy as np
from app.core.logger import logger

class VowelAnalysisService:
    VOWELS = ['a', 'i', 'u', 'e', 'o']

    @staticmethod
    def extract_vowels(audio_path: str, phonemes_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts F1, F2, F3 for each vowel and calculates Vowel Space Area.
        """
        try:
            snd = parselmouth.Sound(audio_path)
            # max_formant = 5500 for females, 5000 for males, we use 5500 as safe default
            formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
            
            vowel_data = []
            f1_list = []
            f2_list = []
            f3_list = []
            
            for ph in phonemes_data:
                symbol = ph["symbol"].lower()
                if symbol in VowelAnalysisService.VOWELS:
                    # Get the midpoint of the vowel duration
                    midpoint = ph["start"] + (ph["duration"] / 2.0)
                    
                    f1 = call(formants, "Get value at time", 1, midpoint, 'Hertz', 'Linear')
                    f2 = call(formants, "Get value at time", 2, midpoint, 'Hertz', 'Linear')
                    f3 = call(formants, "Get value at time", 3, midpoint, 'Hertz', 'Linear')
                    
                    if not np.isnan(f1) and not np.isnan(f2):
                        vowel_data.append({
                            "vowel": symbol,
                            "f1": round(f1, 2),
                            "f2": round(f2, 2),
                            "f3": round(f3, 2) if not np.isnan(f3) else 0.0,
                            "time": round(midpoint, 3)
                        })
                        f1_list.append(f1)
                        f2_list.append(f2)
                        if not np.isnan(f3):
                            f3_list.append(f3)
            
            # Calculate average F1, F2, F3
            avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
            avg_f2 = float(np.mean(f2_list)) if f2_list else 0.0
            avg_f3 = float(np.mean(f3_list)) if f3_list else 0.0
            
            # Calculate Vowel Space Area (VSA) based on convex hull or triangle area of a, i, u
            vsa = 0.0
            a_vowels = [v for v in vowel_data if v["vowel"] == 'a']
            i_vowels = [v for v in vowel_data if v["vowel"] == 'i']
            u_vowels = [v for v in vowel_data if v["vowel"] == 'u']
            
            if a_vowels and i_vowels and u_vowels:
                # Get mean coords
                a_f1, a_f2 = np.mean([v["f1"] for v in a_vowels]), np.mean([v["f2"] for v in a_vowels])
                i_f1, i_f2 = np.mean([v["f1"] for v in i_vowels]), np.mean([v["f2"] for v in i_vowels])
                u_f1, u_f2 = np.mean([v["f1"] for v in u_vowels]), np.mean([v["f2"] for v in u_vowels])
                
                # Area of triangle: 0.5 * |xA(yB - yC) + xB(yC - yA) + xC(yA - yB)| where x=F1, y=F2
                vsa = 0.5 * abs(a_f1*(i_f2 - u_f2) + i_f1*(u_f2 - a_f2) + u_f1*(a_f2 - i_f2))
                
            return {
                "vowels": vowel_data,
                "f1_mean": round(avg_f1, 2),
                "f2_mean": round(avg_f2, 2),
                "f3_mean": round(avg_f3, 2),
                "vsa": round(vsa, 2)
            }
        except Exception as e:
            logger.error(f"Formant extraction error: {e}")
            return {
                "vowels": [],
                "f1_mean": 0.0,
                "f2_mean": 0.0,
                "f3_mean": 0.0,
                "vsa": 0.0
            }
