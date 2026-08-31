import parselmouth
from parselmouth.praat import call
from typing import Dict, Any, List, Optional
import numpy as np
from app.core.logger import logger
from collections import defaultdict

class VowelService:
    VOWELS = ['a', 'i', 'u', 'e', 'o']
    
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
    def extract_and_compare(audio_path: str, phonemes_data: List[Dict], male_cand: Optional[Dict] = None, female_cand: Optional[Dict] = None) -> Dict[str, Any]:
        male_space = VowelService.NATIVE_BASELINES["Male"]
        female_space = VowelService.NATIVE_BASELINES["Female"]
        
        if male_cand and isinstance(male_cand.get("vowel_profile"), dict) and male_cand["vowel_profile"]:
            vp = male_cand["vowel_profile"]
            if "a" in vp and "f1" in vp["a"]:
                male_space = vp
                
        if female_cand and isinstance(female_cand.get("vowel_profile"), dict) and female_cand["vowel_profile"]:
            vp = female_cand["vowel_profile"]
            if "a" in vp and "f1" in vp["a"]:
                female_space = vp
        
        vowels_out = []
        ellipses = []
        
        try:
            snd = parselmouth.Sound(audio_path)
            formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
            
            raw_vowels = defaultdict(list)
            
            for ph in phonemes_data:
                symbol = ph.get("phoneme", "").lower()
                if symbol in VowelService.VOWELS:
                    start_time = ph["start_time"]
                    end_time = ph["end_time"]
                    midpoint = start_time + ((end_time - start_time) / 2.0)
                    
                    f1 = call(formants, "Get value at time", 1, midpoint, 'Hertz', 'Linear')
                    f2 = call(formants, "Get value at time", 2, midpoint, 'Hertz', 'Linear')
                    
                    if not np.isnan(f1) and not np.isnan(f2):
                        raw_vowels[symbol].append((f1, f2))
                        
            # Expected Vowels order
            for symbol in VowelService.VOWELS:
                b_m = male_space.get(symbol, VowelService.NATIVE_BASELINES["Male"][symbol])
                b_f = female_space.get(symbol, VowelService.NATIVE_BASELINES["Female"][symbol])
                
                if symbol in raw_vowels and len(raw_vowels[symbol]) > 0:
                    samples = np.array(raw_vowels[symbol])
                    centroid_f1 = np.mean(samples[:, 0])
                    centroid_f2 = np.mean(samples[:, 1])
                    
                    dist_m = np.sqrt((centroid_f1 - b_m["f1"])**2 + (centroid_f2 - b_m["f2"])**2)
                    dist_f = np.sqrt((centroid_f1 - b_f["f1"])**2 + (centroid_f2 - b_f["f2"])**2)
                    
                    dist = min(dist_m, dist_f)
                    
                    # Normalize distance: Max distance ~ 400Hz for a completely different vowel
                    match = max(0.0, min(100.0, 100.0 - (dist / 400.0) * 100.0))
                    
                    # Estimate radii for ellipse if we have enough samples, otherwise fixed default
                    if len(samples) >= 3:
                        r_f1 = np.std(samples[:, 0]) * 1.5
                        r_f2 = np.std(samples[:, 1]) * 1.5
                    else:
                        r_f1 = 40.0
                        r_f2 = 60.0
                        
                    vowels_out.append({
                        "phoneme": symbol,
                        "user": {"f1": round(float(centroid_f1), 1), "f2": round(float(centroid_f2), 1)},
                        "native_male": {"f1": b_m["f1"], "f2": b_m["f2"]},
                        "native_female": {"f1": b_f["f1"], "f2": b_f["f2"]},
                        "match": round(float(match))
                    })
                    
                    ellipses.append({
                        "phoneme": symbol,
                        "center": {"f1": round(float(centroid_f1), 1), "f2": round(float(centroid_f2), 1)},
                        "radius_f1": round(float(max(20.0, min(100.0, r_f1))), 1),
                        "radius_f2": round(float(max(30.0, min(150.0, r_f2))), 1)
                    })
                    
        except Exception as e:
            logger.error(f"Formant extraction error: {e}")
            
        return {
            "vowels": vowels_out,
            "ellipse": ellipses
        }
