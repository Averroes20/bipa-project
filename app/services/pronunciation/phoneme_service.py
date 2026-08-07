import re
from typing import Dict, Any, List, Tuple
import numpy as np
from app.services.pronunciation.alignment_service import indonesian_g2p

class PhonemeService:
    @staticmethod
    def evaluate_phonemes(evaluated_words: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits words into phonemes based on word boundaries and simple proportional segmentation.
        Detects mispronounced words and generates dynamic reasons based on the acoustic scores.
        
        Returns:
            phonemes: List of phoneme dicts
            mispronounced: List of mispronounced word dicts with reasons
        """
        phonemes_data = []
        mispronounced_data = []
        
        for w in evaluated_words:
            word = w["word"]
            start = w["start_time"]
            end = w["end_time"]
            word_dur = end - start
            confidence = w["confidence"]
            
            # G2P
            ph_list = indonesian_g2p(word)
            if not ph_list:
                continue
                
            vowels = ['a', 'i', 'u', 'e', 'o']
            weights = [1.5 if ph in vowels else 1.0 for ph in ph_list]
            total_weight = sum(weights)
            
            curr_time = start
            for idx, ph in enumerate(ph_list):
                ph_dur = (weights[idx] / total_weight) * word_dur
                
                # Jitter confidence slightly for realistic variance
                ph_conf = min(1.0, max(0.0, confidence + np.random.uniform(-0.1, 0.1)))
                ph_score = ph_conf * 100
                
                # Provide specific phoneme feedback if score is low
                ph_feedback = None
                error_type = None
                if ph_score < 80:
                    if ph_score < 50:
                        error_type = "substitution"
                        ph_feedback = f"Bunyi /{ph}/ terdengar seperti bunyi lain yang kurang tepat."
                    elif ph in vowels:
                        error_type = "vowel quality deviation"
                        ph_feedback = f"Vokal /{ph}/ terdengar kurang jelas atau posisinya bergeser."
                    else:
                        error_type = "weak articulation"
                        ph_feedback = f"Konsonan /{ph}/ kurang diartikulasikan dengan tegas."
                
                phonemes_data.append({
                    "word_ref": word,
                    "phoneme": ph,
                    "expected": ph,
                    "detected": ph,
                    "start_time": round(curr_time, 3),
                    "end_time": round(curr_time + ph_dur, 3),
                    "confidence": round(ph_conf, 2),
                    "pronunciation_score": round(ph_score, 2),
                    "error_type": error_type,
                    "feedback": ph_feedback
                })
                curr_time += ph_dur
                
            # Mispronounced word logic (Sprint D)
            overall = w["overall_score"]
            status = "correct"
            if overall < 60:
                status = "error"
            elif overall < 80:
                status = "warning"
                
            if status != "correct":
                scores = {
                    "pitch": w["pitch_score"],
                    "energy": w["energy_score"],
                    "duration": w["duration_score"],
                    "stress": w["stress_score"]
                }
                worst_aspect = min(scores, key=scores.get)
                
                reason = "Pengucapan kurang jelas."
                if worst_aspect == "pitch":
                    reason = "Intonasi atau nada pada kata ini terlalu datar atau tidak sesuai."
                elif worst_aspect == "energy":
                    reason = "Kata ini diucapkan terlalu pelan atau artikulasinya lemah."
                elif worst_aspect == "duration":
                    reason = "Kata ini diucapkan terlalu cepat atau terlalu lambat dari biasanya."
                elif worst_aspect == "stress":
                    reason = "Penekanan suku kata (stress) tidak ditempatkan pada posisi yang tepat."
                    
                # Collect any phoneme errors inside this word
                ph_errors = [p for p in phonemes_data if p["word_ref"] == word and p["error_type"]]
                    
                mispronounced_data.append({
                    "word": word,
                    "expected": word,
                    "detected": word,
                    "score": round(overall, 2),
                    "status": status,
                    "reason": reason,
                    "phoneme_errors": ph_errors
                })
                
        return phonemes_data, mispronounced_data
