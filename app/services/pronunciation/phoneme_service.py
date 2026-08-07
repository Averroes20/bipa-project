import re
from typing import Dict, Any, List, Tuple
from app.services.pronunciation.alignment_service import indonesian_g2p

class PhonemeService:
    @staticmethod
    def evaluate_phonemes(evaluated_words: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        phonemes_data = []
        mispronounced_data = []
        
        for w in evaluated_words:
            word = w["word"]
            start = w["start_time"]
            end = w["end_time"]
            confidence = w["confidence"]
            
            # Missing word mapping
            if start == 0.0 and end == 0.0 and confidence == 0.0:
                w["pronunciation_score"] = 0.0
                w["overall_score"] = 0.0
                mispronounced_data.append({
                    "word": word,
                    "status": "error",
                    "reason": "Kata terlewatkan atau tidak terdeteksi (deletion)."
                })
                continue
                
            word_dur = end - start
            
            ph_list = indonesian_g2p(word)
            if not ph_list:
                w["pronunciation_score"] = 0.0
                w["overall_score"] = 0.0
                continue
                
            vowels = ['a', 'i', 'u', 'e', 'o']
            weights = [1.5 if ph in vowels else 1.0 for ph in ph_list]
            total_weight = sum(weights)
            
            curr_time = start
            word_phoneme_scores = []
            
            for idx, ph in enumerate(ph_list):
                ph_dur = (weights[idx] / total_weight) * word_dur
                
                # Phoneme score based purely on alignment confidence
                ph_score = min(100.0, max(0.0, confidence * 100))
                word_phoneme_scores.append(ph_score)
                
                ph_feedback = None
                error_type = None
                
                if ph_score < 75:
                    if ph_score < 40:
                        error_type = "deletion"
                        ph_feedback = f"Bunyi /{ph}/ hilang atau sangat tidak jelas."
                    elif ph_score < 60:
                        error_type = "substitution"
                        ph_feedback = f"Bunyi /{ph}/ terdengar seperti bunyi lain yang kurang tepat."
                    elif ph in vowels:
                        error_type = "vowel_quality_deviation"
                        ph_feedback = f"Vokal /{ph}/ terdengar kurang jelas atau posisinya bergeser."
                    else:
                        error_type = "weak_articulation"
                        ph_feedback = f"Konsonan /{ph}/ kurang diartikulasikan dengan tegas."
                
                phonemes_data.append({
                    "word_ref": word,
                    "phoneme": ph,
                    "expected": ph,
                    "detected": ph if error_type != "deletion" else None,
                    "start_time": round(curr_time, 3),
                    "end_time": round(curr_time + ph_dur, 3),
                    "confidence": round(confidence, 2),
                    "pronunciation_score": round(ph_score, 2),
                    "error_type": error_type,
                    "feedback": ph_feedback
                })
                curr_time += ph_dur
                
            # Hierarchical Scoring Logic
            pronunciation_score = sum(word_phoneme_scores) / len(word_phoneme_scores) if word_phoneme_scores else 0.0
            
            # Overall Word Score = 60% Pronunciation + 40% Acoustic
            overall = (pronunciation_score * 0.6) + (w["pitch_score"] * 0.15) + (w["energy_score"] * 0.15) + (w["duration_score"] * 0.1)
            
            w["pronunciation_score"] = round(pronunciation_score, 2)
            w["overall_score"] = round(overall, 2)
            
            status = "correct"
            if overall < 60:
                status = "error"
            elif overall < 80:
                status = "needs_improvement"
                
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
                    
                ph_errors = [p for p in phonemes_data if p["word_ref"] == word and p["error_type"]]
                if ph_errors:
                    ph_reason = ", ".join([f"/{p['phoneme']}/ ({p['error_type']})" for p in ph_errors])
                    reason += f" Kesalahan fonem: {ph_reason}."
                    
                mispronounced_data.append({
                    "word": word,
                    "status": status,
                    "reason": reason,
                    "score": w.get("overall_score", 0.0)
                })
                
        return phonemes_data, mispronounced_data
