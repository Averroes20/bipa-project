import Levenshtein
import re
from typing import Dict, Any, List
from app.services.pipeline.alignment_service import indonesian_g2p

class PronunciationService:
    @staticmethod
    def detect_errors(target_text: str, transcribed_words: List[Dict], phonemes_data: List[Dict] = None) -> Dict[str, List[Dict]]:
        if not target_text:
            return {"word_errors": [], "phoneme_errors": []}
            
        target_words = [re.sub(r'[^a-z]', '', w.lower()) for w in target_text.split()]
        target_words = [w for w in target_words if w]
        
        spoken_words = [w["word"] for w in transcribed_words if w.get("word")]
        
        word_errors = []
        ops = Levenshtein.editops(target_words, spoken_words)
        
        for op, tgt_idx, spk_idx in ops:
            if op == "replace":
                word_errors.append({
                    "type": "Substitution",
                    "expected": target_words[tgt_idx],
                    "detected": spoken_words[spk_idx],
                    "confidence": round(transcribed_words[spk_idx].get("confidence", 0.5) * 100, 1)
                })
            elif op == "delete":
                word_errors.append({
                    "type": "Deletion",
                    "expected": target_words[tgt_idx],
                    "detected": "",
                    "confidence": 100.0
                })
            elif op == "insert":
                word_errors.append({
                    "type": "Insertion",
                    "expected": "",
                    "detected": spoken_words[spk_idx],
                    "confidence": round(transcribed_words[spk_idx].get("confidence", 0.5) * 100, 1)
                })

        phoneme_errors = []
        if phonemes_data:
            target_phonemes = []
            for w in target_words:
                target_phonemes.extend(indonesian_g2p(w))
            
            spoken_phonemes = [p["symbol"] for p in phonemes_data]
            
            ph_ops = Levenshtein.editops(target_phonemes, spoken_phonemes)
            
            # Map all target phonemes first as correct
            ph_results = [{"expected": ph, "detected": ph, "is_correct": True, "confidence": 100.0} for ph in target_phonemes]
            
            for op, tgt_idx, spk_idx in ph_ops:
                if op == "replace":
                    ph_results[tgt_idx] = {
                        "expected": target_phonemes[tgt_idx],
                        "detected": spoken_phonemes[spk_idx],
                        "is_correct": False,
                        "confidence": round(phonemes_data[spk_idx].get("confidence", 0.5) * 100, 1)
                    }
                elif op == "delete":
                    ph_results[tgt_idx] = {
                        "expected": target_phonemes[tgt_idx],
                        "detected": "",
                        "is_correct": False,
                        "confidence": 100.0
                    }
                elif op == "insert":
                    # For insertions, there is no expected phoneme.
                    # We just log it.
                    phoneme_errors.append({
                        "expected": "",
                        "detected": spoken_phonemes[spk_idx],
                        "is_correct": False,
                        "confidence": round(phonemes_data[spk_idx].get("confidence", 0.5) * 100, 1)
                    })
            phoneme_errors.extend(ph_results)

        word_acc = sum(w["confidence"] * 100 for w in transcribed_words) / max(1, len(transcribed_words))
        phoneme_acc = 100.0
        if phonemes_data:
            phoneme_acc = sum(p["confidence"] * 100 for p in phonemes_data) / max(1, len(phonemes_data))
            
        pronunciation_score = (word_acc * 0.4) + (phoneme_acc * 0.6)

        return {
            "pronunciation_score": round(pronunciation_score, 1),
            "word_score": round(word_acc, 1),
            "phoneme_score": round(phoneme_acc, 1),
            "word_errors": word_errors,
            "phoneme_errors": phoneme_errors
        }

class ScoringService:
    @staticmethod
    def calculate_scores(
        words_data: List[Dict],
        phonemes_data: List[Dict],
        native_sim: Dict,
        features: Dict,
        pitch_stats: Dict
    ) -> Dict[str, Any]:
        """
        Calculates all required scores with metadata, including DTW over contours.
        """
        from app.audio_engine.comparison import dtw_distance
        
        # Pronunciation Score (40% Phoneme, 30% Word, 20% Align Conf, 10% Acoustic Sim)
        ph_scores = [p["confidence"] * 100 for p in phonemes_data]
        ph_acc = sum(ph_scores) / len(ph_scores) if ph_scores else 50.0
        
        word_confs = [w["confidence"] * 100 for w in words_data]
        word_acc = sum(word_confs) / len(word_confs) if word_confs else 50.0
        
        align_conf = sum(word_confs) / len(word_confs) if word_confs else 50.0
        
        ac_sim = native_sim.get("similarity", 50.0)
        
        pron_score = (ph_acc * 0.4) + (word_acc * 0.3) + (align_conf * 0.2) + (ac_sim * 0.1)
        
        # Fluency Score (based on WPM and pauses)
        wpm = features.get("wpm", 0)
        pause_ratio = features.get("pause_ratio", 0)
        fluency_score = max(0, min(100, 100 - (pause_ratio * 100) + min(20, (wpm / 150) * 20)))
        
        # DTW Score computation
        ref_gender = native_sim.get("reference_gender", "Male").lower()
        best_cand = native_sim.get(f"best_{ref_gender}_candidate")
        
        dtw_pitch_score = 100.0
        dtw_energy_score = 100.0
        if best_cand:
            ref_pitch = best_cand.get("pitch_contour", [])
            ref_energy = best_cand.get("energy_contour", [])
            user_pitch = pitch_stats.get("contour", [])
            user_energy = features.get("energy_contour", [])
            
            p_dist = dtw_distance(user_pitch, ref_pitch) if user_pitch and ref_pitch else 0
            e_dist = dtw_distance(user_energy, ref_energy) if user_energy and ref_energy else 0
            
            # Convert distances to scores (0-100)
            dtw_pitch_score = max(0, 100 - (p_dist / max(1, len(user_pitch)) * 0.5))
            dtw_energy_score = max(0, 100 - (e_dist / max(1, len(user_energy)) * 50.0))
            
        dtw_overall = (dtw_pitch_score * 0.6) + (dtw_energy_score * 0.4)
        
        # Intonation Score (based on Pitch stats, Native similarity, and DTW score)
        pitch_variance = pitch_stats.get("variance", 0)
        intonation_score = max(0, min(100, (ac_sim * 0.4) + (dtw_overall * 0.4) + min(20, pitch_variance / 50)))

        
        # Clarity Score
        clarity_score = features.get("clarity_score", 50.0)
        
        overall = (pron_score * 0.4) + (fluency_score * 0.25) + (intonation_score * 0.2) + (clarity_score * 0.15)
        
        return {
            "overallScore": round(overall, 1),
            "pronunciation": round(pron_score, 1),
            "fluency": round(fluency_score, 1),
            "intonation": round(intonation_score, 1),
            "clarity": round(clarity_score, 1),
            "nativeSimilarity": round(ac_sim, 1),
            "dtw_score": round(dtw_overall, 1),
            "analysisMetadata": {
                "pronunciation_basis": {
                    "Phoneme Accuracy": f"{round(ph_acc, 1)}%",
                    "Word Accuracy": f"{round(word_acc, 1)}%",
                    "Alignment Confidence": f"{round(align_conf, 1)}%",
                    "Acoustic Similarity": f"{round(ac_sim, 1)}%"
                },
                "fluency_basis": {
                    "Speech Rate": f"{features.get('wpm', 0)} WPM",
                    "Pause Duration Avg": f"{features.get('avg_pause', 0)}s",
                    "Pause Ratio": f"{round(features.get('pause_ratio', 0) * 100, 1)}%"
                },
                "dtw_basis": {
                    "Pitch Match": f"{round(dtw_pitch_score, 1)}%",
                    "Energy Match": f"{round(dtw_energy_score, 1)}%"
                }
            }
        }
