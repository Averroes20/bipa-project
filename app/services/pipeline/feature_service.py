import numpy as np
import librosa
from typing import Dict, Any, List
from app.audio_engine.embeddings import extract_embedding, compute_embedding_score

class FeatureExtractionService:
    @staticmethod
    def extract_prosody_and_clarity(audio_22k: np.ndarray, sr_22k: int, words_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts fluency (pauses, speech rate) and clarity (ZCR, SNR, centroid) metrics.
        """
        duration = float(librosa.get_duration(y=audio_22k, sr=sr_22k))
        
        # Energy and Pauses
        energy_raw = librosa.feature.rms(y=audio_22k)[0]
        pause_timeline = []
        if len(energy_raw) > 0:
            silence_mask = energy_raw < (np.mean(energy_raw) * 0.5)
            pause_ratio = float(np.sum(silence_mask) / len(energy_raw))
        else:
            silence_mask = np.array([])
            pause_ratio = 0.0
        
        # Estimate average pause duration and build timeline
        if len(silence_mask) > 0:
            diff = np.diff(silence_mask.astype(int))
            pause_starts = np.where(diff == 1)[0]
            pause_ends = np.where(diff == -1)[0]
            
            # Match starts and ends
            if len(pause_starts) > 0 or len(pause_ends) > 0:
                if len(pause_ends) > 0 and (len(pause_starts) == 0 or pause_ends[0] < pause_starts[0]):
                    pause_starts = np.insert(pause_starts, 0, 0)
                if len(pause_starts) > 0 and (len(pause_ends) == 0 or pause_starts[-1] > pause_ends[-1]):
                    pause_ends = np.append(pause_ends, len(silence_mask) - 1)
                    
                pause_lengths = (pause_ends - pause_starts) * (duration / len(energy_raw))
                avg_pause = float(np.mean(pause_lengths)) if len(pause_lengths) > 0 else 0.0
                
                for s, e in zip(pause_starts, pause_ends):
                    start_time = float(s * (duration / len(energy_raw)))
                    end_time = float(e * (duration / len(energy_raw)))
                    pause_timeline.append({
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(end_time - start_time, 3)
                    })
            else:
                avg_pause = 0.0
        else:
            avg_pause = 0.0

        # Fluency / Speech Rate
        word_count = len([w for w in words_data if w["word"]])
        speech_rate = word_count / duration if duration > 0 else 0
        wpm = speech_rate * 60
        
        # Clarity Analysis
        zcr = librosa.feature.zero_crossing_rate(audio_22k)[0]
        centroid = librosa.feature.spectral_centroid(y=audio_22k, sr=sr_22k)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_22k, sr=sr_22k)[0]
        try:
            contrast = librosa.feature.spectral_contrast(y=audio_22k, sr=sr_22k)
            contrast_mean = float(np.mean(contrast))
        except Exception:
            contrast_mean = 0.0
            
        zcr_mean = float(np.mean(zcr))
        centroid_mean = float(np.mean(centroid))
        bandwidth_mean = float(np.mean(bandwidth))
        
        # Simple SNR estimation (Signal to Noise)
        if len(silence_mask) > 0:
            frame_length = len(audio_22k) // len(silence_mask)
            audio_silence_mask = np.repeat(silence_mask, frame_length)
            if len(audio_silence_mask) < len(audio_22k):
                audio_silence_mask = np.append(audio_silence_mask, [False] * (len(audio_22k) - len(audio_silence_mask)))
            else:
                audio_silence_mask = audio_silence_mask[:len(audio_22k)]
                
            signal_power = np.mean(audio_22k[~audio_silence_mask]**2) if np.any(~audio_silence_mask) else 1e-10
            noise_power = np.mean(audio_22k[audio_silence_mask]**2) if np.any(audio_silence_mask) else 1e-10
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 30.0
        else:
            snr = 30.0
            
        clarity_score = ((centroid_mean / 4000.0) * 0.5) + ((zcr_mean * 10.0) * 0.3) + (min(snr, 30) / 30 * 0.2)
        clarity_score = max(0.0, min(1.0, clarity_score))
        
        if clarity_score > 0.75:
            clarity_exp = "High signal-to-noise ratio with clear spectral energy."
        elif clarity_score > 0.5:
            clarity_exp = "Adequate voice stability, but some frequencies are muffled."
        else:
            clarity_exp = "Low clarity due to background noise or poor articulation."

        return {
            "duration": round(duration, 2),
            "pause_ratio": round(pause_ratio, 2),
            "avg_pause": round(avg_pause, 2),
            "pause_timeline": pause_timeline,
            "wpm": round(wpm, 1),
            "snr": round(snr, 1),
            "zcr_mean": round(zcr_mean, 4),
            "centroid_mean": round(centroid_mean, 2),
            "bandwidth_mean": round(bandwidth_mean, 2),
            "contrast_mean": round(contrast_mean, 2),
            "clarity_score": round(clarity_score * 100, 1),
            "clarity_explanation": clarity_exp,
            "energy_contour": [round(float(e), 3) for e in energy_raw]
        }

    @staticmethod
    def extract_native_similarity(audio_16k: np.ndarray, sr_16k: int, repo) -> Dict[str, Any]:
        """
        Extracts speech embeddings and computes native similarity.
        Also returns the best candidate contours for DTW scoring.
        """
        user_emb = extract_embedding(audio_16k, sr_16k)
        
        male_cands = repo.get_top_k_candidates(user_emb, k=10, gender="male")
        female_cands = repo.get_top_k_candidates(user_emb, k=10, gender="female")
        
        # Original candidates had .embedding_vector, now they have dictionaries
        # Wait, compute_embedding_score expects a list of objects that have embeddings?
        # Let's check compute_embedding_score. It probably expects the pgvector string or array.
        
        male_sim, best_male = FeatureExtractionService._compute_and_get_best(user_emb, male_cands)
        female_sim, best_female = FeatureExtractionService._compute_and_get_best(user_emb, female_cands)
        
        overall_sim = max(male_sim, female_sim)
        ref_gender = "Female" if female_sim > male_sim else "Male"
        
        return {
            "similarity": round(overall_sim * 100, 1),
            "reference_gender": ref_gender,
            "confidence": round(min(100, (overall_sim + 0.1) * 100), 1),
            "user_embedding": user_emb,
            "male_score": male_sim,
            "female_score": female_sim,
            "best_male_candidate": best_male,
            "best_female_candidate": best_female
        }

    @staticmethod
    def _compute_and_get_best(user_emb: np.ndarray, candidates: List[Dict]) -> tuple[float, Dict]:
        from numpy.linalg import norm
        best_score = 0.0
        best_cand = None
        for c in candidates:
            # emb could be string from postgres vector type or list
            cand_emb = c.get("embedding_vector")
            if isinstance(cand_emb, str):
                try:
                    import ast
                    cand_emb = ast.literal_eval(cand_emb)
                except Exception:
                    pass
            if cand_emb is not None:
                cand_vec = np.array(cand_emb)
                score = np.dot(user_emb, cand_vec) / (norm(user_emb) * norm(cand_vec) + 1e-10)
                if score > best_score:
                    best_score = score
                    best_cand = c
        return float(best_score), best_cand
