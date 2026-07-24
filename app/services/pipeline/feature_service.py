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
        if len(energy_raw) > 0:
            silence_mask = energy_raw < (np.mean(energy_raw) * 0.5)
            pause_ratio = float(np.sum(silence_mask) / len(energy_raw))
        else:
            silence_mask = np.array([])
            pause_ratio = 0.0
        
        # Estimate average pause duration
        # Find continuous silence blocks
        if len(silence_mask) > 0:
            diff = np.diff(silence_mask.astype(int))
            pause_starts = np.where(diff == 1)[0]
            pause_ends = np.where(diff == -1)[0]
            
            # Match starts and ends
            if len(pause_starts) > 0 and len(pause_ends) > 0:
                if pause_ends[0] < pause_starts[0]:
                    pause_starts = np.insert(pause_starts, 0, 0)
                if pause_starts[-1] > pause_ends[-1]:
                    pause_ends = np.append(pause_ends, len(silence_mask) - 1)
                    
                pause_lengths = (pause_ends - pause_starts) * (duration / len(energy_raw))
                avg_pause = float(np.mean(pause_lengths)) if len(pause_lengths) > 0 else 0.0
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
        
        zcr_mean = float(np.mean(zcr))
        centroid_mean = float(np.mean(centroid))
        
        # Simple SNR estimation (Signal to Noise)
        if len(silence_mask) > 0:
            # Need to match audio length with silence_mask via repeating since silence_mask is frame-based
            frame_length = len(audio_22k) // len(silence_mask)
            audio_silence_mask = np.repeat(silence_mask, frame_length)
            # pad to match exact length
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
            "wpm": round(wpm, 1),
            "snr": round(snr, 1),
            "clarity_score": round(clarity_score * 100, 1),
            "clarity_explanation": clarity_exp,
            "energy_contour": [round(float(e), 3) for e in energy_raw] # Keep original for DB storage
        }

    @staticmethod
    def extract_native_similarity(audio_16k: np.ndarray, sr_16k: int, repo) -> Dict[str, Any]:
        """
        Extracts speech embeddings and computes native similarity.
        """
        user_emb = extract_embedding(audio_16k, sr_16k)
        
        male_cands = repo.get_top_k_candidates(user_emb, k=10, gender="male")
        female_cands = repo.get_top_k_candidates(user_emb, k=10, gender="female")
        
        male_sim = compute_embedding_score(user_emb, male_cands)
        female_sim = compute_embedding_score(user_emb, female_cands)
        
        overall_sim = max(male_sim, female_sim)
        ref_gender = "Female" if female_sim > male_sim else "Male"
        
        return {
            "similarity": round(overall_sim * 100, 1),
            "reference_gender": ref_gender,
            "confidence": round(min(100, (overall_sim + 0.1) * 100), 1),
            "user_embedding": user_emb,
            "male_score": male_sim,
            "female_score": female_sim
        }
