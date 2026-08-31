import numpy as np
from typing import Dict, Any, List
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class IntonationService:
    @staticmethod
    def _evaluate_single(user_np: np.ndarray, native_contour: List[float]) -> tuple[float, float, float, np.ndarray]:
        if not native_contour or len(native_contour) < 2:
            return 0.0, 0.0, 0.0, np.array([])
            
        n_np = np.array(native_contour).flatten()
        # Time-normalize native contour to match user length for charting and correlation
        if len(user_np) > 1:
            n_interp = np.interp(np.linspace(0, 1, len(user_np)), np.linspace(0, 1, len(n_np)), n_np)
        else:
            n_interp = n_np

        distance, _ = fastdtw(user_np.reshape(-1, 1), n_interp.reshape(-1, 1), dist=euclidean)
        
        if len(user_np) > 1 and len(n_interp) > 1:
            corr_matrix = np.corrcoef(user_np, n_interp)
            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            correlation = 0.0
            
        # Slope diff
        def get_slope(y):
            if len(y) < 2: return 0.0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
            
        u_slope = float(get_slope(user_np))
        n_slope = float(get_slope(n_interp))
        slope_diff = abs(u_slope - n_slope)
        
        max_dist = 150.0 * len(user_np)
        dtw_score = max(0, 100 - (float(distance) / max(1, max_dist)) * 100)
        corr_score = max(0, float(correlation) * 100)
        similarity = (dtw_score * 0.4) + (corr_score * 0.4) + (max(0, 100 - slope_diff*50) * 0.2)
        
        return float(similarity), float(distance), float(correlation), n_interp

    @staticmethod
    def analyze_pitch_pattern(user_contour: List[float], native_contour: List[float]) -> Dict[str, Any]:
        if not user_contour or not native_contour:
            return {}
            
        u_np = np.array(user_contour)
        n_np = np.array(native_contour)
        
        if len(u_np) < 5 or len(n_np) < 5:
            return {}
            
        # Terminal drop is difference between last 10% and the previous 20%
        # Or simpler: just the difference between start and end of the contour
        u_drop = u_np[-1] - u_np[0]
        n_drop = n_np[-1] - n_np[0]
        
        diff = abs(u_drop - n_drop)
        
        if u_drop < -20:
            pattern = "falling"
        elif u_drop > 20:
            pattern = "rising"
        elif abs(u_drop) <= 20 and abs(n_drop) <= 20:
            pattern = "flat"
        else:
            pattern = "overshoot"
            
        if diff < 10:
            summary = f"Kalimat Anda berakhir dengan pola {pattern} yang sangat mendekati native speaker."
            rec = "Pertahankan intonasi ini pada kalimat serupa."
        elif diff <= 25:
            summary = f"Pola {pattern} Anda cukup baik, namun sedikit berbeda tingkatannya dengan native."
            rec = "Perhatikan seberapa tajam nada Anda di akhir kalimat."
        else:
            summary = f"Pola intonasi Anda jauh berbeda dari native speaker."
            rec = "Cobalah untuk lebih mendengarkan dan meniru pergerakan nada di akhir kalimat."

        return {
            "pattern": pattern,
            "terminal_drop_hz": round(abs(u_drop), 1),
            "native_drop_hz": round(abs(n_drop), 1),
            "difference_hz": round(diff, 1),
            "summary": summary,
            "recommendation": rec
        }

    @staticmethod
    def compare_contours(user_contour: List[float], male_contour: List[float], female_contour: List[float]) -> Dict[str, Any]:
        """
        Compares two pitch contours using DTW, Correlation, and Movement.
        """
        if not user_contour:
            return {
                "user_contour": [],
                "male_contour": [],
                "female_contour": [],
                "male_similarity": 0.0,
                "female_similarity": 0.0,
                "similarity_score": 0.0,
                "dtw_distance": 0.0,
                "correlation": 0.0,
                "preferred_reference": "Unknown",
                "pattern": "neutral",
                "sentence_ending": "neutral",
                "pitch_variance": 0.0,
                "pitch_insight": {}
            }
            
        u_np = np.array(user_contour).flatten()
        
        m_sim, m_dist, m_corr, m_interp = IntonationService._evaluate_single(u_np, male_contour)
        f_sim, f_dist, f_corr, f_interp = IntonationService._evaluate_single(u_np, female_contour)
        
        pref = "Female" if f_sim > m_sim else "Male"
        sim_score = max(m_sim, f_sim)
        dtw = f_dist if pref == "Female" else m_dist
        corr = f_corr if pref == "Female" else m_corr
        
        # Simple pattern extraction
        variance = float(np.var(u_np))
        
        # Sentence ending (last 20% vs middle)
        idx_80 = int(len(u_np) * 0.8)
        if idx_80 > 0 and len(u_np) - idx_80 > 1:
            mid_mean = np.mean(u_np[:idx_80])
            end_mean = np.mean(u_np[idx_80:])
            if end_mean > mid_mean + 10:
                ending = "rising"
            elif end_mean < mid_mean - 10:
                ending = "falling"
            else:
                ending = "neutral"
        else:
            ending = "neutral"
            
        if variance > 500:
            pattern = "expressive"
        elif variance < 100:
            pattern = "monotone"
        else:
            pattern = "natural"
            
        native_contour = f_interp if pref == "Female" else m_interp
        pitch_insight = IntonationService.analyze_pitch_pattern(u_np.tolist(), native_contour.tolist())

        return {
            "user_contour": [float(x) for x in u_np],
            "male_contour": [float(x) for x in m_interp] if len(m_interp) > 0 else [],
            "female_contour": [float(x) for x in f_interp] if len(f_interp) > 0 else [],
            "male_similarity": round(m_sim, 2),
            "female_similarity": round(f_sim, 2),
            "similarity_score": round(sim_score, 2),
            "dtw_distance": round(dtw, 2),
            "correlation": round(corr, 2),
            "preferred_reference": pref,
            "pattern": pattern,
            "sentence_ending": ending,
            "pitch_variance": round(variance, 2),
            "pitch_insight": pitch_insight
        }
