import json
from collections import defaultdict
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.audio_models import AudioAnalysis
from app.repositories.analysis_repository import AnalysisRepository

class ComparisonAnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AnalysisRepository(db)

    def get_comparison_analytics(self, user_id: str) -> Dict[str, Any]:
        analyses = self.db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id).order_by(AudioAnalysis.created_at.asc()).all()
        
        # Native baseline defaults (since they are generated randomly or fixed if real dataset not present)
        # Assuming the new pipeline sets these properly, we'll mock realistic native targets.
        native_male = {"pronunciation": 95, "fluency": 92, "intonation": 90, "clarity": 94}
        native_female = {"pronunciation": 96, "fluency": 94, "intonation": 93, "clarity": 96}
        
        if not analyses:
             return self._empty_response(native_male, native_female)

        now = datetime.utcnow()
        seven_days_ago = now - timedelta(days=7)
        
        user_scores = {"pronunciation": [], "fluency": [], "intonation": [], "clarity": [], "similarity": []}
        recent_fluency = []
        old_fluency = []
        
        errors_tally = defaultdict(int)
        phoneme_errors_tally = defaultdict(int)
        vowel_errors_tally = defaultdict(int)
        
        wpm_total = []
        pause_duration_total = []
        
        for record in analyses:
            detail = {}
            try:
                if record.analysis_detail:
                    detail = json.loads(record.analysis_detail)
            except Exception:
                continue
                
            dims = detail.get("dimensions", {})
            user_scores["pronunciation"].append(dims.get("pronunciation", 0))
            user_scores["fluency"].append(dims.get("fluency", 0))
            user_scores["intonation"].append(dims.get("intonation", 0))
            user_scores["clarity"].append(dims.get("clarity", 0))
            user_scores["similarity"].append(dims.get("accent", 0))
            
            is_recent_week = record.created_at >= seven_days_ago if record.created_at else False
            is_prev_week = (seven_days_ago > record.created_at >= (seven_days_ago - timedelta(days=7))) if record.created_at else False
            
            if is_recent_week:
                recent_fluency.append(dims.get("fluency", 0))
            elif is_prev_week:
                old_fluency.append(dims.get("fluency", 0))
                
            # Speaking stats
            features = detail.get("analysisMetadata", {}).get("fluency_basis", {})
            wpm_val = str(features.get("Speech Rate", "0")).replace(" WPM", "")
            try:
                wpm_total.append(float(wpm_val))
            except:
                pass
                
            pause_val = str(features.get("Pause Duration Avg", "0")).replace("s", "")
            try:
                pause_duration_total.append(float(pause_val))
            except:
                pass
                
            # Errors for phonemes / words
            for err in detail.get("errors", []):
                word = str(err.get("word", "")).lower()
                if word:
                    errors_tally[word] += 1
            
            for ph in detail.get("pronunciation", {}).get("phonemes", []):
                ph_score = ph.get("score", 100)
                if ph_score < 75:
                    symbol = str(ph.get("symbol", "")).lower()
                    if symbol in ["a", "i", "u", "e", "o"]:
                        vowel_errors_tally[symbol] += 1
                    elif symbol:
                        phoneme_errors_tally[symbol] += 1
                             
        avg = lambda x: sum(x)/len(x) if x else 0
        
        user_avg = {
            "pronunciation": avg(user_scores["pronunciation"]),
            "fluency": avg(user_scores["fluency"]),
            "intonation": avg(user_scores["intonation"]),
            "clarity": avg(user_scores["clarity"]),
            "similarity": avg(user_scores["similarity"])
        }
        
        # Radar Data
        radar_data = [
            {"dimension": "Pronunciation", "You": round(user_avg["pronunciation"], 1), "Native Reference": 95},
            {"dimension": "Fluency", "You": round(user_avg["fluency"], 1), "Native Reference": 93},
            {"dimension": "Intonation", "You": round(user_avg["intonation"], 1), "Native Reference": 91},
            {"dimension": "Clarity", "You": round(user_avg["clarity"], 1), "Native Reference": 95},
            {"dimension": "Similarity", "You": round(user_avg["similarity"], 1), "Native Reference": 100},
        ]
        
        # Word / Phoneme lists
        top_words = [{"word": k, "frequency": v, "accuracy": max(0, 100 - v*5)} for k, v in sorted(errors_tally.items(), key=lambda i: i[1], reverse=True)[:5]]
        top_diff_phonemes = [{"phoneme": k, "frequency": v, "accuracy": max(0, 100 - v*10)} for k, v in sorted(phoneme_errors_tally.items(), key=lambda i: i[1], reverse=True)[:5]]
        
        # Vowels
        vowel_stats = [{"vowel": k, "frequency": v, "deviation": min(100, v*15)} for k, v in sorted(vowel_errors_tally.items(), key=lambda i: i[1], reverse=True)[:5]]
        
        # Speaking Stats
        speaking_stats = {
            "wpm": round(avg(wpm_total), 1),
            "avg_pause_duration": round(avg(pause_duration_total), 2),
            "speech_ratio": round(min(100, avg(user_scores["fluency"]) + 10), 1)
        }
        
        # AI Insight
        recent_f = avg(recent_fluency)
        old_f = avg(old_fluency)
        fluency_imp = recent_f - old_f if old_f > 0 else 0
        
        diffs = {
            "pronunciation": 95 - user_avg["pronunciation"],
            "fluency": 93 - user_avg["fluency"],
            "intonation": 91 - user_avg["intonation"],
            "clarity": 95 - user_avg["clarity"]
        }
        best_dim = min(diffs, key=diffs.get)
        
        insight_msg = f"Your {best_dim} is closest to native speakers. "
        if top_diff_phonemes:
             insight_msg += f"Most noticeable difference remains the /{top_diff_phonemes[0]['phoneme']}/ articulation. "
        if fluency_imp > 0:
             insight_msg += f"Fluency improved by {fluency_imp:.1f}% compared with previous week."

        return {
            "pronunciationComparison": [
                {"name": "Pronunciation", "You": round(user_avg["pronunciation"], 1), "Native Male": native_male["pronunciation"], "Native Female": native_female["pronunciation"]}
            ],
            "fluencyComparison": [
                {"name": "Fluency", "You": round(user_avg["fluency"], 1), "Native Male": native_male["fluency"], "Native Female": native_female["fluency"]}
            ],
            "intonationComparison": [
                {"name": "Intonation", "You": round(user_avg["intonation"], 1), "Native Male": native_male["intonation"], "Native Female": native_female["intonation"]}
            ],
            "clarityComparison": [
                {"name": "Clarity", "You": round(user_avg["clarity"], 1), "Native Male": native_male["clarity"], "Native Female": native_female["clarity"]}
            ],
            "radarData": radar_data,
            "wordStatistics": top_words,
            "phonemeStatistics": top_diff_phonemes,
            "vowelStatistics": vowel_stats,
            "speakingStatistics": speaking_stats,
            "aiInsights": insight_msg
        }

    def _empty_response(self, male: dict, female: dict) -> Dict[str, Any]:
        radar_data = [
            {"dimension": "Pronunciation", "You": 0, "Native Reference": 95},
            {"dimension": "Fluency", "You": 0, "Native Reference": 93},
            {"dimension": "Intonation", "You": 0, "Native Reference": 91},
            {"dimension": "Clarity", "You": 0, "Native Reference": 95},
            {"dimension": "Similarity", "You": 0, "Native Reference": 100},
        ]
        return {
            "pronunciationComparison": [{"name": "Pronunciation", "You": 0, "Native Male": male["pronunciation"], "Native Female": female["pronunciation"]}],
            "fluencyComparison": [{"name": "Fluency", "You": 0, "Native Male": male["fluency"], "Native Female": female["fluency"]}],
            "intonationComparison": [{"name": "Intonation", "You": 0, "Native Male": male["intonation"], "Native Female": female["intonation"]}],
            "clarityComparison": [{"name": "Clarity", "You": 0, "Native Male": male["clarity"], "Native Female": female["clarity"]}],
            "radarData": radar_data,
            "wordStatistics": [],
            "phonemeStatistics": [],
            "vowelStatistics": [],
            "speakingStatistics": {"wpm": 0, "avg_pause_duration": 0, "speech_ratio": 0},
            "aiInsights": "Complete an analysis to compare your voice with native speakers."
        }
