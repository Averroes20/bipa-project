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

    def get_comparison_analytics(self, user_id: str, period: str = "all") -> Dict[str, Any]:
        from sqlalchemy import func, case
        from app.models.audio_models import AnalysisWord, AnalysisPhoneme

        now = datetime.utcnow()
        period_days = 0
        if period == "7d":
            period_days = 7
        elif period == "30d":
            period_days = 30
        elif period == "3m":
            period_days = 90

        all_analyses = self.db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id).order_by(AudioAnalysis.created_at.asc()).all()
        
        current_period_analyses = []
        previous_period_analyses = []
        
        if period == "all" or period_days == 0:
            current_period_analyses = all_analyses
        else:
            current_cutoff = now - timedelta(days=period_days)
            previous_cutoff = now - timedelta(days=period_days * 2)
            
            for a in all_analyses:
                if a.created_at >= current_cutoff:
                    current_period_analyses.append(a)
                elif a.created_at >= previous_cutoff:
                    previous_period_analyses.append(a)
        
        if not current_period_analyses:
             return self._empty_response()

        # Helper to extract dimensions
        def extract_dims(record):
            detail = {}
            try:
                if record.analysis_detail:
                    detail = json.loads(record.analysis_detail)
            except Exception:
                pass
            return {
                "pronunciation": detail.get("pronunciation_score", 0) or 0,
                "fluency": detail.get("fluency_score", 0) or 0,
                "intonation": detail.get("intonation_score", 0) or 0,
                "rhythm": detail.get("rhythm_score", 0) or 0,
                "word_stress": detail.get("accent_score", 0) or 0
            }

        # Calculate previous averages
        prev_scores = {"pronunciation": [], "fluency": [], "intonation": [], "rhythm": []}
        for record in previous_period_analyses:
            dims = extract_dims(record)
            prev_scores["pronunciation"].append(dims["pronunciation"])
            prev_scores["fluency"].append(dims["fluency"])
            prev_scores["intonation"].append(dims["intonation"])
            prev_scores["rhythm"].append(dims["rhythm"])
            
        avg = lambda x: sum(x)/len(x) if x else 0
        
        prev_avg = {
            "pronunciation": avg(prev_scores["pronunciation"]),
            "fluency": avg(prev_scores["fluency"]),
            "intonation": avg(prev_scores["intonation"]),
            "rhythm": avg(prev_scores["rhythm"])
        }

        # Calculate current averages and progress
        user_scores = {"pronunciation": [], "fluency": [], "intonation": [], "rhythm": [], "word_stress": []}
        wpm_total = []
        pause_ratio_total = []
        progress = []
        
        analysis_ids = [record.id for record in current_period_analyses]
        
        # Pre-fetch word counts per analysis for WPM calculation
        word_counts_query = self.db.query(
            AnalysisWord.analysis_id,
            func.count(AnalysisWord.id).label("count")
        ).filter(AnalysisWord.analysis_id.in_(analysis_ids)).group_by(AnalysisWord.analysis_id).all()
        word_counts = {row.analysis_id: row.count for row in word_counts_query}
        
        for record in current_period_analyses:
            dims = extract_dims(record)
            
            user_scores["pronunciation"].append(dims["pronunciation"])
            user_scores["fluency"].append(dims["fluency"])
            user_scores["intonation"].append(dims["intonation"])
            user_scores["rhythm"].append(dims["rhythm"])
            user_scores["word_stress"].append(dims["word_stress"])
            
            progress.append({
                "date": record.created_at.strftime("%b %d") if record.created_at else "",
                "pronunciation": round(dims["pronunciation"], 1),
                "fluency": round(dims["fluency"], 1),
                "intonation": round(dims["intonation"], 1),
                "rhythm": round(dims["rhythm"], 1)
            })
            
            # Speaking stats (WPM = words / (duration / 60))
            duration_val = float(record.duration or 0)
            word_count = word_counts.get(record.id, 0)
            
            if duration_val > 0 and word_count > 0:
                wpm = word_count / (duration_val / 60)
                wpm_total.append(wpm)
                
            pause_ratio_val = float(record.pause_ratio or 0)
            pause_ratio_total.append(pause_ratio_val * 100) # Convert to percentage
                             
        user_avg = {
            "pronunciation": avg(user_scores["pronunciation"]),
            "fluency": avg(user_scores["fluency"]),
            "intonation": avg(user_scores["intonation"]),
            "rhythm": avg(user_scores["rhythm"]),
            "word_stress": avg(user_scores["word_stress"])
        }
        
        changes = {
            "pronunciation": None,
            "fluency": None,
            "intonation": None,
            "rhythm": None
        }
        
        if previous_period_analyses:
             changes = {
                 "pronunciation": round(user_avg["pronunciation"] - prev_avg["pronunciation"], 1),
                 "fluency": round(user_avg["fluency"] - prev_avg["fluency"], 1),
                 "intonation": round(user_avg["intonation"] - prev_avg["intonation"], 1),
                 "rhythm": round(user_avg["rhythm"] - prev_avg["rhythm"], 1)
             }
        
        radar_data = [
            {"dimension": "Pronunciation", "You": round(user_avg["pronunciation"], 1)},
            {"dimension": "Fluency", "You": round(user_avg["fluency"], 1)},
            {"dimension": "Intonation", "You": round(user_avg["intonation"], 1)},
            {"dimension": "Rhythm", "You": round(user_avg["rhythm"], 1)},
            {"dimension": "Word Stress", "You": round(user_avg["word_stress"], 1)},
        ]
        
        # Word Analytics aggregation using SQL
        words_query = self.db.query(
            func.lower(AnalysisWord.word).label("word"),
            func.count(AnalysisWord.id).label("attempts"),
            func.count(func.distinct(AnalysisWord.analysis_id)).label("unique_sessions"),
            func.avg(AnalysisWord.overall_score).label("avg_score")
        ).filter(AnalysisWord.analysis_id.in_(analysis_ids)).group_by(func.lower(AnalysisWord.word)).all()
        
        prev_analysis_ids = [record.id for record in previous_period_analyses]
        prev_words_query = []
        if prev_analysis_ids:
            prev_words_query = self.db.query(
                func.lower(AnalysisWord.word).label("word"),
                func.avg(AnalysisWord.overall_score).label("avg_score")
            ).filter(AnalysisWord.analysis_id.in_(prev_analysis_ids)).group_by(func.lower(AnalysisWord.word)).all()
        prev_words_map = {w.word: float(w.avg_score or 0) for w in prev_words_query}
        
        word_stats = []
        for w in words_query:
            attempts = w.attempts
            unique_sessions = w.unique_sessions
            avg_score = float(w.avg_score or 0)
            
            conf = "Low"
            if unique_sessions >= 2:
                conf = "High"
            elif attempts >= 2:
                conf = "Medium"
                
            trend = None
            if w.word in prev_words_map:
                trend = round(avg_score - prev_words_map[w.word], 1)

            word_stats.append({
                "word": w.word,
                "attempts": attempts,
                "unique_sessions": unique_sessions,
                "avg_score": round(avg_score, 1),
                "confidence": conf,
                "trend": trend,
                "error_rate": 0 # kept for compatibility with insights if needed, but not used in UI
            })
            
        conf_rank = {"High": 0, "Medium": 1, "Low": 2}
        sort_key_words = lambda x: (conf_rank[x["confidence"]], -x["unique_sessions"], x["avg_score"])
        
        word_stats.sort(key=sort_key_words)
        top_words = word_stats[:5]

        # Phoneme Analytics aggregation using SQL
        phonemes_query = self.db.query(
            func.lower(AnalysisPhoneme.phoneme).label("phoneme"),
            func.count(AnalysisPhoneme.id).label("attempts"),
            func.count(func.distinct(AnalysisPhoneme.analysis_id)).label("unique_sessions"),
            func.sum(case((AnalysisPhoneme.pronunciation_score < 75, 1), else_=0)).label("errors"),
            func.avg(AnalysisPhoneme.pronunciation_score).label("avg_score")
        ).filter(AnalysisPhoneme.analysis_id.in_(analysis_ids)).group_by(func.lower(AnalysisPhoneme.phoneme)).all()
        
        prev_phonemes_query = []
        if prev_analysis_ids:
            prev_phonemes_query = self.db.query(
                func.lower(AnalysisPhoneme.phoneme).label("phoneme"),
                func.avg(AnalysisPhoneme.pronunciation_score).label("avg_score")
            ).filter(AnalysisPhoneme.analysis_id.in_(prev_analysis_ids)).group_by(func.lower(AnalysisPhoneme.phoneme)).all()
        prev_phonemes_map = {p.phoneme: float(p.avg_score or 0) for p in prev_phonemes_query}
        
        phoneme_stats = []
        for p in phonemes_query:
            attempts = p.attempts
            unique_sessions = p.unique_sessions
            avg_score = float(p.avg_score or 0)
            
            conf = "Low"
            if unique_sessions >= 2:
                conf = "High"
            elif attempts >= 2:
                conf = "Medium"
                
            errors = p.errors or 0
            error_rate = (errors / attempts * 100) if attempts > 0 else 0
            accuracy = 100 - error_rate
            
            trend = None
            if p.phoneme in prev_phonemes_map:
                trend = round(avg_score - prev_phonemes_map[p.phoneme], 1)

            phoneme_stats.append({
                "phoneme": p.phoneme,
                "attempts": attempts,
                "unique_sessions": unique_sessions,
                "avg_score": round(avg_score, 1),
                "accuracy": round(accuracy, 1),
                "error_rate": round(error_rate, 1),
                "errors": int(errors),
                "confidence": conf,
                "trend": trend
            })
            
        vowel_set = {"a", "i", "u", "e", "o", "ɛ", "ɔ"}
        vowels = [p for p in phoneme_stats if p["phoneme"] in vowel_set]
        consonants = [p for p in phoneme_stats if p["phoneme"] not in vowel_set]
        
        sort_key_phonemes = lambda x: (conf_rank[x["confidence"]], -x["unique_sessions"], x["avg_score"])
        
        def get_top_phonemes(phoneme_list, limit=5):
            phoneme_list.sort(key=sort_key_phonemes)
            return phoneme_list[:limit]

        top_diff_phonemes = get_top_phonemes(consonants)
        vowel_stats = get_top_phonemes(vowels)
        
        # Speaking Stats
        speaking_stats = {
            "wpm": round(avg(wpm_total), 1),
            "pause_ratio": round(avg(pause_ratio_total), 2),
            "speech_ratio": round(min(100, avg(user_scores["fluency"]) + 10), 1)
        }
        
        # AI Insight strictly from generated data
        scores_map = {
            "Pronunciation": user_avg["pronunciation"],
            "Fluency": user_avg["fluency"],
            "Intonation": user_avg["intonation"],
            "Rhythm": user_avg["rhythm"]
        }
        
        best_dim = max(scores_map, key=scores_map.get)
        worst_dim = min(scores_map, key=scores_map.get)
        
        insight_msg = f"Your strongest area is {best_dim} ({round(scores_map[best_dim], 1)}). "
        insight_msg += f"Your weakest area is {worst_dim} ({round(scores_map[worst_dim], 1)}). "
        
        if top_words and top_words[0]["error_rate"] > 0:
             insight_msg += f"The most problematic word is '{top_words[0]['word']}' ({top_words[0]['error_rate']}% error rate). "
        
        all_diff_phonemes = sorted(phoneme_stats, key=lambda i: (-i["error_rate"], i["avg_score"]))
        if all_diff_phonemes and all_diff_phonemes[0]["error_rate"] > 0:
             insight_msg += f"The most frequent phoneme issue is /{all_diff_phonemes[0]['phoneme']}/ ({all_diff_phonemes[0]['errors']} errors)."

        return {
            "summary": {
                "pronunciation": round(user_avg["pronunciation"], 1),
                "fluency": round(user_avg["fluency"], 1),
                "intonation": round(user_avg["intonation"], 1),
                "rhythm": round(user_avg["rhythm"], 1)
            },
            "changes": changes,
            "progress": progress,
            "radarData": radar_data,
            "wordStatistics": top_words,
            "phonemeStatistics": top_diff_phonemes,
            "vowelStatistics": vowel_stats,
            "speakingStatistics": speaking_stats,
            "aiInsights": insight_msg
        }

    def _empty_response(self) -> Dict[str, Any]:
        radar_data = [
            {"dimension": "Pronunciation", "You": 0},
            {"dimension": "Fluency", "You": 0},
            {"dimension": "Intonation", "You": 0},
            {"dimension": "Rhythm", "You": 0},
            {"dimension": "Word Stress", "You": 0},
        ]
        return {
            "summary": {
                "pronunciation": 0,
                "fluency": 0,
                "intonation": 0,
                "rhythm": 0
            },
            "changes": {
                "pronunciation": None,
                "fluency": None,
                "intonation": None,
                "rhythm": None
            },
            "progress": [],
            "radarData": radar_data,
            "wordStatistics": [],
            "phonemeStatistics": [],
            "vowelStatistics": [],
            "speakingStatistics": {"wpm": 0, "pause_ratio": 0, "speech_ratio": 0},
            "aiInsights": "Complete an analysis to track your historical progress."
        }
