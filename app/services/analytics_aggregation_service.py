import json
from collections import defaultdict
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.audio_models import AudioAnalysis

class AnalyticsAggregationService:
    def __init__(self, db: Session):
        self.db = db

    def get_dashboard_analytics(self, user_id: str, period: str = "all") -> Dict[str, Any]:
        from sqlalchemy import func, case
        from app.models.audio_models import AnalysisWord, AnalysisPhoneme

        now = datetime.utcnow()
        date_filter = None
        if period == "7d":
            date_filter = now - timedelta(days=7)
        elif period == "30d":
            date_filter = now - timedelta(days=30)
        elif period == "3m":
            date_filter = now - timedelta(days=90)
            
        query = self.db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id)
        if date_filter:
            query = query.filter(AudioAnalysis.created_at >= date_filter)

        analyses = query.order_by(AudioAnalysis.created_at.asc()).all()
        
        if not analyses:
            return self._empty_response()
            
        seven_days_ago = now - timedelta(days=7)
        
        history = []
        recent_week_scores = []
        previous_week_scores = []
        total_duration = 0.0
        
        scores_overall = []
        word_stress_scores = []
        
        analysis_ids = []
        for record in analyses:
            analysis_ids.append(record.id)
            total_duration += (record.duration or 0)
            scores_overall.append(record.final_score)
            
            is_recent_week = record.created_at >= seven_days_ago if record.created_at else False
            is_prev_week = (seven_days_ago > record.created_at >= (seven_days_ago - timedelta(days=7))) if record.created_at else False
            
            if is_recent_week:
                recent_week_scores.append(record.final_score)
            elif is_prev_week:
                previous_week_scores.append(record.final_score)
                
            detail = {}
            try:
                if record.analysis_detail:
                    detail = json.loads(record.analysis_detail)
            except Exception:
                pass
                
            dims = detail # Using flat detail object directly
            history.append({
                "date": record.created_at.strftime("%b %d") if record.created_at else "",
                "timestamp": record.created_at.timestamp() if record.created_at else 0,
                "overall": record.final_score,
                "pronunciation": dims.get("pronunciation_score", 0) or 0,
                "fluency": dims.get("fluency_score", 0) or 0,
                "intonation": dims.get("intonation_score", 0) or 0,
                "rhythm": dims.get("rhythm_score", 0) or 0
            })
            word_stress_scores.append(dims.get("accent_score", 0) or 0)
        
        avg_score = sum(scores_overall) / len(scores_overall)
        best_score = max(scores_overall)
        lowest_score = min(scores_overall)
        
        avg_recent = sum(recent_week_scores) / len(recent_week_scores) if recent_week_scores else 0
        avg_prev = sum(previous_week_scores) / len(previous_week_scores) if previous_week_scores else 0
        weekly_improvement = avg_recent - avg_prev if avg_prev > 0 else (avg_recent if avg_recent > 0 else 0)
        
        latest_dims = history[-1] if history else {}
        
        # Word Analytics aggregation using SQL
        words_query = self.db.query(
            func.lower(AnalysisWord.word).label("word"),
            func.sum(case((AnalysisWord.overall_score < 75, 1), else_=0)).label("errors")
        ).filter(AnalysisWord.analysis_id.in_(analysis_ids)).group_by(func.lower(AnalysisWord.word)).all()
        top_words = [{"word": w.word, "count": w.errors} for w in sorted(words_query, key=lambda i: i.errors, reverse=True) if w.errors > 0][:5]

        # Phoneme Analytics aggregation using SQL
        phonemes_query = self.db.query(
            func.lower(AnalysisPhoneme.phoneme).label("phoneme"),
            func.sum(case((AnalysisPhoneme.pronunciation_score < 75, 1), else_=0)).label("errors")
        ).filter(AnalysisPhoneme.analysis_id.in_(analysis_ids)).group_by(func.lower(AnalysisPhoneme.phoneme)).all()
        
        vowel_set = {"a", "i", "u", "e", "o", "ɛ", "ɔ"}
        vowels = [{"vowel": p.phoneme, "count": p.errors} for p in phonemes_query if p.phoneme in vowel_set and p.errors > 0]
        consonants = [{"phoneme": p.phoneme, "count": p.errors} for p in phonemes_query if p.phoneme not in vowel_set and p.errors > 0]
        
        top_phonemes = sorted(consonants, key=lambda i: i["count"], reverse=True)[:5]
        top_vowels = sorted(vowels, key=lambda i: i["count"], reverse=True)[:5]
        
        ai_insight = self._generate_ai_insight(weekly_improvement, latest_dims, top_vowels, top_phonemes)
        
        avg_pronunciation = sum(h.get("pronunciation", 0) for h in history) / len(history) if history else 0
        avg_fluency = sum(h.get("fluency", 0) for h in history) / len(history) if history else 0
        avg_word_stress = sum(word_stress_scores) / len(word_stress_scores) if word_stress_scores else 0

        return {
            "summary": {
                "total_analysis": len(analyses),
                "avg_pronunciation": avg_pronunciation,
                "weekly_improvement": weekly_improvement,
                "avg_fluency": avg_fluency,
                "avg_word_stress": avg_word_stress,
            },
            "progress": history,
            "learningStatistics": {
                "best_score": best_score,
                "avg_score": avg_score,
                "lowest_score": lowest_score,
                "total_practice_time_seconds": total_duration,
                "total_analyses": len(analyses)
            },
            "errors": {
                "most_mispronounced_words": top_words,
                "most_mispronounced_phonemes": top_phonemes,
                "most_difficult_vowels": top_vowels
            },
            "aiInsight": ai_insight
        }
        
    def _generate_ai_insight(self, weekly_improvement: float, latest_dims: dict, top_vowels: list, top_phonemes: list) -> dict:
        insight = {
            "strength": "",
            "needs_improvement": "",
            "recommendation": ""
        }
        
        if weekly_improvement > 0:
             insight["strength"] = f"Overall performance increased by {weekly_improvement:.1f}% this week."
        elif latest_dims.get("fluency", 0) > 80:
             insight["strength"] = "Speech fluency is very stable and consistent."
        elif latest_dims.get("pronunciation", 0) > 80:
             insight["strength"] = "Pronunciation accuracy is highly native-like."
        else:
             insight["strength"] = "Consistently practicing, building a solid baseline."
             
        if top_vowels:
             v = top_vowels[0]["vowel"]
             insight["needs_improvement"] = f"Vowel /{v}/ often differs from native speakers."
             insight["recommendation"] = f"Practice opening your mouth clearly for /{v}/ vowels for 5 minutes daily."
        elif top_phonemes:
             p = top_phonemes[0]["phoneme"]
             insight["needs_improvement"] = f"Consonant /{p}/ is frequently misarticulated."
             insight["recommendation"] = f"Focus on the articulation of /{p}/ during your next practice sessions."
        else:
             insight["needs_improvement"] = "Minor phrasing and intonation inconsistencies."
             insight["recommendation"] = "Listen to native speakers and try to mimic their melodic rhythm."
             
        return insight

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_analysis": 0, "avg_pronunciation": 0, "weekly_improvement": 0, "avg_fluency": 0, "avg_word_stress": 0
            },
            "progress": [],
            "learningStatistics": {
                "best_score": 0, "avg_score": 0, "lowest_score": 0, "total_practice_time_seconds": 0, "total_analyses": 0
            },
            "errors": {
                "most_mispronounced_words": [], "most_mispronounced_phonemes": [], "most_difficult_vowels": []
            },
            "aiInsight": {
                "strength": "No data yet",
                "needs_improvement": "No data yet",
                "recommendation": "Complete your first analysis to receive insights."
            }
        }
