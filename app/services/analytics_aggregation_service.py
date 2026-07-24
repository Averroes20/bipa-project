import json
from collections import defaultdict
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.audio_models import AudioAnalysis

class AnalyticsAggregationService:
    def __init__(self, db: Session):
        self.db = db

    def get_dashboard_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Fetches all user analyses, parses JSON details in memory, 
        and calculates detailed multidimensional dashboard metrics.
        """
        # Fetch all analyses for user ordered by time ascending
        analyses = self.db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id).order_by(AudioAnalysis.created_at.asc()).all()
        
        if not analyses:
            return self._empty_response()
            
        now = datetime.utcnow()
        seven_days_ago = now - timedelta(days=7)
        
        history = []
        recent_week_scores = []
        previous_week_scores = []
        total_duration = 0.0
        
        errors_tally = defaultdict(int)
        phoneme_errors_tally = defaultdict(int)
        vowel_errors_tally = defaultdict(int)
        
        scores_overall = []
        native_similarity_scores = []
        
        for record in analyses:
            total_duration += (record.duration or 0)
            scores_overall.append(record.final_score)
            
            is_recent_week = record.created_at >= seven_days_ago if record.created_at else False
            is_prev_week = (seven_days_ago > record.created_at >= (seven_days_ago - timedelta(days=7))) if record.created_at else False
            
            if is_recent_week:
                recent_week_scores.append(record.final_score)
            elif is_prev_week:
                previous_week_scores.append(record.final_score)
                
            # Parse detail
            detail = {}
            try:
                if record.analysis_detail:
                    detail = json.loads(record.analysis_detail)
            except Exception:
                pass
                
            dims = detail.get("dimensions", {})
            history.append({
                "date": record.created_at.strftime("%b %d") if record.created_at else "",
                "timestamp": record.created_at.timestamp() if record.created_at else 0,
                "overall": record.final_score,
                "pronunciation": dims.get("pronunciation", 0) or 0,
                "fluency": dims.get("fluency", 0) or 0,
                "intonation": dims.get("intonation", 0) or 0,
                "clarity": dims.get("clarity", 0) or 0
            })
            
            if dims.get("accent"):
                native_similarity_scores.append(dims.get("accent"))
            
            # Aggregate Errors
            for err in detail.get("errors", []):
                err_type = err.get("type", "")
                expected = err.get("expected", "")
                detected = err.get("detected", "")
                
                target = expected if expected else detected
                if not target:
                    continue
                    
                target = str(target).lower()
                
                if err_type in ["Substitution", "Deletion", "Insertion", "Mispronunciation"]:
                    if len(target) > 1:
                        errors_tally[target] += 1
                    elif len(target) == 1:
                        if target in ["a", "i", "u", "e", "o"]:
                             vowel_errors_tally[target] += 1
                        else:
                             phoneme_errors_tally[target] += 1
        
        # Calculate summary metrics
        avg_score = sum(scores_overall) / len(scores_overall)
        best_score = max(scores_overall)
        lowest_score = min(scores_overall)
        
        avg_recent = sum(recent_week_scores) / len(recent_week_scores) if recent_week_scores else 0
        avg_prev = sum(previous_week_scores) / len(previous_week_scores) if previous_week_scores else 0
        weekly_improvement = avg_recent - avg_prev if avg_prev > 0 else (avg_recent if avg_recent > 0 else 0)
        
        latest_dims = history[-1] if history else {}
        
        # Sort tallies
        top_words = [{"word": k, "count": v} for k, v in sorted(errors_tally.items(), key=lambda item: item[1], reverse=True)[:5]]
        top_phonemes = [{"phoneme": k, "count": v} for k, v in sorted(phoneme_errors_tally.items(), key=lambda item: item[1], reverse=True)[:5]]
        top_vowels = [{"vowel": k, "count": v} for k, v in sorted(vowel_errors_tally.items(), key=lambda item: item[1], reverse=True)[:5]]
        
        ai_insight = self._generate_ai_insight(weekly_improvement, latest_dims, top_vowels, top_phonemes)
        
        avg_pronunciation = sum(h.get("pronunciation", 0) for h in history) / len(history) if history else 0
        avg_fluency = sum(h.get("fluency", 0) for h in history) / len(history) if history else 0
        avg_native = sum(native_similarity_scores) / len(native_similarity_scores) if native_similarity_scores else 0

        return {
            "summary": {
                "total_analysis": len(analyses),
                "avg_pronunciation": avg_pronunciation,
                "weekly_improvement": weekly_improvement,
                "avg_fluency": avg_fluency,
                "avg_native_similarity": avg_native,
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
                "total_analysis": 0, "avg_pronunciation": 0, "weekly_improvement": 0, "avg_fluency": 0, "avg_native_similarity": 0
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
