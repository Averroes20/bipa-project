import json
import random
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.learning_models import LearningTask, UserProgress, LearningHistory, UserAchievement
from app.models.audio_models import AudioAnalysis
from app.repositories.analysis_repository import AnalysisRepository

class TaskService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AnalysisRepository(db)
        self._ensure_seed_data()
        
    def _ensure_seed_data(self):
        # Insert a few default tasks if none exist
        if self.db.query(LearningTask).count() == 0:
            tasks = [
                LearningTask(title="Mastering the /a/ Vowel", level="Beginner", difficulty="Easy", category="Basic Vowels", focus_area="/a/", learning_objective="Practice clear pronunciation of the open vowel /a/.", target_sentence="Saya makan nasi dan sate ayam.", estimated_duration_mins=1),
                LearningTask(title="Trilled R Practice", level="Beginner", difficulty="Medium", category="Consonants", focus_area="/r/", learning_objective="Produce a clear alveolar trill for the /r/ consonant.", target_sentence="Rina berlari menuju rumah sakit.", estimated_duration_mins=2),
                LearningTask(title="Basic Greeting Intonation", level="Elementary", difficulty="Easy", category="Sentence Intonation", focus_area="Intonation", learning_objective="Use friendly, rising intonation for greetings.", target_sentence="Apa kabar hari ini?", estimated_duration_mins=1),
                LearningTask(title="The Nasal /ng/ Sound", level="Intermediate", difficulty="Hard", category="Difficult Sounds", focus_area="/ng/", learning_objective="Distinguish and pronounce the velar nasal /ng/ clearly.", target_sentence="Burung elang terbang sangat tinggi.", estimated_duration_mins=2)
            ]
            self.db.add_all(tasks)
            self.db.commit()

    def get_progress(self, user_id: str) -> Dict[str, Any]:
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        if not progress:
            progress = UserProgress(user_id=user_id)
            self.db.add(progress)
            self.db.commit()
            
        history = self.db.query(LearningHistory).filter(LearningHistory.user_id == user_id).all()
        achievements = self.db.query(UserAchievement).filter(UserAchievement.user_id == user_id).all()
        
        return {
            "level": progress.current_level,
            "xp": progress.xp,
            "current_streak": progress.current_streak,
            "longest_streak": progress.longest_streak,
            "completed_tasks": progress.completed_tasks_count,
            "history_count": len(history),
            "achievements": [a.achievement_key for a in achievements]
        }

    def get_recommended_task(self, user_id: str) -> Dict[str, Any]:
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        level = progress.current_level if progress else "Beginner"
        
        # Adaptive learning: find weakest phoneme from last analyses
        analyses = self.db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id).order_by(AudioAnalysis.created_at.desc()).limit(10).all()
        
        weak_phoneme = None
        if analyses:
            phoneme_errors = {}
            for a in analyses:
                if a.analysis_detail:
                    try:
                        detail = json.loads(a.analysis_detail)
                        for err in detail.get("errors", []):
                            target = err.get("expected", err.get("detected", "")).lower()
                            if len(target) == 1:
                                phoneme_errors[target] = phoneme_errors.get(target, 0) + 1
                    except Exception:
                        pass
            if phoneme_errors:
                weak_phoneme = sorted(phoneme_errors.items(), key=lambda i: i[1], reverse=True)[0][0]
                
        # Find tasks that match the weak phoneme
        if weak_phoneme:
            task = self.db.query(LearningTask).filter(LearningTask.focus_area.ilike(f"%/{weak_phoneme}/%")).first()
            if task:
                return self._task_to_dict(task, is_adaptive=True)
                
        # Fallback to level-appropriate task
        tasks = self.db.query(LearningTask).filter(LearningTask.level == level).all()
        if not tasks:
            tasks = self.db.query(LearningTask).all()
            
        selected = random.choice(tasks) if tasks else None
        if not selected:
             return None
        return self._task_to_dict(selected, is_adaptive=False)
        
    def complete_task(self, user_id: str, task_id: str, analysis_id: str) -> Dict[str, Any]:
        task = self.db.query(LearningTask).filter(LearningTask.id == task_id).first()
        analysis = self.db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
        progress = self.db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
        
        if not progress:
            progress = UserProgress(user_id=user_id)
            self.db.add(progress)
            
        score = analysis.final_score if analysis else 0.0
        
        # Calculate XP
        xp = 20 # Base completion
        if score >= 80: xp += 15
        if score >= 95: xp += 25
        
        # Streak logic
        now = datetime.utcnow().date()
        if progress.last_practice_date:
            last = progress.last_practice_date.date()
            if last == now - timedelta(days=1):
                progress.current_streak += 1
                xp += 10 # Streak bonus
            elif last < now - timedelta(days=1):
                progress.current_streak = 1 # Reset
        else:
            progress.current_streak = 1
            
        progress.last_practice_date = datetime.utcnow()
        if progress.current_streak > progress.longest_streak:
            progress.longest_streak = progress.current_streak
            
        progress.xp += xp
        progress.completed_tasks_count += 1
        
        # Level up logic
        if progress.xp >= 500 and progress.current_level == "Beginner": progress.current_level = "Elementary"
        if progress.xp >= 1500 and progress.current_level == "Elementary": progress.current_level = "Intermediate"
        if progress.xp >= 3000 and progress.current_level == "Intermediate": progress.current_level = "Advanced"
        
        # Achievements
        unlocked = []
        if progress.completed_tasks_count == 1:
            self._award(user_id, "first_practice", unlocked)
        if progress.current_streak == 7:
            self._award(user_id, "streak_7", unlocked)
            
        # Coaching logic
        coach = "Great job completing the task!"
        if task:
            if score < 70:
                 coach = f"You struggled a bit with this one. Practice {task.focus_area} slowly before increasing speed."
            elif score > 90:
                 coach = f"Excellent pronunciation of {task.focus_area}! You are ready for harder challenges."
        
        hist = LearningHistory(
            user_id=user_id,
            task_id=task_id,
            analysis_id=analysis_id,
            xp_earned=xp,
            score=score,
            ai_coach_feedback=coach
        )
        self.db.add(hist)
        self.db.commit()
        
        return {
            "xp_earned": xp,
            "new_total_xp": progress.xp,
            "new_level": progress.current_level,
            "score": score,
            "coach_feedback": coach,
            "achievements_unlocked": unlocked
        }
        
    def _award(self, user_id: str, key: str, unlocked_list: list):
        exists = self.db.query(UserAchievement).filter_by(user_id=user_id, achievement_key=key).first()
        if not exists:
            self.db.add(UserAchievement(user_id=user_id, achievement_key=key))
            unlocked_list.append(key)
            
    def _task_to_dict(self, task: LearningTask, is_adaptive: bool) -> dict:
        return {
            "id": task.id,
            "title": task.title,
            "level": task.level,
            "difficulty": task.difficulty,
            "category": task.category,
            "focus_area": task.focus_area,
            "learning_objective": task.learning_objective,
            "target_sentence": task.target_sentence,
            "estimated_duration_mins": task.estimated_duration_mins,
            "is_adaptive": is_adaptive
        }
