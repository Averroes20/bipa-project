import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey
from app.core.database import Base

class LearningTask(Base):
    __tablename__ = "learning_tasks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    level = Column(String, nullable=False) # Beginner, Elementary, Intermediate, Advanced, Native-like
    difficulty = Column(String, nullable=False) # Easy, Medium, Hard
    category = Column(String, nullable=False) # Basic Vowels, Consonants, Sentence Intonation, etc.
    focus_area = Column(String, nullable=False) # e.g. /a/, /r/, /ng/, Fluency
    learning_objective = Column(Text, nullable=False)
    target_sentence = Column(Text, nullable=False)
    estimated_duration_mins = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class UserProgress(Base):
    __tablename__ = "user_learning_progress"
    
    user_id = Column(String, primary_key=True)
    current_level = Column(String, default="Beginner")
    xp = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_practice_date = Column(DateTime, nullable=True)
    completed_tasks_count = Column(Integer, default=0)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LearningHistory(Base):
    __tablename__ = "learning_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    task_id = Column(String, ForeignKey("learning_tasks.id"), nullable=True) # Optional for custom analysis
    analysis_id = Column(String, nullable=False) # Ref to AudioAnalysis
    
    xp_earned = Column(Integer, default=0)
    score = Column(Float, default=0.0)
    score_improvement = Column(Float, default=0.0)
    
    ai_coach_feedback = Column(Text, nullable=True)
    
    completed_at = Column(DateTime, default=datetime.utcnow)

class UserAchievement(Base):
    __tablename__ = "user_achievements"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    achievement_key = Column(String, nullable=False) # e.g. "first_practice", "vowel_master", "streak_30"
    unlocked_at = Column(DateTime, default=datetime.utcnow)
