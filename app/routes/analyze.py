from fastapi import APIRouter, Depends, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user
from app.services.pronunciation.pronunciation_pipeline import PronunciationPipeline
from app.repositories.analysis_repository import AnalysisRepository
from app.services.pipeline.audio_preprocessing import AudioPreprocessingService
from app.services.pipeline.feature_service import FeatureExtractionService
from app.audio_engine.embeddings import extract_embedding
from app.services.pipeline.alignment_service import AlignmentService
from app.services.pipeline.vowel_service import VowelAnalysisService
from app.services.pipeline.articulation_service import ArticulationAnalysisService
from app.services.pipeline.pitch_service import PitchAnalysisService
from app.schemas.analysis_response import AnalysisResponse
from app.core.logger import logger
import os
import glob
import numpy as np

router = APIRouter(tags=["Analysis"])

@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    target_text: str = Form(""),
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    evaluator = PronunciationPipeline(db)
    
    async def event_generator():
        async for chunk in evaluator.evaluate(file, user_id, target_text):
            yield f"data: {chunk}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# New CAPT API Endpoints
from app.models.audio_models import AnalysisWord, AnalysisPhoneme, AnalysisIntonation, AnalysisPronunciation, AnalysisFeedback

@router.get("/analysis/{analysis_id}/words")
def get_analysis_words(analysis_id: str, db: Session = Depends(get_db)):
    words = db.query(AnalysisWord).filter(AnalysisWord.analysis_id == analysis_id).all()
    return {"words": words}

@router.get("/analysis/{analysis_id}/phonemes")
def get_analysis_phonemes(analysis_id: str, db: Session = Depends(get_db)):
    phonemes = db.query(AnalysisPhoneme).filter(AnalysisPhoneme.analysis_id == analysis_id).all()
    return {"phonemes": phonemes}

@router.get("/analysis/{analysis_id}/intonation")
def get_analysis_intonation(analysis_id: str, db: Session = Depends(get_db)):
    intonation = db.query(AnalysisIntonation).filter(AnalysisIntonation.analysis_id == analysis_id).first()
    return intonation or {}

@router.get("/analysis/{analysis_id}/mispronounced")
def get_analysis_mispronounced(analysis_id: str, db: Session = Depends(get_db)):
    mis = db.query(AnalysisPronunciation).filter(AnalysisPronunciation.analysis_id == analysis_id).all()
    return {"mispronounced": mis}

@router.get("/analysis/{analysis_id}/feedback")
def get_analysis_feedback(analysis_id: str, db: Session = Depends(get_db)):
    fb = db.query(AnalysisFeedback).filter(AnalysisFeedback.analysis_id == analysis_id).first()
    return fb or {}

@router.get("/analysis/{analysis_id}/vowel-space")
def get_analysis_vowel_space(analysis_id: str, db: Session = Depends(get_db)):
    # Vowel space was integrated via phoneme mapping, we can fetch all phonemes that are vowels
    phonemes = db.query(AnalysisPhoneme).filter(AnalysisPhoneme.analysis_id == analysis_id, AnalysisPhoneme.phoneme.in_(['a','i','u','e','o'])).all()
    return {"vowels": phonemes}

@router.get("/analysis/{analysis_id}/pronunciation")
def get_analysis_pronunciation(analysis_id: str, db: Session = Depends(get_db)):
    # Returns word and phoneme aggregated scores by querying them
    words = db.query(AnalysisWord).filter(AnalysisWord.analysis_id == analysis_id).all()
    return {"total_words_evaluated": len(words)}

@router.get("/analysis/latest", response_model=AnalysisResponse)
def get_latest_analysis(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repo = AnalysisRepository(db)
    analysis = repo.get_latest_analysis(user_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found")
    return analysis

@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_by_id(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repo = AnalysisRepository(db)
    analysis = repo.get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@router.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        await websocket.send_json({
            "feedback": "intonasi stabil"
        })