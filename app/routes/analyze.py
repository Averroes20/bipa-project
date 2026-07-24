from fastapi import APIRouter, Depends, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user
from app.services.bipa_evaluator import BIPAEvaluator
from app.repositories.analysis_repository import AnalysisRepository

router = APIRouter(tags=["Analysis"])

@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    target_text: str = Form(""),
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    evaluator = BIPAEvaluator(db)
    
    async def event_generator():
        async for chunk in evaluator.evaluate_audio_stream(file, user_id, target_text):
            yield f"data: {chunk}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/analysis/latest")
def get_latest_analysis(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repo = AnalysisRepository(db)
    analysis = repo.get_latest_analysis(user_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found")
    return analysis

@router.get("/analysis/{analysis_id}")
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

@router.post("/rebuild-dataset")
def rebuild_dataset():
    return {"status": "dataset reference rebuilt"}

@router.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        await websocket.send_json({
            "feedback": "intonasi stabil"
        })