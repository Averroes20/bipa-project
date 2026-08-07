import os
import glob
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.logger import logger
from app.services.corpus.corpus_builder import CorpusBuilder

router = APIRouter(tags=["Dataset"])

# In-memory progress tracking
rebuild_progress = {
    "status": "idle",
    "progress": 0,
    "current_file": "",
    "total": 0,
    "processed": 0
}

def process_dataset_rebuild(db: Session):
    global rebuild_progress
    builder = CorpusBuilder(db, rebuild_progress)
    builder.build()

@router.post("/dataset/rebuild")
def rebuild_dataset(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    global rebuild_progress
    if rebuild_progress["status"] == "running":
        return {"message": "Rebuild is already running", "status": rebuild_progress["status"]}
        
    rebuild_progress = {
        "status": "running",
        "progress": 0,
        "current_file": "",
        "total": 0,
        "processed": 0
    }
    
    background_tasks.add_task(process_dataset_rebuild, db)
    return {"message": "Dataset rebuild started", "status": "running"}

@router.get("/dataset/rebuild/status")
def get_rebuild_status():
    global rebuild_progress
    return {
        "status": rebuild_progress["status"],
        "progress": rebuild_progress["progress"],
        "current_file": rebuild_progress["current_file"]
    }

from fastapi import Query, UploadFile, File, Form
from typing import Optional
from sqlalchemy import func
import shutil
from app.models.dataset_models import DatasetAudio, DatasetFeature, DatasetContour, DatasetFormant

@router.get("/dataset")
def get_dataset(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    gender: Optional[str] = None
):
    query = db.query(DatasetAudio)
    if search:
        query = query.filter(DatasetAudio.filename.ilike(f"%{search}%"))
    if gender:
        query = query.filter(DatasetAudio.gender == gender)
        
    total = query.count()
    items = query.order_by(DatasetAudio.created_at.desc()).offset((page - 1) * size).limit(size).all()
    
    # Manually serialize to avoid loading everything if not needed, or just let FastAPI handle it.
    # We will build a basic dictionary structure for the list view
    result = []
    for item in items:
        # Load feature for pitch/energy mean
        feature = db.query(DatasetFeature).filter(DatasetFeature.audio_id == item.id).first()
        result.append({
            "id": item.id,
            "filename": item.filename,
            "gender": item.gender,
            "duration": item.duration,
            "language": item.language,
            "transcript": item.transcript,
            "pitch_mean": feature.pitch_mean if feature else 0,
            "energy_mean": feature.energy_mean if feature else 0,
            "created_at": item.created_at
        })
        
    return {
        "items": result,
        "total": total,
        "page": page,
        "size": size
    }

@router.get("/dataset/statistics")
def get_dataset_statistics(db: Session = Depends(get_db)):
    total_audio = db.query(DatasetAudio).count()
    male_count = db.query(DatasetAudio).filter(DatasetAudio.gender == "male").count()
    female_count = db.query(DatasetAudio).filter(DatasetAudio.gender == "female").count()
    
    avg_duration = db.query(func.avg(DatasetAudio.duration)).scalar() or 0.0
    avg_pitch = db.query(func.avg(DatasetFeature.pitch_mean)).scalar() or 0.0
    avg_energy = db.query(func.avg(DatasetFeature.energy_mean)).scalar() or 0.0
    
    # Gender specific
    male_stats = db.query(
        func.avg(DatasetAudio.duration),
        func.avg(DatasetFeature.pitch_mean),
        func.avg(DatasetFeature.energy_mean)
    ).outerjoin(DatasetFeature, DatasetFeature.audio_id == DatasetAudio.id).filter(DatasetAudio.gender == "male").first()
    
    female_stats = db.query(
        func.avg(DatasetAudio.duration),
        func.avg(DatasetFeature.pitch_mean),
        func.avg(DatasetFeature.energy_mean)
    ).outerjoin(DatasetFeature, DatasetFeature.audio_id == DatasetAudio.id).filter(DatasetAudio.gender == "female").first()

    return {
        "total_audio": total_audio,
        "male_count": male_count,
        "female_count": female_count,
        "avg_duration": float(avg_duration),
        "avg_pitch": float(avg_pitch),
        "avg_energy": float(avg_energy),
        "gender_stats": [
            {
                "name": "Male",
                "count": male_count,
                "duration": float(male_stats[0] or 0),
                "pitch": float(male_stats[1] or 0),
                "energy": float(male_stats[2] or 0)
            },
            {
                "name": "Female",
                "count": female_count,
                "duration": float(female_stats[0] or 0),
                "pitch": float(female_stats[1] or 0),
                "energy": float(female_stats[2] or 0)
            }
        ]
    }

@router.get("/dataset/{id}")
def get_dataset_by_id(id: str, db: Session = Depends(get_db)):
    audio = db.query(DatasetAudio).filter(DatasetAudio.id == id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    feature = db.query(DatasetFeature).filter(DatasetFeature.audio_id == id).first()
    formant = db.query(DatasetFormant).filter(DatasetFormant.audio_id == id).first()
    contour = db.query(DatasetContour).filter(DatasetContour.audio_id == id).first()
    
    return {
        "audio": audio,
        "feature": {
            k: v for k, v in feature.__dict__.items() if not k.startswith('_') and k != 'embedding_vector'
        } if feature else None,
        "formant": formant,
        "contour": contour
    }

@router.post("/dataset/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    gender: str = Form(...),
    transcript: Optional[str] = Form("")
):
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be male or female")
        
    dataset_dir = os.path.join(os.getcwd(), "dataset", "native", gender)
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_path = os.path.join(dataset_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # We could optionally trigger a single file corpus rebuild here, 
    # but the instructions say they will hit /dataset/rebuild to process all.
    return {"message": "File uploaded successfully", "filename": file.filename}

@router.delete("/dataset/{id}")
def delete_dataset(id: str, db: Session = Depends(get_db)):
    audio = db.query(DatasetAudio).filter(DatasetAudio.id == id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Delete physical file
    file_path = os.path.join(os.getcwd(), "dataset", "native", audio.gender, audio.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass
            
    db.delete(audio)
    db.commit()
    return {"message": "Dataset deleted successfully"}

@router.get("/dataset/audio/{id}")
def get_dataset_audio(id: str, db: Session = Depends(get_db)):
    audio = db.query(DatasetAudio).filter(DatasetAudio.id == id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return audio

@router.get("/dataset/feature/{id}")
def get_dataset_feature(id: str, db: Session = Depends(get_db)):
    feature = db.query(DatasetFeature).filter(DatasetFeature.audio_id == id).first()
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return {k: v for k, v in feature.__dict__.items() if not k.startswith('_') and k != 'embedding_vector'}

@router.get("/dataset/contour/{id}")
def get_dataset_contour(id: str, db: Session = Depends(get_db)):
    contour = db.query(DatasetContour).filter(DatasetContour.audio_id == id).first()
    if not contour:
        raise HTTPException(status_code=404, detail="Contour not found")
    return contour

@router.get("/dataset/formant/{id}")
def get_dataset_formant(id: str, db: Session = Depends(get_db)):
    formant = db.query(DatasetFormant).filter(DatasetFormant.audio_id == id).first()
    if not formant:
        raise HTTPException(status_code=404, detail="Formant not found")
    return formant
