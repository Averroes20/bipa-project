from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.deps import get_current_user
from app.services.task_service import TaskService
from app.models.learning_models import LearningTask

router = APIRouter(tags=["Learning Tasks"])

class TaskCompleteRequest(BaseModel):
    task_id: str
    analysis_id: str

@router.get("/tasks/recommended")
def get_recommended_task(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = TaskService(db)
    task = service.get_recommended_task(user_id)
    if not task:
        raise HTTPException(status_code=404, detail="No tasks available")
    return task

@router.post("/tasks/complete")
def complete_task(
    req: TaskCompleteRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = TaskService(db)
    return service.complete_task(user_id, req.task_id, req.analysis_id)

@router.get("/progress")
def get_progress(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = TaskService(db)
    return service.get_progress(user_id)
