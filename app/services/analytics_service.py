from sqlalchemy.orm import Session
from app.repositories.analysis_repository import AnalysisRepository

def rebuild_global_statistics_service(db: Session) -> None:
    repo = AnalysisRepository(db)
    repo.rebuild_global_statistics()