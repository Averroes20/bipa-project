from fastapi import APIRouter, Depends, HTTPException
import json
from app.models.audio_models import AudioAnalysis
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user
from app.repositories.analysis_repository import AnalysisRepository
from app.services.analytics_service import rebuild_global_statistics_service
from app.services.analytics_aggregation_service import AnalyticsAggregationService
from app.services.comparison_analytics_service import ComparisonAnalyticsService
from app.schemas.analysis import DashboardSummaryResponse, UserProgressResponse

router = APIRouter(tags=["Analytics & Dashboard"])

@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    return repo.get_dashboard_summary()

@router.get("/analytics/dashboard")
def get_full_dashboard(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = AnalyticsAggregationService(db)
    return service.get_dashboard_analytics(user_id)

@router.get("/analytics/comparison")
def get_full_comparison(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = ComparisonAnalyticsService(db)
    return service.get_comparison_analytics(user_id)

@router.get("/user/progress", response_model=UserProgressResponse)
def user_progress(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repo = AnalysisRepository(db)
    return repo.get_user_progress(user_id)

@router.get("/user/history")
def user_history(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repo = AnalysisRepository(db)
    return repo.get_user_history(user_id)

@router.post("/analytics/rebuild")
def rebuild_analytics(db: Session = Depends(get_db)):
    rebuild_global_statistics_service(db)
    return {"status": "analytics rebuilt successfully"}

@router.get("/analytics/global")
def get_global(db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    native_stats = repo.get_dataset_reference()
    
    # User stats from global_statistics
    summary = repo.get_dashboard_summary()
    user_stats = {
        "overall": {
            "pitch_mean": summary["avg_pitch"],
            "energy_mean": summary["avg_energy"],
            "pause_ratio": summary["avg_pause"],
            "duration": 0.0
        }
    }

    return {
        "user": user_stats,
        "native": native_stats,
    }

@router.get("/analysis/latest")
def get_latest_analysis(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    analysis = db.query(AudioAnalysis).filter(AudioAnalysis.user_id == user_id).order_by(AudioAnalysis.created_at.desc()).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found")
    
    result = json.loads(analysis.analysis_detail) if analysis.analysis_detail else {}
    result["id"] = analysis.id
    return result

@router.get("/analysis/{id}")
def get_analysis_by_id(
    id: str,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    analysis = db.query(AudioAnalysis).filter(AudioAnalysis.id == id, AudioAnalysis.user_id == user_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = json.loads(analysis.analysis_detail) if analysis.analysis_detail else {}
    result["id"] = analysis.id
    return result

@router.get("/analytics/phonetic-deviations")
def get_phonetic_deviations(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from app.models.audio_models import UserPhonemeSummary
    
    summaries = db.query(UserPhonemeSummary).filter(UserPhonemeSummary.user_id == user_id).all()
    
    if not summaries:
        return {
            "vowels": [],
            "consonants": [],
            "mostDifficultVowels": [],
            "mostDifficultConsonants": [],
            "bestPronouncedSounds": [],
            "mostImprovedSounds": [],
            "aiSummary": ""
        }
        
    vowel_set = {"a", "i", "u", "e", "o"}
    
    vowels_data = []
    consonants_data = []
    
    for s in summaries:
        data = {
            "phoneme": s.phoneme,
            "accuracy": s.accuracy,
            "mistakes": s.mistake_count,
            "occurrences": s.occurrences,
            "improvement": 0,
            "confidence": 0.9
        }
        if s.phoneme in vowel_set:
            vowels_data.append(data)
        else:
            consonants_data.append(data)
            
    most_diff_v = sorted(vowels_data, key=lambda x: (x["accuracy"], -x["mistakes"]))
    most_diff_c = sorted(consonants_data, key=lambda x: (x["accuracy"], -x["mistakes"]))
    best = sorted(vowels_data + consonants_data, key=lambda x: (-x["accuracy"], -x["occurrences"]))
    
    ai_summary = "Your pronunciation is generally clear."
    if most_diff_c:
        ai_summary = f"You consistently struggle with the consonant /{most_diff_c[0]['phoneme']}/. "
    if most_diff_v:
        ai_summary += f"Practice the vowel /{most_diff_v[0]['phoneme']}/ to improve clarity. "
    if best:
        ai_summary += f"Excellent pronunciation of /{best[0]['phoneme']}/!"
        
    return {
        "phoneticDeviations": {
            "vowels": vowels_data,
            "consonants": consonants_data
        },
        "mostDifficultVowels": most_diff_v[:3],
        "mostDifficultConsonants": most_diff_c[:3],
        "bestPronouncedSounds": best[:3],
        "mostImprovedSounds": [],
        "aiSummary": ai_summary.strip()
    }
