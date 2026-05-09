from fastapi import APIRouter, Depends, UploadFile, File
from app.services import db
from app.services.db_ops import save_analysis,map_analysis_data
from app.services.progress_service import update_user_progress
from app.utils.json_safe import to_python
from app.services.reference_service import get_dataset_reference_from_db
from app.services.interpretation import generate_feedback_v2
from app.services import (
    audio_loader,
    feature_extraction,
    dataset_loader,
    comparison,
    scoring,
    interpretation,
    embedding,
    llm
)
from app.services.comparison import compare_embedding
from app.services.vector_service import compute_embedding_score, get_top_k_from_db
from app.services.dataset_service import rebuild_dataset_referennce
from fastapi import WebSocket
from app.services.db import SessionLocal
from sqlalchemy import text
from app.core.deps import get_current_user

router = APIRouter()

@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    # 🎧 Load audio
    audio_data = audio_loader.load_audio(file)

    # 🎧 Feature
    user_features = feature_extraction.extract_features(
        audio_data["audio_22k"],
        audio_data["sr_22k"]
    )

    # 🤖 Embedding
    user_emb = embedding.extract_embedding(
        audio_data["audio_16k"],
        audio_data["sr_16k"]
    )

    # # 📂 Cached dataset
    # dataset = dataset_loader.get_dataset()

    # # 🔍 DTW
    # comparison_result = comparison.compare(user_features, dataset)

    male_candidates = get_top_k_from_db(user_emb, k=10, gender="male")
    female_candidates = get_top_k_from_db(user_emb, k=10, gender="female")

    dataset_filtered = {
        "male": male_candidates,
        "female": female_candidates
    }

    comparison_result = comparison.compare(user_features, dataset_filtered)

    # 🤖 Embedding
    embedding_scores = {
        "male": compute_embedding_score(user_emb, dataset_filtered["male"]),
        "female": compute_embedding_score(user_emb, dataset_filtered["female"])
    }


    # 📊 Score
    score_result = scoring.compute_score(
        comparison_result,
        embedding_scores
    )

    # 💬 Feedback
    dataset_reference = get_dataset_reference_from_db()

    feedback = generate_feedback_v2(user_features, dataset_reference)

    ai_feedback = llm.generate_natural_feedback({
        "features": user_features,
        "scores": score_result,
        "rule_based": feedback
    })

    # 💾 Save
    data = map_analysis_data(
        user_id,
        user_features,
        score_result,
        embedding_scores,
        user_emb,
        ai_feedback
    )

    save_analysis(data)

    update_user_progress(
        user_id,
        score_result["overall"]
    )

    return to_python({
        "score": score_result["overall"],
        "male": score_result["male"],
        "female": score_result["female"],
        "features": user_features,
        "feedback": feedback,
        "ai_feedback": ai_feedback
    })

@router.post("/rebuild-dataset")
def rebuild_dataset():
    rebuild_dataset_referennce()
    return {"status": "dataset reference rebuilt"}

@router.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()

        # process chunk audio
        # extract feature / embedding

        await websocket.send_json({
            "feedback": "intonasi stabil"
        })

@router.get("/dashboard/summary")
def dashboard_summary():
    db = SessionLocal()

    try:
        result = db.execute(text("""
            SELECT
                AVG(final_score) as avg_score,
                COUNT(*) as total_sessions,
                AVG(pitch_mean) as avg_pitch
            FROM audio_analysis
        """))
        row = result.mappings().fetchone()
        return dict(row) if row else {}
    
    finally:
        db.close()


@router.get("/user/{user_id}/progress")
def user_progress(user_id: str = Depends(get_current_user)):
    db = SessionLocal()

    try:
        result = db.execute(
            text("""
                SELECT
                    AVG(final_score) as avg_score,
                    MAX(final_score) as best_score,
                    COUNT(*) as sessions
                FROM audio_analysis
                WHERE user_id = :user_id
            """),
            {"user_id": user_id}
        )

        row = result.mappings().fetchone()
        return dict(row) if row else {}

    finally:
        db.close()

@router.post("/analytics/rebuild")
def rebuild_analytics():
    from app.services.analytics_service import rebuild_global_statistics
    rebuild_global_statistics()
    return {"status": "analytics rebuilt"}

@router.get("/analytics/global")
def get_global():
    db = SessionLocal()

    try:
        user_stats = db.execute(text("""
            SELECT gender_label, pitch_mean, energy_mean, pause_ratio, duration
            FROM global_statistics
        """)).fetchall()

        native_stats = db.execute(text("""
            SELECT gender_label, pitch_mean, energy_mean, pause_ratio, duration
            FROM dataset_reference
        """)).fetchall()

        def to_dict(rows):
            return {
                r[0]: {
                    "pitch_mean": float(r[1]),
                    "energy_mean": float(r[2]),
                    "pause_ratio": float(r[3]),
                    "duration": float(r[4])
                }
                for r in rows
            }
        
        user = to_dict(user_stats)
        native = to_dict(native_stats)

        gap = {}
        for g in user:
            if g in native:
                gap[g] = {
                    k: user[g][k] - native[g][k]
                    for k in user[g]
                }

        return{
            "user": user,
            "native": native,
            "gap": gap
        }
    
    finally:
        db.close()