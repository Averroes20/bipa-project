from fastapi import APIRouter, UploadFile, File
from app.services.db_ops import save_analysis,map_analysis_data
from app.services.similiarity import search_similar
from app.utils.json_safe import to_python
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
from fastapi import WebSocket
from app.services.db import SessionLocal
from sqlalchemy import text
from app.services.user_service import get_or_create_user

router = APIRouter()

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), user_id: str = None):

     # 🔥 AUTO HANDLE USER
    user_id = user_id or get_or_create_user()

    # 🎧 Load audio
    audio_data = audio_loader.load_audio(file)

    # 🎧 Feature (DTW)
    user_features = feature_extraction.extract_features(
        audio_data["audio_22k"],
        audio_data["sr_22k"]
    )

    # 🤖 Embedding
    user_emb = embedding.extract_embedding(
        audio_data["audio_16k"],
        audio_data["sr_16k"]
    )

    # 📂 Dataset
    dataset = dataset_loader.preprocess_dataset()

    # 🔍 DTW
    comparison_result = comparison.compare(user_features, dataset)

    # 🤖 Embedding compare
    embedding_scores = {
        "male": compare_embedding(user_emb, dataset["male_embeddings"]),
        "female": compare_embedding(user_emb, dataset["female_embeddings"])
    }

    # 📊 Scoring
    score_result = scoring.compute_score(
        comparison_result,
        embedding_scores
    )

    # 💬 Rule-based feedback
    feedback = interpretation.generate_feedback(score_result)

    # 🤖 AI feedback
    ai_feedback = llm.generate_natural_feedback({
        "features": user_features,
        "scores": score_result,
        "rule_based": feedback
    })

    # 🧠 Mapping data (🔥 INI YANG TADI LO TANYA)
    data = map_analysis_data(
        user_id,
        user_features,
        score_result,
        embedding_scores,
        user_emb,
        ai_feedback
    )

    # 💾 Save ke DB
    save_analysis(data)

    # 🎯 Response
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
    from app.services.dataset_loader import preprocess_dataset
    preprocess_dataset()
    return {"status": "dataset rebuilt"}

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

    result = db.execute("""
        SELECT
            AVG(final_score) as avg_score,
            COUNT(*) as total_sessions,
            AVG(pitch_mean) as avg_pitch
        FROM audio_analysis
    """)

    return dict(result.fetchone())

@router.get("/user/{user_id}/progress")
def user_progress(user_id: str):
    db = SessionLocal()

    result = db.execute(text(f"""
        SELECT
            AVG(final_score) as avg_score,
            MAX(final_score) as best_score,
            COUNT(*) as sessions
        FROM audio_analysis
        WHERE user_id = :user_id
    """))

    return dict(result.fetchone())