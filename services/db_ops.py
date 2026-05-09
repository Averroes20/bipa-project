from app.services.db import SessionLocal
from sqlalchemy import text
import json

def to_float(x):
    if x is None:
        return None
    return float(x)

def map_analysis_data(user_id, features, scores, embedding_scores, user_emb, ai_feedback):
    import json

    return {
        "user_id": str(user_id),
        "gender_label": None,

        "pitch_mean": float(features.get("pitch_mean", 0) or 0),
        "pitch_range": float(features.get("pitch_range", 0) or 0),
        "energy_mean": float(features.get("energy_mean", 0) or 0),
        "pause_ratio": float(features.get("pause_ratio", 0)),
        "duration": float(features.get("duration", 0)),

        "dtw_score_male": float(scores["male"]["dtw"]),
        "dtw_score_female": float(scores["female"]["dtw"]),
        "embedding_score_male": float(embedding_scores["male"]),
        "embedding_score_female": float(embedding_scores["female"]),
        "final_score": float(scores["overall"]),

        # 🔥 INI KUNCI
        "embedding": json.dumps(user_emb.tolist()),

        "ai_feedback": ai_feedback
    }

def save_analysis(data):
    db = SessionLocal()

    try:
        db.execute(text("""
            INSERT INTO audio_analysis (
                user_id,
                gender_label,
                pitch_mean,
                pitch_range,
                energy_mean,
                pause_ratio,
                duration,
                dtw_score_male,
                dtw_score_female,
                embedding_score_male,
                embedding_score_female,
                final_score,
                embedding,
                ai_feedback
            )
            VALUES (
                :user_id,
                :gender_label,
                :pitch_mean,
                :pitch_range,
                :energy_mean,
                :pause_ratio,
                :duration,
                :dtw_score_male,
                :dtw_score_female,
                :embedding_score_male,
                :embedding_score_female,
                :final_score,
                :embedding,
                :ai_feedback
            )
        """), data)

        db.commit()  # 🔥 INI WAJIB

    except Exception as e:
        print("DB ERROR:", e)  # 🔥 DEBUG
    finally:
        db.close()