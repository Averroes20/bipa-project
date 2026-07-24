from sqlalchemy import text

from app.services.db import SessionLocal


def update_user_progress(user_id, score):
    db = SessionLocal()

    try:
        row = db.execute(text("""
            SELECT avg_score, best_score, total_sessions
            FROM user_progress
            WHERE user_id = :user_id
        """), {"user_id": user_id}).fetchone()

        if row:
            avg, best, total_sessions = row

            new_total_sessions = total_sessions + 1
            new_avg = ((avg * total_sessions) + score) / new_total_sessions
            new_best = max(best, score)

            db.execute(text("""
                UPDATE user_progress
                SET 
                    avg_score = :avg, 
                    best_score = :best, 
                    total_sessions = :total_sessions,
                    updated_at = NOW()
                WHERE user_id = :user_id
            """), {
                "avg": new_avg,
                "best": new_best,
                "total_sessions": new_total_sessions,
                "user_id": user_id
            })

        else:
            db.execute(text("""
                INSERT INTO user_progress (
                    user_id, avg_score, best_score, total_sessions
                ) VALUES (
                    :user_id, :avg_score, :best_score, 1)
            """), {
                "user_id": user_id,
                "avg_score": score,
                "best_score": score
            })

            db.commit()
    finally:
        db.close()