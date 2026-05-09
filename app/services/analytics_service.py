from sqlalchemy import text

from app.services.db import SessionLocal


def rebuild_global_statistics():
    db = SessionLocal()
    
    try:
        # ambil avg dari audio_analysis
        rows = db.execute(text("""
            SELECT
                gender_label,
                AVG(pitch_mean),
                AVG(energy_mean),
                AVG(pause_ratio),
                AVG(duration)
            FROM audio_analysis
            WHERE gender_label IS NOT NULL
            GROUP BY gender_label
        """)).fetchall()

        #clear data lama
        db.execute(text("DELETE FROM global_statistics"))

        #insert data baru
        for row in rows:
            db.execute(text("""
                INSERT INTO global_statistics (
                    gender_label, 
                    pitch_mean, 
                    energy_mean, 
                    pause_ratio, 
                    duration)
                VALUES (:gender_label, :pitch_mean, :energy_mean, :pause_ratio, :duration)
            """), {
                "gender_label": row[0],
                "pitch_mean": float(row[1] or 0),
                "energy_mean": float(row[2] or 0),
                "pause_ratio": float(row[3] or 0),
                "duration": float(row[4] or 0)
            })
            
        db.commit()

    finally:
        db.close()