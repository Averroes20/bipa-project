from sqlalchemy import text
from app.services.db import SessionLocal


def get_dataset_reference_from_db():
    db = SessionLocal()

    try:
        result = db.execute(text("""
            SELECT
                gender_label,
                pitch_mean,
                energy_mean,
                pause_ratio,
                duration
            FROM dataset_reference
        """)).fetchall()

        data = {}

        for row in result:
            gender = row[0]

            data[gender] = {
                "pitch_mean": float(row[1] or 0),
                "energy_mean": float(row[2] or 0),
                "pause_ratio": float(row[3] or 0),
                "duration": float(row[4] or 0)
            }

        return data

    finally:
        db.close()