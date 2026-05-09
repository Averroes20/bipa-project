from sqlalchemy import text
from app.services.db import engine

def update_global_stats():
    query = """
    INSERT INTO global_statistics (
        gender_label,
        avg_pitch_mean,
        avg_pitch_range,
        avg_energy_mean,
        avg_pause_ratio,
        sample_count,
        updated_at
    )
    SELECT
        gender_label,
        AVG(pitch_mean),
        AVG(pitch_range),
        AVG(energy_mean),
        AVG(pause_ratio),
        COUNT(*),
        NOW()
    FROM audio_analysis
    GROUP BY gender_label;
    """

    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()