import os

from sqlalchemy import text

from app.services import audio_loader, feature_extraction, embedding
from app.services.db import SessionLocal

DATASET_PATH = "dataset/native"

def rebuild_dataset_referennce():
    db = SessionLocal()

    try:
        db.execute(text("DELETE FROM dataset_reference"))

        def to_pgvector(arr):
            return "[" + ",".join(map(str, arr)) + "]"

        for gender in ["male", "female"]:
            folder = os.path.join(DATASET_PATH, gender)

            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)

                # 🎧 load audio
                audio_data = audio_loader.load_audio(filepath)

                # 🎧 feature
                features = feature_extraction.extract_features(
                    audio_data["audio_22k"],
                    audio_data["sr_22k"]
                )

                # 🤖 embedding (INI YANG LO CARI)
                emb = embedding.extract_embedding(
                    audio_data["audio_16k"],
                    audio_data["sr_16k"]
                )

                # 💾 insert
                db.execute(text("""
                    INSERT INTO dataset_reference (
                        gender_label,
                        pitch_mean,
                        energy_mean,
                        pause_ratio,
                        duration,
                        embedding_vector
                    ) VALUES (
                        :gender,
                        :pitch_mean,
                        :energy_mean,
                        :pause_ratio,
                        :duration,
                        :embedding_vector
                    )
                """), {
                    "gender": gender,
                    "pitch_mean": float(features.get("pitch_mean", 0)),
                    "energy_mean": float(features.get("energy_mean", 0)),
                    "pause_ratio": float(features.get("pause_ratio", 0)),
                    "duration": float(features.get("duration", 0)),
                    "embedding_vector": to_pgvector(emb.tolist())
                })

        db.commit()

    finally:
        db.close()