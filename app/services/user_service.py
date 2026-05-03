import uuid
from sqlalchemy import text
from app.services.db import SessionLocal

def get_or_create_user():
    db = SessionLocal()

    try:
        result = db.execute(text("SELECT id FROM users LIMIT 1"))
        row = result.fetchone()

        if row:
            return str(row[0])  # 🔥 SAFE

        # create dummy user
        new_id = str(uuid.uuid4())

        db.execute(text("""
            INSERT INTO users (id, name)
            VALUES (:id, :name)
        """), {
            "id": new_id,
            "name": "guest"
        })

        db.commit()

        return new_id

    finally:
        db.close()