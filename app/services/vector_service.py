import numpy as np
from sqlalchemy import text
from app.services.db import SessionLocal
from sklearn.metrics.pairwise import cosine_similarity

def to_pgvector(arr):
    return "[" + ",".join(map(str, arr)) + "]"

def get_top_k_from_db(user_emb, k=20, gender=None):
    db = SessionLocal()

    try:
        base_query = """
            SELECT *
            FROM dataset_reference
        """

        # 🔥 kalau ada filter gender
        if gender:
            base_query += " WHERE gender_label = :gender"

        base_query += """
            ORDER BY embedding_vector <-> :user_emb
            LIMIT :k
        """

        query = text(base_query)

        params = {
            "user_emb": to_pgvector(user_emb.tolist()),
            "k": k
        }

        if gender:
            params["gender"] = gender

        result = db.execute(query, params)

        return result.fetchall()

    finally:
        db.close()

def compute_embedding_score(user_emb, candidates):
    import numpy as np
    import re
    from sklearn.metrics.pairwise import cosine_similarity

    def clean_vector(vec):
        vec = np.array(vec, dtype=float)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def parse_embedding(emb):
        if isinstance(emb, list):
            return emb

        if isinstance(emb, str):
            values = re.split(r"[,\s]+", emb.strip("[]"))
            return [float(v) for v in values if v]

        raise ValueError("Invalid embedding format")

    scores = []

    user_vec = clean_vector(user_emb).reshape(1, -1)

    for c in candidates:
        try:
            emb = parse_embedding(c.embedding_vector)
            emb_vec = clean_vector(emb).reshape(1, -1)

            if emb_vec.shape[1] != user_vec.shape[1]:
                continue  # skip corrupt data

            sim = cosine_similarity(user_vec, emb_vec)[0][0]
            scores.append(sim)

        except Exception as e:
            print("🔥 SKIP DATA ERROR:", e)
            continue

    return sum(scores) / len(scores) if scores else 0.0