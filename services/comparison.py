from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

def dtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2)
    return distance

def compare(user, dataset):
    result = {"male": [], "female": []}

    for gender in ["male", "female"]: 
        for ref in dataset[gender]:

            if not isinstance(ref, dict): 
                continue

            pitch_dist = dtw_distance(user["pitch"], ref["pitch"])
            energy_dist = dtw_distance(user["energy"], ref["energy"])

            result[gender].append({
                "pitch": pitch_dist,
                "energy": energy_dist,
                "pause": abs(user["pause_ratio"] - ref["pause_ratio"])
            })

    return result

def safe_vector(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, dtype=np.float32)

        # ❗ VALIDASI
        if arr.ndim == 1:
            return arr.reshape(1, -1)

        if arr.ndim == 2:
            return arr

    raise ValueError(f"Invalid embedding format or shape: {x}")

def compare_embedding(user_emb, dataset_embs):
    scores = []

    user_vec = safe_vector(user_emb)

    for emb in dataset_embs:
        emb_vec = safe_vector(emb)

        # 🔥 FILTER DIMENSION MISMATCH
        if user_vec.shape[1] != emb_vec.shape[1]:
            continue

        sim = cosine_similarity(user_vec, emb_vec)[0][0]
        scores.append(sim)

    if len(scores) == 0:
        raise ValueError("No valid embeddings to compare")

    return sum(scores) / len(scores)