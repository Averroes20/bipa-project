import numpy as np

def normalize(score):
    return max(0, 100 - score)

def aggregate(distances):
    return {
        "pitch": np.mean([d["pitch"] for d in distances]),
        "energy": np.mean([d["energy"] for d in distances]),
        "pause": np.mean([d["pause"] for d in distances])
    }

def compute_score(comparison, embedding_scores):
    result = {}

    for gender in ["male", "female"]:
        pitch = float(np.mean([d["pitch"] for d in comparison[gender]]))
        energy = float(np.mean([d["energy"] for d in comparison[gender]]))
        pause = float(np.mean([d["pause"] for d in comparison[gender]]))

        dtw_score = float(np.mean([
            normalize_distance(pitch),
            normalize_distance(energy),
            normalize_distance(pause)
        ]))

        emb_score = float(normalize_similarity(embedding_scores[gender]))

        final_score = float(
            0.4 * dtw_score +
            0.6 * emb_score
        )

        result[gender] = {
            "dtw": dtw_score,
            "embedding": emb_score,
            "final": final_score
        }

    result["overall"] = float(max(
        result["male"]["final"],
        result["female"]["final"]
    ))

    return result

def normalize_distance(dist):
    return max(0, 100 - dist)

def normalize_similarity(sim):
    return sim * 100  # cosine similarity (0–1)