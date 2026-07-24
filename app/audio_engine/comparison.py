import numpy as np
from typing import Dict, Any, List

def dtw_distance(seq1: List[float], seq2: List[float]) -> float:
    """Compute DTW distance between 2 sequences using fastdtw if available or Euclidean DTW fallback."""
    try:
        from fastdtw import fastdtw
        distance, _ = fastdtw(seq1, seq2)
        return float(distance)
    except Exception:
        # Simple L1 distance fallback if fastdtw is unavailable
        l1 = min(len(seq1), len(seq2))
        if l1 == 0:
            return 0.0
        return float(np.sum(np.abs(np.array(seq1[:l1]) - np.array(seq2[:l1]))))

def compare_features(user: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    """Compare user features against male and female candidates."""
    result: Dict[str, List[Dict[str, float]]] = {"male": [], "female": []}

    for gender in ["male", "female"]:
        candidates = dataset.get(gender, [])
        for ref in candidates:
            # ref can be SQLAlchemy Row or dict
            ref_dict = ref._asdict() if hasattr(ref, "_asdict") else (dict(ref) if hasattr(ref, "keys") else {})
            ref_pitch = ref_dict.get("pitch", [0.0])
            ref_energy = ref_dict.get("energy", [0.0])
            ref_pause = ref_dict.get("pause_ratio", 0.0)

            pitch_dist = dtw_distance(user.get("pitch", []), ref_pitch if isinstance(ref_pitch, list) else [])
            energy_dist = dtw_distance(user.get("energy", []), ref_energy if isinstance(ref_energy, list) else [])
            pause_dist = abs(user.get("pause_ratio", 0.0) - ref_pause)

            result[gender].append({
                "pitch": pitch_dist,
                "energy": energy_dist,
                "pause": pause_dist
            })

    return result

def normalize_distance(dist: float) -> float:
    return max(0.0, 100.0 - dist)

def normalize_similarity(sim: float) -> float:
    return sim * 100.0  # Cosine similarity (0–1 to 0-100)

def compute_score(comparison_result: Dict[str, Any], embedding_scores: Dict[str, float]) -> Dict[str, Any]:
    """Computes DTW, embedding, and overall final scores across genders."""
    result: Dict[str, Any] = {}

    for gender in ["male", "female"]:
        items = comparison_result.get(gender, [])
        if items:
            pitch = float(np.mean([d["pitch"] for d in items]))
            energy = float(np.mean([d["energy"] for d in items]))
            pause = float(np.mean([d["pause"] for d in items]))
        else:
            pitch, energy, pause = 0.0, 0.0, 0.0

        dtw_score = float(np.mean([
            normalize_distance(pitch),
            normalize_distance(energy),
            normalize_distance(pause)
        ]))

        emb_sim = embedding_scores.get(gender, 0.0)
        emb_score = float(normalize_similarity(emb_sim))

        final_score = float(0.4 * dtw_score + 0.6 * emb_score)

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
