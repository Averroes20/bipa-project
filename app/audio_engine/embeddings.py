import re
import numpy as np
import torch
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity

_processor = None
_model = None

def get_wav2vec2():
    global _processor, _model
    if _processor is None or _model is None:
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            _model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        except Exception as e:
            print("Wav2Vec2 loading error:", e)
            return None, None
    return _processor, _model

def extract_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract Wav2Vec2 mean hidden state embedding."""
    proc, mod = get_wav2vec2()
    if proc is None or mod is None:
        # Fallback dummy embedding if transformers model is unavailable
        return np.zeros(768, dtype=float)

    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    inputs = proc(audio_tensor, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = mod(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def compute_embedding_score(user_emb: np.ndarray, candidates: List[Any]) -> float:
    """Compute average cosine similarity between user embedding and candidate dataset embeddings."""
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
            raw_emb = c.embedding_vector if hasattr(c, "embedding_vector") else c[0]
            emb = parse_embedding(raw_emb)
            emb_vec = clean_vector(emb).reshape(1, -1)

            if emb_vec.shape[1] != user_vec.shape[1]:
                continue

            sim = cosine_similarity(user_vec, emb_vec)[0][0]
            scores.append(float(sim))
        except Exception:
            continue

    return float(sum(scores) / len(scores)) if scores else 0.0
