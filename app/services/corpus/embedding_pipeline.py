import numpy as np
from typing import Dict, Any
from app.audio_engine.embeddings import extract_embedding

class CorpusEmbeddingPipeline:
    @staticmethod
    def extract_embedding(audio_16k: np.ndarray, sr_16k: int) -> str:
        """
        Extracts speech embedding vector using existing neural engine.
        Returns string representation for pgvector compatibility.
        """
        emb = extract_embedding(audio_16k, sr_16k)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
        return str(emb_list)
