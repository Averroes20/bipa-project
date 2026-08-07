import re
from typing import Dict, Any, List
import whisper
import numpy as np
from functools import lru_cache
from app.core.logger import logger

@lru_cache(maxsize=1)
def get_whisper_model(size="base"):
    logger.info(f"Loading Whisper model '{size}'...")
    return whisper.load_model(size)

# Fallback G2P for missing imports if any
def indonesian_g2p(word: str) -> List[str]:
    word = word.lower()
    return list(word)

class AlignmentService:
    @staticmethod
    def align(audio: np.ndarray, target_text: str = "") -> Dict[str, Any]:
        """
        Uses Whisper to force align audio with words and extract boundaries.
        Returns:
            {
                "words": [
                    {"word": "selamat", "start": 0.21, "end": 0.74, "confidence": 0.98},
                    ...
                ]
            }
        """
        w_model = get_whisper_model("base")
        
        # We can pass initial_prompt to guide Whisper if target_text is provided
        options = {"language": "id", "word_timestamps": True}
        if target_text:
            options["initial_prompt"] = target_text

        result = w_model.transcribe(audio, **options)
        
        words_data = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_text = word_info["word"].strip().lower()
                clean_word = re.sub(r'[^a-z]', '', word_text)
                if not clean_word:
                    continue
                    
                start = word_info["start"]
                end = word_info["end"]
                prob = word_info.get("probability", 0.5)
                
                # Confidence scaling
                confidence = min(1.0, max(0.0, prob))
                
                words_data.append({
                    "word": clean_word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": round(confidence, 2)
                })
                
        return {
            "words": words_data
        }
