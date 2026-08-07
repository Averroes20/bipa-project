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

def indonesian_g2p(word: str) -> List[str]:
    word = word.lower()
    return list(word)

def align_transcription_to_target(transcribed_words: List[Dict], target_text: str) -> List[Dict]:
    """
    Aligns the Whisper transcribed words to the expected target_text using a simple 
    dynamic programming string matching approach (Needleman-Wunsch inspired) at the word level.
    """
    if not target_text.strip():
        return transcribed_words
        
    target_words = [re.sub(r'[^a-z]', '', w.lower()) for w in target_text.split() if re.sub(r'[^a-z]', '', w.lower())]
    if not target_words:
        return transcribed_words
        
    # Create DP table
    n = len(target_words)
    m = len(transcribed_words)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            t_word = target_words[i-1]
            s_word = transcribed_words[j-1]["word"]
            
            # Simple match cost: 0 if exact match, else Levenshtein-like distance between strings
            # For simplicity, just use 0 if exact, 0.5 if prefix/substring, 1 otherwise
            cost = 0 if t_word == s_word else (0.5 if (t_word in s_word or s_word in t_word) else 1)
            
            dp[i][j] = min(
                dp[i-1][j-1] + cost,    # substitution/match
                dp[i-1][j] + 1,         # deletion (target word missing in transcription)
                dp[i][j-1] + 1          # insertion (extra word in transcription)
            )
            
    # Backtrack
    i, j = n, m
    aligned_result = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            t_word = target_words[i-1]
            s_word = transcribed_words[j-1]["word"]
            cost = 0 if t_word == s_word else (0.5 if (t_word in s_word or s_word in t_word) else 1)
            
            if dp[i][j] == dp[i-1][j-1] + cost:
                # Match or substitution
                aligned = transcribed_words[j-1].copy()
                aligned["word"] = t_word
                aligned_result.append(aligned)
                i -= 1
                j -= 1
                continue
                
        if i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            # Deletion (Target word is missing in audio)
            aligned_result.append({
                "word": target_words[i-1],
                "start": 0.0,
                "end": 0.0,
                "confidence": 0.0,
                "missing": True
            })
            i -= 1
        else:
            # Insertion (Extra word in audio, ignore it for the final target mapping)
            j -= 1
            
    aligned_result.reverse()
    
    # Forward fill timestamps for missing words to make them look somewhat chronological
    last_end = 0.0
    for w in aligned_result:
        if w.get("missing"):
            w["start"] = last_end
            w["end"] = last_end + 0.1 # dummy short duration
        else:
            last_end = w["end"]
            
    return aligned_result

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
                
                confidence = min(1.0, max(0.0, prob))
                
                words_data.append({
                    "word": clean_word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": round(confidence, 2)
                })
                
        # Force alignment against target_text
        if target_text:
            words_data = align_transcription_to_target(words_data, target_text)
                
        return {
            "target_text": target_text,
            "normalized_target_text": " ".join([re.sub(r'[^a-z]', '', w.lower()) for w in target_text.split() if re.sub(r'[^a-z]', '', w.lower())]),
            "words": words_data
        }

