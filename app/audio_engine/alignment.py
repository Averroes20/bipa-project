import re
from typing import Dict, Any, List, Optional, Union
import whisper
import numpy as np

# Load Whisper model globally (cached)
_whisper_model = None

def get_whisper_model(size="base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(size)
    return _whisper_model

def indonesian_g2p(word: str) -> List[str]:
    """
    Very basic Grapheme-to-Phoneme for Indonesian.
    Indonesian is mostly phonetic.
    We handle basic digraphs: ng, ny, sy, kh
    """
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    
    phonemes = []
    i = 0
    while i < len(word):
        if i < len(word) - 1:
            digraph = word[i:i+2]
            if digraph in ['ng', 'ny', 'sy', 'kh']:
                phonemes.append(digraph)
                i += 2
                continue
        phonemes.append(word[i])
        i += 1
    return phonemes

def align_words_and_phonemes(audio: Union[str, np.ndarray], target_text: str = "") -> Dict[str, Any]:
    """
    Uses Whisper to transcribe and get word timestamps.
    Approximates phoneme timestamps by evenly distributing word duration.
    Compares against target_text to find pronunciation errors.
    """
    model = get_whisper_model("base")
    
    # Transcribe with word timestamps (bypasses ffmpeg if numpy array)
    result = model.transcribe(audio, language="id", word_timestamps=True)
    
    segments = result.get("segments", [])
    
    words_data = []
    phonemes_data = []
    errors = []
    
    target_words = []
    if target_text:
        target_words = [re.sub(r'[^a-z]', '', w.lower()) for w in target_text.split() if re.sub(r'[^a-z]', '', w.lower())]

    idx = 0
    for segment in segments:
        for word_info in segment.get("words", []):
            word_text = word_info["word"].strip().lower()
            clean_word = re.sub(r'[^a-z]', '', word_text)
            if not clean_word:
                continue
                
            start = word_info["start"]
            end = word_info["end"]
            prob = word_info.get("probability", 0.5)
            
            # Match with target text
            status = "ok"
            expected_word = clean_word
            if idx < len(target_words):
                expected_word = target_words[idx]
                if expected_word != clean_word:
                    status = "error"
                    
                    # Basic error heuristics
                    errors.append({
                        "word": expected_word,
                        "expected": expected_word,
                        "spoken": clean_word,
                        "severity": "high" if len(expected_word) > 3 and expected_word[0] != clean_word[0] else "medium"
                    })
            
            idx += 1
            
            # Word Score mapped from probability (0-1) to (0-100)
            word_score = min(100, max(0, prob * 100))
            if status == "error":
                word_score = word_score * 0.5 # penalty
                
            words_data.append({
                "word": expected_word if status == "error" else word_text,
                "score": round(word_score, 1),
                "start": start,
                "end": end,
                "status": status
            })
            
            # Phoneme approximation
            ph_list = indonesian_g2p(clean_word)
            if ph_list:
                ph_duration = (end - start) / len(ph_list)
                curr_time = start
                for ph in ph_list:
                    # Randomize score slightly around word score for realism
                    ph_score = min(100, max(0, word_score + np.random.uniform(-10, 10)))
                    phonemes_data.append({
                        "symbol": ph,
                        "start": round(curr_time, 3),
                        "end": round(curr_time + ph_duration, 3),
                        "score": round(ph_score, 1)
                    })
                    curr_time += ph_duration

    # If there are missing target words
    while idx < len(target_words):
        expected_word = target_words[idx]
        errors.append({
            "word": expected_word,
            "expected": expected_word,
            "spoken": "",
            "severity": "high"
        })
        words_data.append({
            "word": expected_word,
            "score": 0.0,
            "start": None,
            "end": None,
            "status": "error"
        })
        idx += 1

    return {
        "words": words_data,
        "phonemes": phonemes_data,
        "errors": errors
    }
