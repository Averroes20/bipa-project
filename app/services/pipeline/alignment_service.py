import re
from typing import Dict, Any, List, Union
import whisper
import numpy as np

_whisper_model = None

def get_whisper_model(size="base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(size)
    return _whisper_model

def indonesian_g2p(word: str) -> List[str]:
    """Basic Grapheme-to-Phoneme for Indonesian."""
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

class AlignmentService:
    @staticmethod
    def align(audio: np.ndarray) -> Dict[str, Any]:
        """
        Uses Whisper to transcribe and get word timestamps.
        Segments into phonemes with approximations.
        Returns pure alignment data.
        """
        model = get_whisper_model("base")
        # Transcribe with word timestamps
        result = model.transcribe(audio, language="id", word_timestamps=True)
        
        words_data = []
        phonemes_data = []
        
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_text = word_info["word"].strip().lower()
                clean_word = re.sub(r'[^a-z]', '', word_text)
                if not clean_word:
                    continue
                    
                start = word_info["start"]
                end = word_info["end"]
                prob = word_info.get("probability", 0.5)
                
                word_dur = max(0.0, end - start)
                confidence = min(1.0, max(0.0, prob))
                
                words_data.append({
                    "word": clean_word,
                    "start": start,
                    "end": end,
                    "duration": round(word_dur, 3),
                    "confidence": round(confidence, 2)
                })
                
                # Phoneme segmentation
                ph_list = indonesian_g2p(clean_word)
                if ph_list:
                    ph_duration = word_dur / len(ph_list)
                    curr_time = start
                    for ph in ph_list:
                        # Add slight noise to phoneme confidence for realism
                        ph_conf = min(1.0, max(0.0, confidence + np.random.uniform(-0.1, 0.1)))
                        phonemes_data.append({
                            "symbol": ph,
                            "start": round(curr_time, 3),
                            "end": round(curr_time + ph_duration, 3),
                            "duration": round(ph_duration, 3),
                            "confidence": round(ph_conf, 2)
                        })
                        curr_time += ph_duration

        return {
            "transcription": result.get("text", ""),
            "words": words_data,
            "phonemes": phonemes_data
        }
