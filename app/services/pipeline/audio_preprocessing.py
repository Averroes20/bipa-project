import numpy as np
import librosa
import tempfile
from typing import Dict, Any, Union
from fastapi import HTTPException

class AudioPreprocessingService:
    @staticmethod
    def process(file: Union[Any, str]) -> Dict[str, Any]:
        """
        Loads audio, normalizes volume, trims long silence (VAD), 
        and extracts 16k and 22k Hz arrays.
        """
        try:
            if hasattr(file, "file"):
                data = file.file.read()
            elif hasattr(file, "read"):
                data = file.read()
            elif isinstance(file, str):
                with open(file, "rb") as f:
                    data = f.read()
            else:
                raise ValueError("Unsupported file format for audio loading")

            # Validate non-empty
            if not data or len(data) < 100:
                raise HTTPException(status_code=400, detail="Corrupted or empty audio file.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            # Load audio using librosa (handles stereo->mono and resampling automatically)
            # Load at 22050 for Prosody/Formants
            y_22k, sr_22k = librosa.load(tmp_path, sr=22050, mono=True)
            
            if len(y_22k) == 0 or np.all(y_22k == 0):
                 raise HTTPException(status_code=400, detail="Audio file contains no valid signal.")

            # Trim leading/trailing silence (VAD)
            y_22k_trimmed, index = librosa.effects.trim(y_22k, top_db=30)
            
            if len(y_22k_trimmed) == 0:
                raise HTTPException(status_code=400, detail="Audio file contains only silence.")

            # Loudness normalization (peak normalization to 0.9)
            peak = np.max(np.abs(y_22k_trimmed))
            if peak > 0:
                y_22k_norm = y_22k_trimmed * (0.9 / peak)
            else:
                y_22k_norm = y_22k_trimmed

            # Resample to 16000 for Whisper/Embeddings
            y_16k_norm = librosa.resample(y_22k_norm, orig_sr=22050, target_sr=16000)

            return {
                "audio_22k": y_22k_norm,
                "sr_22k": 22050,
                "audio_16k": y_16k_norm,
                "sr_16k": 16000,
                "temp_path": tmp_path
            }
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
