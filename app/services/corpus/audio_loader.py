import os
from typing import Dict, Any
from app.services.pipeline.audio_preprocessing import AudioPreprocessingService

class CorpusAudioLoader:
    @staticmethod
    def load(file_path: str) -> Dict[str, Any]:
        """
        Reuses AudioPreprocessingService to load audio, perform noise reduction, 
        and prepare 16k, 22k versions and temp path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        return AudioPreprocessingService.process(file_path)
