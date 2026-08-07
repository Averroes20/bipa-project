from typing import Dict, Any, List
import numpy as np

class IntonationAnalysisService:
    @staticmethod
    def analyze(pitch_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes pitch contour to detect sentence ending and pattern (Question vs Statement).
        """
        contour = pitch_stats.get("contour", [])
        if not contour or len(contour) < 5:
            return {
                "sentence_ending": "Flat",
                "pattern": "Statement",
                "pitch_variance": pitch_stats.get("variance", 0.0)
            }
            
        # Get the last 20% of the pitch contour to determine sentence ending
        tail_len = max(3, int(len(contour) * 0.2))
        tail = contour[-tail_len:]
        
        # Calculate slope of the tail
        x = np.arange(len(tail))
        slope, _ = np.polyfit(x, tail, 1)
        
        # A positive slope at the end usually indicates a question or continuation
        # A negative slope indicates a statement
        
        if slope > 2.0:
            ending = "Rising"
            pattern = "Question"
        elif slope < -2.0:
            ending = "Falling"
            pattern = "Statement"
        else:
            ending = "Flat"
            pattern = "Statement"
            
        return {
            "sentence_ending": ending,
            "pattern": pattern,
            "pitch_variance": pitch_stats.get("variance", 0.0)
        }
