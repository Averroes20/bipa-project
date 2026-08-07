from typing import Dict, Any, List
from app.services.pipeline.vowel_service import VowelAnalysisService

class CorpusFormantPipeline:
    @staticmethod
    def extract_formants(temp_path: str, phonemes_data: List[Dict]) -> Dict[str, Any]:
        """
        Extracts F1, F2, F3 and overall vowel profile.
        """
        vowel_data = VowelAnalysisService.extract_vowels(temp_path, phonemes_data)
        
        # We need generic F1, F2, F3 averages if available, else 0
        vowel_space = vowel_data.get("vowelSpace", [])
        f1_list = [v.get("F1", 0) for v in vowel_space if v.get("F1")]
        f2_list = [v.get("F2", 0) for v in vowel_space if v.get("F2")]
        f3_list = [v.get("F3", 0) for v in vowel_space if v.get("F3")]
        
        f1_avg = sum(f1_list) / len(f1_list) if f1_list else 0.0
        f2_avg = sum(f2_list) / len(f2_list) if f2_list else 0.0
        f3_avg = sum(f3_list) / len(f3_list) if f3_list else 0.0
        
        return {
            "f1": f1_avg,
            "f2": f2_avg,
            "f3": f3_avg,
            "vowel_profile": vowel_space
        }
