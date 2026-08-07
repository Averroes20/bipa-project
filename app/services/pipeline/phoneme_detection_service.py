from typing import Dict, Any, List

class PhonemeDetectionService:
    CRITICAL_PHONEMES = ['r', 'ng', 'sy', 'kh', 'c', 'j']

    @staticmethod
    def analyze(phoneme_errors: List[Dict]) -> Dict[str, Any]:
        """
        Filters and scores specific critical phonemes for Indonesian learners.
        """
        critical_stats = {
            ph: {"count": 0, "correct": 0, "errors": []}
            for ph in PhonemeDetectionService.CRITICAL_PHONEMES
        }
        
        for p in phoneme_errors:
            expected = p.get("expected", "").lower()
            detected = p.get("detected", "").lower()
            
            # For each critical phoneme, check if it was expected
            if expected in critical_stats:
                critical_stats[expected]["count"] += 1
                if p.get("is_correct", False):
                    critical_stats[expected]["correct"] += 1
                else:
                    critical_stats[expected]["errors"].append({
                        "detected": detected,
                        "confidence": p.get("confidence", 0.0)
                    })
                    
        # Calculate scores
        results = {}
        overall_critical_acc = 0.0
        total_critical = 0
        
        for ph, stats in critical_stats.items():
            if stats["count"] > 0:
                acc = (stats["correct"] / stats["count"]) * 100
                total_critical += 1
                overall_critical_acc += acc
                results[ph] = {
                    "accuracy": round(acc, 1),
                    "occurrences": stats["count"],
                    "common_mistake": stats["errors"][0]["detected"] if stats["errors"] else None
                }
                
        if total_critical > 0:
            avg_acc = overall_critical_acc / total_critical
        else:
            avg_acc = 100.0
            
        return {
            "critical_phonemes_accuracy": round(avg_acc, 1),
            "details": results
        }
