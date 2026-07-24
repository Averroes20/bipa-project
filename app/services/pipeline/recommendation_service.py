from typing import Dict, Any, List

class RecommendationService:
    @staticmethod
    def generate(scores: Dict, features: Dict, errors: List[Dict], target_text: str) -> List[Dict]:
        recs = []
        
        # Strengths
        if scores["fluency"] >= 75:
            recs.append({"type": "Strength", "message": "✓ Stable speech rhythm and pace"})
        if scores["intonation"] >= 75:
            recs.append({"type": "Strength", "message": "✓ Good natural intonation"})
            
        if not errors and scores["pronunciation"] >= 80:
             recs.append({"type": "Strength", "message": "✓ Excellent articulation across all words"})
             
        # Needs Improvement & Exercises
        if errors:
            # Take the most severe/first error
            err = errors[0]
            if err["type"] == "Substitution":
                recs.append({
                    "type": "Needs Improvement", 
                    "message": f"• '{err['expected']}' was mispronounced as '{err['detected']}'"
                })
                recs.append({
                    "type": "Exercises",
                    "message": f"Practice\n\"{err['expected']}\"\n5 repetitions.\nFocus on precise articulation."
                })
            elif err["type"] == "Deletion":
                recs.append({
                    "type": "Needs Improvement", 
                    "message": f"• Skipped or mumbled the word '{err['expected']}'"
                })
                recs.append({
                    "type": "Exercises",
                    "message": f"Practice\n\"{err['expected']}\"\n5 repetitions.\nEnsure you do not swallow syllables."
                })
            else:
                 recs.append({
                    "type": "Needs Improvement", 
                    "message": f"• Pronunciation of '{err['expected'] or err['detected']}' was unclear"
                })
                 recs.append({
                    "type": "Exercises",
                    "message": f"Practice\n\"{err['expected'] or target_text}\"\n5 repetitions slowly."
                })
        else:
            if scores["fluency"] < 60:
                recs.append({
                    "type": "Needs Improvement", 
                    "message": "• Frequent or long pauses detected"
                })
                recs.append({
                    "type": "Exercises",
                    "message": f"Practice\n\"{target_text}\"\nFocus on reading smoothly without stopping."
                })
            elif scores["intonation"] < 60:
                recs.append({
                    "type": "Needs Improvement", 
                    "message": "• Pitch variation is too flat or erratic"
                })
                recs.append({
                    "type": "Exercises",
                    "message": f"Practice\n\"{target_text}\"\nListen to native speakers and mimic their melody."
                })
            else:
                recs.append({
                    "type": "Exercises",
                    "message": "Excellent work! Try practicing longer and more complex sentences next."
                })
                
        return recs
