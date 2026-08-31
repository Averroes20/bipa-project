from typing import Dict, Any, List

class FeedbackService:
    @staticmethod
    def generate_feedback(
        word_scores: List[Dict],
        mispronounced: List[Dict],
        intonation: Dict,
        vowel: Dict,
        scoring: Dict,
        accent: Dict = None
    ) -> Dict[str, List[str]]:
        """
        Generates specific, rule-based AI Teacher feedback with 3 sections.
        """
        strengths = []
        focus = []
        practice = []
        
        # 1. Strengths
        if scoring.get("fluency_score", 0) > 80:
            strengths.append(f"Fluency sangat stabil ({scoring['fluency_score']}).")
        else:
            if scoring.get("overall_score", 0) > 75:
                strengths.append(f"Pengucapan secara keseluruhan cukup jelas dengan skor {scoring.get('overall_score')}.")
                
        if accent and accent.get("speaking_rate_wpm", 0) > 130:
            wpm = round(accent.get("speaking_rate_wpm", 0))
            strengths.append(f"Tempo bicara berada dalam rentang native ({wpm} WPM).")
            
        if intonation.get("similarity_score", 0) > 85:
            strengths.append(f"Pola intonasi sangat mirip dengan native speaker ({intonation['similarity_score']}% kecocokan).")
            
        if not strengths:
            strengths.append("Anda telah berhasil menyelesaikan sesi analisis ini.")

        # 2. Focus Next
        if mispronounced:
            worst_word = mispronounced[0]
            focus.append(f"Latih kata '{worst_word['word']}' karena skornya masih {worst_word.get('score', 0)}.")
            
            # Additional phonetic errors if any
            if "reason" in worst_word and "Kesalahan fonem" in worst_word["reason"]:
                focus.append(f"Perhatikan detail fonem pada kata '{worst_word['word']}': {worst_word['reason'].split('Kesalahan fonem: ')[-1]}")
        elif scoring.get("intonation_score", 0) < 75:
            focus.append(f"Intonasi Anda (skor {scoring.get('intonation_score')}) perlu lebih bervariasi agar tidak terdengar kaku.")
        else:
            focus.append("Pengucapan Anda sangat baik, fokuslah pada mempertahankan ritme bicara Anda.")

        if vowel and "vowels" in vowel:
            vowels_list = vowel["vowels"]
            if vowels_list:
                worst_vowel = min(vowels_list, key=lambda x: x.get("match", 100))
                if worst_vowel.get("match", 100) < 80:
                    focus.append(f"Bunyi vokal /{worst_vowel['phoneme']}/ masih kurang tepat ({worst_vowel['match']}% kecocokan).")
                    
        # 3. Practice Now
        if mispronounced:
            worst_word = mispronounced[0]
            practice.append(f"Ulangi kata '{worst_word['word']}' sebanyak 5 kali sambil mendengarkan referensi Native.")
        else:
            practice.append("Latihlah kalimat ini secara utuh tanpa terputus.")
            
        practice.append("Bandingkan pelafalan Anda dengan Native Female/Male menggunakan fitur Playback Comparison.")

        return {
            "strengths": strengths,
            "focus": focus,
            "practice": practice
        }
