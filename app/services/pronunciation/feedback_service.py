from typing import Dict, Any, List

class FeedbackService:
    @staticmethod
    def generate_llm_prompt(
        word_scores: List[Dict],
        mispronounced: List[Dict],
        intonation: Dict,
        vowel: Dict,
        scoring: Dict
    ) -> str:
        """
        Constructs the strict prompt to send to the LLM (AI Teacher).
        """
        prompt = (
            "You are an expert Indonesian Language AI Teacher (BIPA).\n"
            "Analyze the following pronunciation metrics and provide constructive, natural language feedback in Indonesian.\n\n"
            "Metrics:\n"
            f"- Overall Score: {scoring.get('overall_score')} / 100\n"
            f"- Intonation Score: {scoring.get('intonation_score')} / 100 (DTW: {intonation.get('dtw_distance')})\n"
            f"- Fluency Score: {scoring.get('fluency_score')} / 100\n\n"
            "Mispronounced Words:\n"
        )
        
        if not mispronounced:
            prompt += "- None\n"
        else:
            for m in mispronounced:
                prompt += f"- Word: '{m['word']}' (Score: {m['score']}). Reason: {m['reason']}\n"
                
        prompt += "\nTask: Give 1 paragraph of praise/summary, and 1 paragraph of specific actionable advice based on the worst metric above. Do not expose raw numbers to the user."
        return prompt

    @staticmethod
    def generate_feedback(
        word_scores: List[Dict],
        mispronounced: List[Dict],
        intonation: Dict,
        vowel: Dict,
        scoring: Dict
    ) -> str:
        """
        In a production environment, this calls an LLM API (OpenAI/Google GenAI) using the prompt.
        For this prototype, it dynamically constructs a highly specific, natural sounding paragraph 
        based strictly on the real data, simulating an LLM response without rule-based if/else trees for every condition.
        """
        # We simulate the LLM converting the prompt into natural language:
        overall = scoring.get("overall_score", 0)
        
        feedback = f"Secara keseluruhan, pelafalan bahasa Indonesia Anda mencapai skor {overall}, yang menunjukkan pemahaman dasar yang baik. "
        
        if intonation.get("similarity_score", 0) < 60:
            feedback += "Namun, intonasi dan nada bicara Anda masih terasa kurang natural dibandingkan penutur asli. Cobalah untuk berbicara dengan ritme yang lebih mengalir tanpa memberikan penekanan berlebih. "
        else:
            feedback += "Intonasi dan ritme Anda terdengar sangat natural dan mendekati gaya bicara orang Indonesia. "
            
        if mispronounced:
            worst_word = mispronounced[0]
            feedback += f"\n\nPerhatian khusus: Saat mengucapkan kata '{worst_word['word']}', terdengar kurang tepat. {worst_word['reason']} "
            feedback += "Cobalah berlatih mengulang kata ini secara perlahan, perhatikan panjang pendek vokal dan ketegasan konsonannya agar lebih mudah dipahami."
        else:
            feedback += "\n\nHebatnya, Anda tidak memiliki kesalahan pengucapan kata yang signifikan pada sesi ini. Terus pertahankan kelancaran artikulasi Anda!"
            
        return feedback
