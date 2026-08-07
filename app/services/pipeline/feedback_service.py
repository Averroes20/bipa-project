from typing import Dict, Any, List

class FeedbackService:
    @staticmethod
    def generate(
        pronunciation: Dict[str, Any],
        vowel: Dict[str, Any],
        articulation: Dict[str, Any],
        accent: Dict[str, Any],
        intonation: Dict[str, Any],
        phoneme_det: Dict[str, Any]
    ) -> List[str]:
        """
        Generates natural language insights combining all features.
        """
        insights = []
        
        # Pronunciation & Articulation Insight
        clarity = articulation.get("speech_clarity", 0)
        pron_score = pronunciation.get("pronunciation_score", 0)
        if pron_score > 80 and clarity > 80:
            insights.append("Pengucapan Anda sangat jelas dan mudah dipahami, artikulasi sudah setara penutur asli.")
        elif pron_score > 60:
            insights.append("Pengucapan Anda cukup baik, namun pada beberapa kata artikulasi terdengar kurang tajam.")
        else:
            insights.append("Artikulasi Anda masih kurang jelas. Cobalah untuk membuka mulut lebih lebar saat melafalkan kata-kata.")
            
        # Vowel Insight
        vsa = vowel.get("vsa", 0)
        if vsa < 10000:
            insights.append("Ruang vokal (Vowel Space) Anda sempit, yang berarti perbedaan antara huruf vokal (a, i, u) kurang kontras.")
        else:
            insights.append("Pembedaan huruf vokal Anda sudah sangat tegas.")
            
        # Accent & Intonation Insight
        acc_class = accent.get("accent_classification", "Unknown")
        pattern = intonation.get("pattern", "Statement")
        ending = intonation.get("sentence_ending", "Flat")
        
        if acc_class == "Native Indonesia":
            insights.append(f"Aksen dan ritme Anda terdengar sangat natural seperti orang Indonesia, dengan pola kalimat berupa {pattern.lower()} ({ending.lower()}).")
        elif acc_class == "English":
            insights.append("Irama bicara Anda cenderung memiliki penekanan (stress-timed) gaya bahasa Inggris. Dalam bahasa Indonesia, cobalah untuk lebih datar dan seragam antar suku kata.")
        elif acc_class == "Mandarin":
            insights.append("Terdapat fluktuasi nada yang kuat pada suku kata Anda (ciri khas nada Mandarin). Bahasa Indonesia tidak menggunakan nada (toneless), cobalah berbicara lebih mengalir.")
        
        # Phoneme Insight
        crit_acc = phoneme_det.get("critical_phonemes_accuracy", 100)
        details = phoneme_det.get("details", {})
        mistakes = []
        for ph, stats in details.items():
            if stats["accuracy"] < 70 and stats["common_mistake"]:
                mistakes.append(f"'{ph}' (terdengar seperti '{stats['common_mistake']}')")
                
        if mistakes:
            insights.append(f"Perhatikan pelafalan konsonan berikut: {', '.join(mistakes)}.")
        elif crit_acc > 85:
            insights.append("Konsonan kritis bahasa Indonesia (seperti R, NG, KH, dll) dilafalkan dengan sangat baik.")
            
        return insights
