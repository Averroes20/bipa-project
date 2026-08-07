import asyncio
import concurrent.futures
from typing import Dict, Any, Union
from sqlalchemy.orm import Session
import numpy as np
from app.core.logger import logger
from app.repositories.analysis_repository import AnalysisRepository

from app.services.pipeline.audio_preprocessing import AudioPreprocessingService
from app.services.pipeline.pitch_service import PitchAnalysisService
from app.services.pipeline.feature_service import FeatureExtractionService

# New Pronunciation CAPT Services
from app.services.pronunciation.alignment_service import AlignmentService
from app.services.pronunciation.word_service import WordService
from app.services.pronunciation.phoneme_service import PhonemeService
from app.services.pronunciation.vowel_service import VowelService
from app.services.pronunciation.intonation_service import IntonationService
from app.services.pronunciation.accent_service import AccentService
from app.services.pronunciation.scoring_service import ScoringService
from app.services.pronunciation.feedback_service import FeedbackService

class PronunciationPipeline:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AnalysisRepository(db)
        
    async def evaluate(self, file: Union[Any, str], user_id: str, target_text: str = ""):
        """
        The main CAPT engine pipeline yielding SSE chunks.
        """
        import json
        async def send_progress(step: str, percent: int):
            return json.dumps({"status": "progress", "step": step, "progress": percent}) + "\n"

        yield await send_progress("Validating & Preprocessing Audio...", 10)
        audio_data = AudioPreprocessingService.process(file)
        audio_16k = audio_data["audio_16k"]
        audio_22k = audio_data["audio_22k"]
        sr_16k = audio_data["sr_16k"]
        sr_22k = audio_data["sr_22k"]
        temp_path = audio_data["temp_path"]
        
        yield await send_progress("Running Forced Alignment...", 25)
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            align_task = loop.run_in_executor(pool, AlignmentService.align, audio_16k, target_text)
            pitch_task = loop.run_in_executor(pool, PitchAnalysisService.analyze, audio_22k, sr_22k)
            
            align_data, pitch_stats = await asyncio.gather(align_task, pitch_task)
            
        words_boundary = align_data.get("words", [])
        final_target_text = align_data.get("target_text", target_text)
        duration_total = float(len(audio_22k) / sr_22k)
        
        yield await send_progress("Extracting Acoustic Features...", 40)
        features = FeatureExtractionService.extract_prosody_and_clarity(audio_22k, sr_22k, words_boundary)
        energy_contour = features.get("energy_contour", [])
        pitch_contour = pitch_stats.get("contour", [])
        
        yield await send_progress("Evaluating Pronunciation at Word & Phoneme Level...", 55)
        evaluated_words = WordService.evaluate_words(words_boundary, pitch_contour, energy_contour, duration_total)
        phonemes_data, mispronounced = PhonemeService.evaluate_phonemes(evaluated_words)
        
        native_sim = FeatureExtractionService.extract_native_similarity(audio_16k, sr_16k, self.repo)
        male_cand = native_sim.get("best_male_candidate") or {}
        female_cand = native_sim.get("best_female_candidate") or {}
        
        yield await send_progress("Comparing Vowel Space & Intonation...", 70)
        vowel_data = VowelService.extract_and_compare(temp_path, phonemes_data, male_cand, female_cand)
        
        m_pitch = male_cand.get("pitch_contour", [])
        f_pitch = female_cand.get("pitch_contour", [])
        
        intonation_data = IntonationService.compare_contours(pitch_contour, m_pitch, f_pitch)
        
        yield await send_progress("Calculating Final Scores & AI Feedback...", 85)
        accent_data = AccentService.analyze(evaluated_words)
        fluency_ratio = 1.0 - features.get("pause_ratio", 0.0)
        scoring = ScoringService.aggregate_scores(
            evaluated_words, phonemes_data, intonation_data, accent_data, vowel_data, fluency_ratio
        )
        feedback_text = FeedbackService.generate_feedback(
            evaluated_words, mispronounced, intonation_data, vowel_data, scoring
        )
        
        yield await send_progress("Saving Detailed Analysis...", 95)
        
        from app.utils.json_safe import to_python
        
        analysis = self.repo.save_analysis(
            user_id=user_id,
            user_features={
                "pitch_mean": pitch_stats.get("mean", 0.0),
                "pitch_range": pitch_stats.get("range", 0.0),
                "energy_mean": float(np.mean(energy_contour)) if energy_contour else 0.0,
                "duration": duration_total,
                "pause_ratio": features.get("pause_ratio", 0.0),
                "pitch": pitch_contour,
                "energy": energy_contour
            },
            score_result={
                "overall": scoring.get("overall_score", 0.0),
                "male": {"dtw": intonation_data.get("dtw_distance", 0.0) if intonation_data.get("preferred_reference") == "Male" else 0.0},
                "female": {"dtw": intonation_data.get("dtw_distance", 0.0) if intonation_data.get("preferred_reference") == "Female" else 0.0}
            },
            embedding_scores={
                "male": native_sim.get("male_score", 0.0),
                "female": native_sim.get("female_score", 0.0)
            },
            user_emb=native_sim["user_embedding"],
            ai_feedback=feedback_text,
            gender_label=native_sim["reference_gender"],
            analysis_detail=json.dumps(to_python(scoring))
        )
        
        analysis_id = str(analysis.id)
        
        from app.models.audio_models import AnalysisWord, AnalysisPhoneme, AnalysisPronunciation, AnalysisIntonation, AnalysisFeedback
        
        word_db_objects = {}
        for w in evaluated_words:
            aw = AnalysisWord(
                analysis_id=analysis_id,
                word=w["word"],
                start_time=float(w["start_time"]),
                end_time=float(w["end_time"]),
                confidence=float(w["confidence"]),
                pronunciation_score=float(w["pronunciation_score"]),
                pitch_score=float(w["pitch_score"]),
                energy_score=float(w["energy_score"]),
                duration_score=float(w["duration_score"]),
                stress_score=float(w["stress_score"]),
                overall_score=float(w["overall_score"])
            )
            self.db.add(aw)
            self.db.flush()
            word_db_objects[w["word"]] = aw.id
            
        for p in phonemes_data:
            ap = AnalysisPhoneme(
                analysis_id=analysis_id,
                word_id=word_db_objects.get(p["word_ref"]),
                phoneme=p["phoneme"],
                start_time=float(p["start_time"]),
                end_time=float(p["end_time"]),
                confidence=float(p["confidence"]),
                pronunciation_score=float(p["pronunciation_score"]),
                feedback=p["feedback"]
            )
            self.db.add(ap)
            
        for m in mispronounced:
            am = AnalysisPronunciation(
                analysis_id=analysis_id,
                mispronounced_word=m["word"],
                score=float(m["score"]),
                reason=m["reason"]
            )
            self.db.add(am)
            
        ai = AnalysisIntonation(
            analysis_id=analysis_id,
            similarity_score=float(intonation_data.get("similarity_score", 0.0)),
            dtw_distance=float(intonation_data.get("dtw_distance", 0.0)),
            correlation=float(intonation_data.get("correlation", 0.0)),
            slope_diff=0.0,
            user_contour=json.dumps(to_python(intonation_data.get("user_contour", []))),
            native_contour=json.dumps(to_python(
                intonation_data.get("female_contour") if intonation_data.get("preferred_reference") == "Female" else intonation_data.get("male_contour", [])
            ))
        )
        self.db.add(ai)
        
        af = AnalysisFeedback(
            analysis_id=analysis_id,
            ai_teacher_feedback=feedback_text
        )
        self.db.add(af)
        
        self.db.commit()
        
        # Ensure accent_data is a dict (if AccentService is mocked to return float or list)
        if not isinstance(accent_data, dict):
            accent_data = {"accent_classification": "Neutral"}

        result_payload = {
            "id": analysis_id,
            "overall_score": scoring.get("overall_score", 0.0),
            "dimensions": {
                "intonation": scoring.get("intonation_score", 0.0),
                "pronunciation": scoring.get("pronunciation_score", 0.0),
                "fluency": scoring.get("fluency_score", 0.0),
                "clarity": scoring.get("word_score", 0.0),
                "accent": scoring.get("accent_score", 0.0)
            },
            "similarity": {
                "male": float(native_sim.get("male_score", 0.0)),
                "female": float(native_sim.get("female_score", 0.0))
            },
            "voice_profile": native_sim.get("reference_gender", "Unknown"),
            "pronunciation": {
                "transcription": final_target_text,
                "words": [{"word": w["word"], "start": float(w["start_time"]), "end": float(w["end_time"]), "duration": float(w["end_time"]) - float(w["start_time"]), "confidence": float(w["confidence"]), "score": float(w["pronunciation_score"]), "status": next((m["status"] for m in mispronounced if m["word"] == w["word"]), "correct")} for w in evaluated_words],
                "phonemes": [{"symbol": p["phoneme"], "start": float(p["start_time"]), "end": float(p["end_time"]), "duration": float(p["end_time"]) - float(p["start_time"]), "confidence": float(p["confidence"]), "score": float(p["pronunciation_score"])} for p in phonemes_data],
                "pronunciation_score": scoring.get("pronunciation_score", 0.0),
                "word_score": scoring.get("word_score", 0.0),
                "phoneme_score": scoring.get("phoneme_score", 0.0),
                "errors": mispronounced
            },
            "pitch": {
                "mean": float(pitch_stats.get("mean", 0.0)),
                "range": float(pitch_stats.get("range", 0.0)),
                "contour": [float(c) for c in pitch_contour]
            },
            "energy": {
                "mean": float(np.mean(energy_contour)) if energy_contour else 0.0,
                "contour": [float(e) for e in energy_contour]
            },
            "pause": {
                "ratio": float(features.get("pause_ratio", 0.0)),
                "timeline": features.get("pause_timeline", [])
            },
            "phonetics": {
                "vowel_space": vowel_data.get("user_space", []),
                "native_male_space": vowel_data.get("native_male_space", {}),
                "native_female_space": vowel_data.get("native_female_space", {}),
                "vowels": [{"vowel": v["vowel"], "accuracy": float(v.get("accuracy", 0.0))} for v in vowel_data.get("user_space", [])]
            },
            "articulation": {
                "zcr": float(features.get("zcr_mean", 0.0)),
                "spectral_centroid": float(features.get("centroid_mean", 0.0)),
                "spectral_bandwidth": float(features.get("bandwidth_mean", 0.0)),
                "spectral_contrast": float(features.get("contrast_mean", 0.0)),
                "speech_clarity": float(features.get("clarity_score", 0.0))
            },
            "accent": {
                "speaking_rate_wpm": float(features.get("wpm", 0.0)),
                "rhythm_variance": float(scoring.get("rhythm_score", 0.0)),
                "stress_density": 0.0,
                "pitch_variance": float(intonation_data.get("pitch_variance", 0.0)),
                "pause_ratio": float(features.get("pause_ratio", 0.0)),
                "accent_classification": accent_data.get("accent_classification", "Neutral")
            },
            "intonation": {
                "sentence_ending": intonation_data.get("sentence_ending", "neutral"),
                "pattern": intonation_data.get("pattern", "neutral"),
                "pitch_variance": float(intonation_data.get("pitch_variance", 0.0)),
                "similarity_score": float(intonation_data.get("similarity_score", 0.0)),
                "user_contour": intonation_data.get("user_contour", []),
                "male_contour": intonation_data.get("male_contour", []),
                "female_contour": intonation_data.get("female_contour", []),
                "male_similarity": float(intonation_data.get("male_similarity", 0.0)),
                "female_similarity": float(intonation_data.get("female_similarity", 0.0)),
                "preferred_reference": intonation_data.get("preferred_reference", "Unknown")
            },
            "phoneme_detection": {
                "critical_phonemes_accuracy": scoring.get("phoneme_score", 0.0),
                "details": phonemes_data
            },
            "vowel_analysis": {
                "vowels": vowel_data.get("user_space", []),
                "f1_mean": float(vowel_data.get("f1_mean", 0)),
                "f2_mean": float(vowel_data.get("f2_mean", 0)),
                "f3_mean": float(vowel_data.get("f3_mean", 0)),
                "vsa": float(vowel_data.get("vsa", 0))
            },
            "errors": mispronounced,
            "recommendation": [
                {"type": "Strength", "message": "Your overall attempt was recorded successfully."},
                {"type": "Feedback", "message": feedback_text}
            ],
            "analysisMetadata": {"processed": True}
        }
        
        yield json.dumps({"status": "complete", "result": result_payload}) + "\n"
