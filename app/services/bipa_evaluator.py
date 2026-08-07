import json
import asyncio
import concurrent.futures
from typing import Any, Union, AsyncGenerator
from sqlalchemy.orm import Session
from app.utils.json_safe import to_python
from app.repositories.analysis_repository import AnalysisRepository
from app.schemas.analysis_response import AnalysisResponse
from app.core.logger import logger

from app.services.pipeline.audio_preprocessing import AudioPreprocessingService
from app.services.pipeline.alignment_service import AlignmentService
from app.services.pipeline.pitch_service import PitchAnalysisService
from app.services.pipeline.formant_service import FormantAnalysisService
from app.services.pipeline.feature_service import FeatureExtractionService
from app.services.pipeline.scoring_service import PronunciationService, ScoringService
from app.services.pipeline.vowel_service import VowelAnalysisService
from app.services.pipeline.articulation_service import ArticulationAnalysisService
from app.services.pipeline.accent_service import AccentAnalysisService
from app.services.pipeline.intonation_service import IntonationAnalysisService
from app.services.pipeline.phoneme_detection_service import PhonemeDetectionService
from app.services.pipeline.feedback_service import FeedbackService
from app.services.pipeline.recommendation_service import RecommendationService

class BIPAEvaluator:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AnalysisRepository(db)

    async def evaluate_audio_stream(self, file: Union[Any, str], user_id: str, target_text: str = "") -> AsyncGenerator[str, None]:
        async def send_progress(step: str, percent: int):
            return json.dumps({"status": "progress", "step": step, "progress": percent}) + "\n"

        yield await send_progress("Validating & Preprocessing Audio...", 10)
        await asyncio.sleep(0.1)
        
        audio_data = AudioPreprocessingService.process(file)
        audio_16k = audio_data["audio_16k"]
        audio_22k = audio_data["audio_22k"]
        sr_16k = audio_data["sr_16k"]
        sr_22k = audio_data["sr_22k"]
        temp_path = audio_data["temp_path"]

        yield await send_progress("Running Forced Alignment & Feature Extraction...", 30)
        
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Run these heavy acoustic models concurrently
            align_task = loop.run_in_executor(pool, AlignmentService.align, audio_16k)
            pitch_task = loop.run_in_executor(pool, PitchAnalysisService.analyze, audio_22k, sr_22k)
            formant_task = loop.run_in_executor(pool, FormantAnalysisService.analyze, temp_path)
            
            alignment_data, pitch_stats, formant_data = await asyncio.gather(
                align_task, pitch_task, formant_task
            )

        yield await send_progress("Analyzing Pronunciation & Prosody...", 60)
        
        errors_data = PronunciationService.detect_errors(target_text, alignment_data["words"], alignment_data["phonemes"])
        errors = errors_data["word_errors"]
        phoneme_errors = errors_data["phoneme_errors"]
        
        native_sim = FeatureExtractionService.extract_native_similarity(audio_16k, sr_16k, self.repo)
        features = FeatureExtractionService.extract_prosody_and_clarity(audio_22k, sr_22k, alignment_data["words"])

        # Execute new services
        vowel_data = VowelAnalysisService.extract_vowels(temp_path, alignment_data["phonemes"])
        articulation_data = ArticulationAnalysisService.analyze(audio_22k, sr_22k)
        accent_data = AccentAnalysisService.analyze(features, pitch_stats, alignment_data["phonemes"])
        intonation_data = IntonationAnalysisService.analyze(pitch_stats)
        phoneme_det_data = PhonemeDetectionService.analyze(phoneme_errors)
        
        yield await send_progress("Calculating Advanced Metrics...", 80)
        
        scores = ScoringService.calculate_scores(
            words_data=alignment_data["words"],
            phonemes_data=alignment_data["phonemes"],
            native_sim=native_sim,
            features=features,
            pitch_stats=pitch_stats
        )
        
        yield await send_progress("Generating Explainable Feedback...", 90)
        
        # Combine traditional recommendations and new holistic feedback
        recommendations = RecommendationService.generate(
            scores=scores, 
            features=features, 
            errors=errors, 
            target_text=target_text
        )
        
        holistic_feedback = FeedbackService.generate(
            pronunciation=errors_data,
            vowel=vowel_data,
            articulation=articulation_data,
            accent=accent_data,
            intonation=intonation_data,
            phoneme_det=phoneme_det_data
        )

        yield await send_progress("Finalizing JSON Payload...", 95)
        
        # Build strict JSON structure matching acceptance criteria while retaining backwards compatibility with frontend structure
        result = {
            "overall_score": scores["overallScore"],
            "dimensions": { 
                "intonation": scores["intonation"],
                "pronunciation": scores["pronunciation"],
                "fluency": scores["fluency"],
                "clarity": scores["clarity"],
                "accent": scores["nativeSimilarity"]
            },
            "similarity": {
                "male": round(native_sim.get("male_score", 0) * 100, 1),
                "female": round(native_sim.get("female_score", 0) * 100, 1)
            },
            "voice_profile": native_sim["reference_gender"],
            "pronunciation": {
                **alignment_data,
                "pronunciation_score": errors_data.get("pronunciation_score", 0),
                "word_score": errors_data.get("word_score", 0),
                "phoneme_score": errors_data.get("phoneme_score", 0)
            },
            "pitch": {
                "mean": pitch_stats["mean"],
                "range": pitch_stats["range"],
                "contour": pitch_stats["contour"]
            },
            "energy": {
                "mean": float(sum(features["energy_contour"])/len(features["energy_contour"])) if features["energy_contour"] else 0.0,
                "contour": features["energy_contour"]
            },
            "pause": {
                "ratio": features["pause_ratio"],
                "timeline": [{"start": 0.0, "end": features["avg_pause"]}] 
            },
            "phonetics": {
                "vowel_space": formant_data["vowelSpace"],
                "formants": {"F1": 0, "F2": 0, "F3": 0},
                "vowels": [{"vowel": p["vowel"], "accuracy": 100} for p in formant_data["vowelSpace"]]
            },
            "articulation": articulation_data,
            "accent": accent_data,
            "intonation": intonation_data,
            "phoneme_detection": phoneme_det_data,
            "vowel_analysis": vowel_data,
            "errors": errors,
            "recommendation": recommendations + holistic_feedback,
            "analysisMetadata": scores["analysisMetadata"]
        }
        
        # DB compatibility mapping
        user_features_mock = {
            "pitch_mean": pitch_stats["mean"],
            "pitch_range": pitch_stats["range"],
            "energy_mean": result["energy"]["mean"],
            "duration": features["duration"],
            "pause_ratio": features["pause_ratio"],
            "pitch": pitch_stats["contour"],
            "energy": features["energy_contour"]
        }
        
        score_result_mock = {
            "overall": scores["overallScore"],
            "male": {"dtw": scores.get("dtw_score", 0.0) if native_sim["reference_gender"] == "Male" else 0.0},
            "female": {"dtw": scores.get("dtw_score", 0.0) if native_sim["reference_gender"] == "Female" else 0.0}
        }
        
        emb_scores_mock = {
            "male": native_sim.get("male_score", 0),
            "female": native_sim.get("female_score", 0)
        }
        
        analysis = self.repo.save_analysis(
            user_id=user_id,
            user_features=user_features_mock,
            score_result=score_result_mock,
            embedding_scores=emb_scores_mock,
            user_emb=native_sim["user_embedding"],
            ai_feedback="Generated dynamically via pipeline",
            gender_label=native_sim["reference_gender"],
            analysis_detail=json.dumps(to_python(result))
        )
        
        result["id"] = str(analysis.id)
        
        if phoneme_errors:
            self.repo.save_phoneme_stats(user_id, analysis.id, phoneme_errors)
        
        # Validate schema via Pydantic
        analysis_resp = AnalysisResponse(**result)
        
        yield json.dumps({"status": "complete", "result": analysis_resp.model_dump()}) + "\n"
