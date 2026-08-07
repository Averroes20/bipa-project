import os
import glob
from sqlalchemy.orm import Session
from app.core.logger import logger
from app.services.corpus.audio_loader import CorpusAudioLoader
from app.services.corpus.feature_pipeline import CorpusFeaturePipeline
from app.services.corpus.embedding_pipeline import CorpusEmbeddingPipeline
from app.services.corpus.formant_pipeline import CorpusFormantPipeline
from app.services.corpus.contour_pipeline import CorpusContourPipeline
from app.services.pipeline.alignment_service import AlignmentService
from app.models.dataset_models import DatasetAudio, DatasetFeature, DatasetContour, DatasetFormant

class CorpusBuilder:
    def __init__(self, db: Session, progress_tracker: dict = None):
        self.db = db
        self.progress = progress_tracker if progress_tracker is not None else {}

    def build(self) -> None:
        """
        Scans dataset/native, extracts features, and bulk inserts into Speech Corpus PostgreSQL.
        """
        try:
            self._update_progress("status", "running")
            self._update_progress("progress", 0)
            
            # 1. Scan dataset/native
            dataset_path = os.path.join(os.getcwd(), "dataset", "native")
            if not os.path.exists(dataset_path):
                self._update_progress("status", "error")
                logger.error(f"Dataset path not found: {dataset_path}")
                return
                
            all_wav_files = []
            for gender in ["male", "female"]:
                gender_path = os.path.join(dataset_path, gender)
                if os.path.exists(gender_path):
                    all_wav_files.extend([(gender, f) for f in glob.glob(os.path.join(gender_path, "*.wav"))])
            
            total_files = len(all_wav_files)
            self._update_progress("total", total_files)
            self._update_progress("processed", 0)
            
            if total_files == 0:
                self._update_progress("status", "completed")
                return

            # Clear old dataset 
            self.db.query(DatasetAudio).delete()
            self.db.commit()

            batch_size = 50
            batch_objects = []

            for i, (gender, wav_file) in enumerate(all_wav_files):
                self._update_progress("current_file", os.path.basename(wav_file))
                
                try:
                    # Pipeline
                    audio_data = CorpusAudioLoader.load(wav_file)
                    alignment_data = AlignmentService.align(audio_data["audio_16k"])
                    words_data = alignment_data.get("words", [])
                    phonemes_data = alignment_data.get("phonemes", [])
                    
                    features = CorpusFeaturePipeline.extract_features(audio_data["audio_22k"], audio_data["sr_22k"], words_data)
                    embedding_vector = CorpusEmbeddingPipeline.extract_embedding(audio_data["audio_16k"], audio_data["sr_16k"])
                    formants = CorpusFormantPipeline.extract_formants(audio_data["temp_path"], phonemes_data)
                    contours = CorpusContourPipeline.extract_contours(audio_data["audio_22k"], audio_data["sr_22k"], words_data)
                    
                    # Create Objects
                    audio_obj = DatasetAudio(
                        filename=os.path.basename(wav_file),
                        gender=gender,
                        duration=audio_data["duration"],
                        sample_rate=22050,
                        language="id-ID"
                    )
                    
                    feature_obj = DatasetFeature(
                        **features,
                        embedding_vector=embedding_vector
                    )
                    
                    contour_obj = DatasetContour(**contours)
                    formant_obj = DatasetFormant(**formants)
                    
                    audio_obj.feature = feature_obj
                    audio_obj.contour = contour_obj
                    audio_obj.formant = formant_obj
                    
                    batch_objects.append(audio_obj)
                    
                except Exception as e:
                    logger.error(f"Failed to process {wav_file} in pipeline: {e}")
                
                # Batch insert
                if len(batch_objects) >= batch_size or (i + 1) == total_files:
                    try:
                        self.db.add_all(batch_objects)
                        self.db.commit()
                        batch_objects.clear()
                    except Exception as e:
                        self.db.rollback()
                        logger.error(f"Batch insert failed: {e}")
                        batch_objects.clear()

                self._update_progress("processed", i + 1)
                self._update_progress("progress", int(((i + 1) / total_files) * 100))

            self._update_progress("status", "completed")
            
        except Exception as e:
            logger.error(f"Corpus Builder Error: {e}")
            self._update_progress("status", "error")

    def _update_progress(self, key: str, value: Any):
        if self.progress is not None:
            self.progress[key] = value
