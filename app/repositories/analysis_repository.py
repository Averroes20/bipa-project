import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from app.models.audio_models import AudioAnalysis, GlobalStatistics

class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_analysis(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        score_result: Dict[str, Any],
        embedding_scores: Dict[str, Any],
        user_emb: Any,
        ai_feedback: Optional[str] = None,
        gender_label: Optional[str] = None,
        analysis_detail: Optional[str] = None
    ) -> AudioAnalysis:
        emb_json = json.dumps(user_emb.tolist() if hasattr(user_emb, "tolist") else user_emb)

        analysis = AudioAnalysis(
            user_id=str(user_id),
            gender_label=gender_label,
            pitch_mean=float(user_features.get("pitch_mean", 0.0) or 0.0),
            pitch_range=float(user_features.get("pitch_range", 0.0) or 0.0),
            energy_mean=float(user_features.get("energy_mean", 0.0) or 0.0),
            pause_ratio=float(user_features.get("pause_ratio", 0.0) or 0.0),
            duration=float(user_features.get("duration", 0.0) or 0.0),
            dtw_score_male=float(score_result.get("male", {}).get("dtw", 0.0)),
            dtw_score_female=float(score_result.get("female", {}).get("dtw", 0.0)),
            embedding_score_male=float(embedding_scores.get("male", 0.0)),
            embedding_score_female=float(embedding_scores.get("female", 0.0)),
            final_score=float(score_result.get("overall", 0.0)),
            embedding=emb_json,
            ai_feedback=ai_feedback if isinstance(ai_feedback, str) else json.dumps(ai_feedback or {}),
            analysis_detail=analysis_detail
        )

        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis

    def save_phoneme_stats(self, user_id: str, analysis_id: str, phoneme_errors: List[Dict[str, Any]]) -> None:
        from app.models.audio_models import PhonemeStatistic, UserPhonemeSummary
        
        for p in phoneme_errors:
            expected = p.get("expected", "")
            detected = p.get("detected", "")
            is_correct = p.get("is_correct", False)
            conf = float(p.get("confidence", 0.0))
            
            phoneme = expected if expected else detected
            if not phoneme:
                continue
                
            stat = PhonemeStatistic(
                user_id=str(user_id),
                analysis_id=str(analysis_id),
                phoneme=phoneme,
                expected_phoneme=expected,
                detected_phoneme=detected,
                is_correct=is_correct,
                confidence=conf
            )
            self.db.add(stat)
            
            summary = self.db.query(UserPhonemeSummary).filter_by(user_id=str(user_id), phoneme=phoneme).first()
            if not summary:
                summary = UserPhonemeSummary(
                    user_id=str(user_id),
                    phoneme=phoneme,
                    occurrences=0,
                    correct_count=0,
                    mistake_count=0
                )
                self.db.add(summary)
            
            summary.occurrences += 1
            if is_correct:
                summary.correct_count += 1
            else:
                summary.mistake_count += 1
                
            summary.accuracy = round((summary.correct_count / summary.occurrences) * 100, 1)
            
        self.db.commit()

    def get_top_k_candidates(self, user_emb: Any, k: int = 10, gender: Optional[str] = None) -> List[Any]:
        """Fetch top K candidates from dataset_feature vector DB."""
        try:
            emb_arr = user_emb.tolist() if hasattr(user_emb, "tolist") else user_emb
            pg_vector_str = "[" + ",".join(map(str, emb_arr)) + "]"

            base_query = """
                SELECT c.pitch_contour, c.energy_contour, f.embedding_vector, form.vowel_profile
                FROM dataset_feature f
                JOIN dataset_audio a ON a.id = f.audio_id
                LEFT JOIN dataset_contour c ON c.audio_id = a.id
                LEFT JOIN dataset_formant form ON form.audio_id = a.id
            """
            params: Dict[str, Any] = {"user_emb": pg_vector_str, "k": k}

            if gender:
                base_query += " WHERE a.gender = :gender"
                params["gender"] = gender

            base_query += " ORDER BY f.embedding_vector::vector <-> :user_emb::vector LIMIT :k"

            result = self.db.execute(text(base_query), params)
            rows = result.fetchall()
            
            candidates = []
            for row in rows:
                try:
                    pitch_contour = row[0] if row[0] else []
                    if isinstance(pitch_contour, str):
                        pitch_contour = json.loads(pitch_contour)
                    energy_contour = row[1] if row[1] else []
                    if isinstance(energy_contour, str):
                        energy_contour = json.loads(energy_contour)
                    # row[2] is the embedding_vector
                    vowel_profile = row[3] if len(row) > 3 and row[3] else {}
                    if isinstance(vowel_profile, str):
                        vowel_profile = json.loads(vowel_profile)
                    candidates.append({
                        "pitch_contour": pitch_contour,
                        "energy_contour": energy_contour,
                        "embedding_vector": row[2],
                        "vowel_profile": vowel_profile
                    })
                except Exception:
                    pass
            return candidates
        except Exception as e:
            self.db.rollback()
            # Fallback
            query = text("""
                SELECT c.pitch_contour, c.energy_contour, f.embedding_vector, form.vowel_profile 
                FROM dataset_feature f
                JOIN dataset_audio a ON a.id = f.audio_id
                LEFT JOIN dataset_contour c ON c.audio_id = a.id
                LEFT JOIN dataset_formant form ON form.audio_id = a.id
            """ + (" WHERE a.gender = :gender" if gender else "") + " LIMIT :k")
            params = {"k": k}
            if gender:
                params["gender"] = gender
            res = self.db.execute(query, params)
            rows = res.fetchall()
            candidates = []
            for row in rows:
                try:
                    pitch_contour = row[0] if row[0] else []
                    if isinstance(pitch_contour, str):
                        pitch_contour = json.loads(pitch_contour)
                    energy_contour = row[1] if row[1] else []
                    if isinstance(energy_contour, str):
                        energy_contour = json.loads(energy_contour)
                    vowel_profile = row[3] if len(row) > 3 and row[3] else {}
                    if isinstance(vowel_profile, str):
                        vowel_profile = json.loads(vowel_profile)
                    candidates.append({
                        "pitch_contour": pitch_contour,
                        "energy_contour": energy_contour,
                        "embedding_vector": row[2],
                        "vowel_profile": vowel_profile
                    })
                except Exception:
                    pass
            return candidates

    def get_dataset_reference(self) -> Dict[str, Any]:
        """Fetches male/female reference baseline statistics."""
        try:
            result = self.db.execute(text("""
                SELECT a.gender, AVG(f.pitch_mean), AVG(f.energy_mean), AVG(f.pause_ratio), AVG(a.duration) 
                FROM dataset_audio a
                LEFT JOIN dataset_feature f ON f.audio_id = a.id
                GROUP BY a.gender
            """)).fetchall()
            dataset_ref = {}
            for r in result:
                gender = r[0] if r[0] else "unknown"
                dataset_ref[gender] = {
                    "pitch_mean": float(r[1] or 0),
                    "energy_mean": float(r[2] or 0),
                    "pause_ratio": float(r[3] or 0),
                    "duration": float(r[4] or 0),
                }
            
            if "male" not in dataset_ref:
                dataset_ref["male"] = {"pitch_mean": 130.0, "energy_mean": 0.05, "pause_ratio": 0.2, "duration": 4.0}
            if "female" not in dataset_ref:
                dataset_ref["female"] = {"pitch_mean": 210.0, "energy_mean": 0.05, "pause_ratio": 0.2, "duration": 4.0}
                
            return dataset_ref
        except Exception:
            return {
                "male": {"pitch_mean": 130.0, "energy_mean": 0.05, "pause_ratio": 0.2, "duration": 4.0},
                "female": {"pitch_mean": 210.0, "energy_mean": 0.05, "pause_ratio": 0.2, "duration": 4.0}
            }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        result = self.db.execute(text("""
            SELECT
                COUNT(*) as total_analysis,
                COALESCE(AVG(final_score), 0) as avg_score,
                COALESCE(AVG(pitch_mean), 0) as avg_pitch,
                COALESCE(AVG(energy_mean), 0) as avg_energy,
                COALESCE(AVG(pause_ratio), 0) as avg_pause
            FROM audio_analysis
        """)).mappings().first()

        if not result:
            return {"total_analysis": 0, "avg_score": 0.0, "avg_pitch": 0.0, "avg_energy": 0.0, "avg_pause": 0.0}

        return {
            "total_analysis": int(result["total_analysis"]),
            "avg_score": float(result["avg_score"]),
            "avg_pitch": float(result["avg_pitch"]),
            "avg_energy": float(result["avg_energy"]),
            "avg_pause": float(result["avg_pause"])
        }

    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        result = self.db.execute(
            text("""
                SELECT
                    COALESCE(AVG(final_score), 0) as avg_score,
                    COALESCE(MAX(final_score), 0) as best_score,
                    COUNT(*) as sessions
                FROM audio_analysis
                WHERE user_id = :user_id
            """),
            {"user_id": str(user_id)}
        ).mappings().fetchone()

        if not result:
            return {"avg_score": 0.0, "best_score": 0.0, "sessions": 0}

        return {
            "avg_score": float(result["avg_score"]),
            "best_score": float(result["best_score"]),
            "sessions": int(result["sessions"])
        }

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        result = self.db.execute(
            text("""
                SELECT
                    created_at,
                    final_score,
                    gender_label,
                    duration
                FROM audio_analysis
                WHERE user_id = :user_id
                ORDER BY created_at ASC
                LIMIT :limit
            """),
            {"user_id": str(user_id), "limit": limit}
        ).mappings().fetchall()

        history = []
        for row in result:
            history.append({
                "date": row["created_at"].strftime("%Y-%m-%d %H:%M") if row["created_at"] else "",
                "score": float(row["final_score"]),
                "gender": row["gender_label"] or "unknown",
                "duration": float(row["duration"])
            })
        return history

    def rebuild_global_statistics(self) -> None:
        rows = self.db.execute(text("""
            SELECT
                gender_label,
                AVG(pitch_mean),
                AVG(energy_mean),
                AVG(pause_ratio),
                AVG(duration)
            FROM audio_analysis
            WHERE gender_label IS NOT NULL
            GROUP BY gender_label
        """)).fetchall()

        self.db.execute(text("DELETE FROM global_statistics"))

        for row in rows:
            self.db.execute(text("""
                INSERT INTO global_statistics (
                    gender_label, pitch_mean, energy_mean, pause_ratio, duration
                ) VALUES (
                    :gender_label, :pitch_mean, :energy_mean, :pause_ratio, :duration
                )
            """), {
                "gender_label": row[0],
                "pitch_mean": float(row[1] or 0),
                "energy_mean": float(row[2] or 0),
                "pause_ratio": float(row[3] or 0),
                "duration": float(row[4] or 0)
            })

        self.db.commit()

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        result = self.db.execute(
            text("SELECT analysis_detail, id FROM audio_analysis WHERE id = :id"),
            {"id": str(analysis_id)}
        ).fetchone()
        
        if result and result[0]:
            try:
                data = json.loads(result[0])
                data["id"] = result[1]
                return data
            except Exception:
                pass
        return None

    def get_latest_analysis(self, user_id: str) -> Optional[Dict[str, Any]]:
        result = self.db.execute(
            text("SELECT analysis_detail, id FROM audio_analysis WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 1"),
            {"user_id": str(user_id)}
        ).fetchone()
        
        if result and result[0]:
            try:
                data = json.loads(result[0])
                data["id"] = result[1]
                return data
            except Exception:
                pass
        return None

    def clear_speech_corpus(self) -> None:
        from app.models.dataset_models import DatasetAudio
        self.db.query(DatasetAudio).delete()
        self.db.commit()
