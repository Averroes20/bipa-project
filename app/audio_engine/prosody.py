import numpy as np
import librosa
from scipy.signal import medfilt
from typing import Dict, Any

def smooth_pitch(pitch: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if len(pitch) == 0:
        return pitch
    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]
    if len(pitch) < kernel_size:
        return pitch
    return medfilt(pitch, kernel_size=kernel_size)

def z_score_normalize(x: np.ndarray) -> np.ndarray:
    x = np.array(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std

def extract_features(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    # Pitch extraction via piptrack
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]

    pitch = smooth_pitch(pitch)
    pitch_normalized = z_score_normalize(pitch)

    # Energy (RMS)
    energy_raw = librosa.feature.rms(y=audio)[0]

    # Pause ratio detection from raw energy
    silence = energy_raw < (np.mean(energy_raw) * 0.5)
    pause_ratio = float(np.sum(silence) / len(energy_raw)) if len(energy_raw) > 0 else 0.0

    # Normalized energy
    energy_normalized = z_score_normalize(energy_raw)

    pitch_mean = float(np.mean(pitch)) if len(pitch) > 0 else 0.0
    pitch_range = float(np.max(pitch) - np.min(pitch)) if len(pitch) > 0 else 0.0
    energy_mean = float(np.mean(energy_raw)) if len(energy_raw) > 0 else 0.0
    duration = float(librosa.get_duration(y=audio, sr=sr))

    return {
        "pitch": pitch.tolist() if isinstance(pitch, np.ndarray) else list(pitch),
        "energy": energy_raw.tolist() if isinstance(energy_raw, np.ndarray) else list(energy_raw),
        "pitch_mean": pitch_mean,
        "pitch_range": pitch_range,
        "energy_mean": energy_mean,
        "duration": duration,
        "pause_ratio": pause_ratio,
    }

def extract_formants(audio_path: str) -> Dict[str, float]:
    """Extract F1, F2, F3 formant frequencies using Parselmouth if available."""
    try:
        import parselmouth
        sound = parselmouth.Sound(audio_path)
        formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
        
        duration = sound.get_total_duration()
        times = np.arange(0, duration, 0.01)

        f1_vals, f2_vals, f3_vals = [], [], []
        for t in times:
            try:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                if f1: f1_vals.append(f1)
                if f2: f2_vals.append(f2)
                if f3: f3_vals.append(f3)
            except Exception:
                continue

        def safe_float(v, default=0.0):
            return float(v) if v is not None and not np.isnan(v) else default

        samples = []
        for f1, f2, f3 in zip(f1_vals, f2_vals, f3_vals):
            if not np.isnan(f1) and not np.isnan(f2) and f1 > 0 and f2 > 0:
                samples.append({"f1": safe_float(f1), "f2": safe_float(f2), "f3": safe_float(f3)})
        
        # Limit to ~30 points to avoid huge JSON payloads but still provide good scatter
        if len(samples) > 30:
            indices = np.linspace(0, len(samples)-1, 30, dtype=int)
            samples = [samples[i] for i in indices]

        return {
            "f1": safe_float(np.mean(f1_vals) if f1_vals else 0.0),
            "f2": safe_float(np.mean(f2_vals) if f2_vals else 0.0),
            "f3": safe_float(np.mean(f3_vals) if f3_vals else 0.0),
            "samples": samples
        }
    except Exception as e:
        print("Formant extraction skipped or parselmouth unavailable:", e)
        return {"f1": 0.0, "f2": 0.0, "f3": 0.0, "samples": []}
