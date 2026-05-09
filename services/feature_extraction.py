import librosa
import numpy as np
from app.utils.normalization import smooth_pitch
from app.utils.normalization import z_score_normalize

def extract_features(audio, sr):
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]

    pitch = smooth_pitch(pitch)
    pitch = z_score_normalize(pitch)

    # Energy (RAW dulu)
    energy_raw = librosa.feature.rms(y=audio)[0]

    # Pause detection pakai RAW
    silence = energy_raw < np.mean(energy_raw) * 0.5
    pause_ratio = np.sum(silence) / len(energy_raw)

    # Baru normalize untuk comparison
    energy = z_score_normalize(energy_raw)
    
    pitch_mean = float(np.mean(pitch)) if len(pitch) else 0.0
    pitch_range = float(np.max(pitch) - np.min(pitch)) if len(pitch) else 0.0
    energy_mean = float(np.mean(energy_raw)) if len(energy_raw) else 0.0

    duration = librosa.get_duration(y=audio, sr=sr)

    return {
        "pitch": pitch.tolist(),
        "energy": energy.tolist(),

        "pitch_mean": pitch_mean,
        "pitch_range": pitch_range,
        "energy_mean": energy_mean,
        "duration": float(duration),
        "pause_ratio": float(pause_ratio),
    }