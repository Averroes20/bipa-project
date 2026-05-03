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

    duration = librosa.get_duration(y=audio, sr=sr)

    return {
        "pitch": pitch.tolist(),
        "energy": energy.tolist(),
        "duration": float(duration),
        "pause_ratio": float(pause_ratio)
    }