import numpy as np
from scipy.signal import medfilt

def smooth_pitch(pitch, kernel_size=5):
    if len(pitch) == 0:
        return pitch

    pitch = np.array(pitch)

    # Remove zero / invalid values
    pitch = pitch[pitch > 0]

    if len(pitch) < kernel_size:
        return pitch

    smoothed = medfilt(pitch, kernel_size=kernel_size)
    return smoothed

def z_score_normalize(x):
    x = np.array(x)

    std = np.std(x)

    if std == 0:
        return np.zeros_like(x)  # 🔥 FIX

    return (x - np.mean(x)) / std