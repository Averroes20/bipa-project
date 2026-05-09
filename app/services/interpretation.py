import math

# =========================
# 🧠 HELPER
# =========================
def safe(v, default=0.0):
    try:
        if v is None or math.isnan(v):
            return default
        return float(v)
    except:
        return default


def relative_diff(user, ref):
    """
    Biar semua feature comparable (scale aware)
    hasil:
    +0.2 = 20% lebih tinggi
    -0.3 = 30% lebih rendah
    """
    user = safe(user)
    ref = safe(ref)

    if ref == 0:
        return 0.0

    return (user - ref) / abs(ref)


def compute_diff(user, ref):
    return {
        "pitch": relative_diff(user.get("pitch_mean"), ref.get("pitch_mean")),
        "energy": relative_diff(user.get("energy_mean"), ref.get("energy_mean")),
        "pause": relative_diff(user.get("pause_ratio"), ref.get("pause_ratio")),
        "duration": relative_diff(user.get("duration"), ref.get("duration")),
    }


def magnitude(diff):
    return sum(abs(v) for v in diff.values())


# =========================
# 🎤 INTERPRETASI PER FEATURE
# =========================
def interpret_pitch(diff, label):
    if diff > 0.15:
        return f"intonasi kamu terdengar lebih tinggi dan ringan dibanding {label}"
    elif diff < -0.15:
        return f"intonasi kamu lebih dalam dan berat dibanding {label}"
    return None


def interpret_energy(diff, label):
    if diff > 0.15:
        return f"cara bicara kamu terdengar lebih tegas dan berenergi dibanding {label}"
    elif diff < -0.15:
        return f"cara bicara kamu lebih santai dan tidak terlalu menekan dibanding {label}"
    return None


def interpret_pause(diff, label):
    if diff > 0.15:
        return f"kamu cenderung lebih sering memberi jeda dibanding {label}"
    elif diff < -0.15:
        return f"bicara kamu lebih mengalir dan cepat tanpa banyak jeda dibanding {label}"
    return None


def interpret_duration(diff, label):
    if diff > 0.2:
        return f"durasi bicara kamu cenderung lebih panjang dibanding {label}"
    elif diff < -0.2:
        return f"durasi bicara kamu lebih singkat dibanding {label}"
    return None


# =========================
# 🧠 PROFILE DETECTION
# =========================
def detect_voice_profile(diff_male, diff_female):
    male_score = magnitude(diff_male)
    female_score = magnitude(diff_female)

    return "female" if female_score < male_score else "male"


# =========================
# 🧠 SUMMARY GENERATOR
# =========================
def generate_summary(profile, features):
    if not features:
        return "karakter suara kamu cukup seimbang dan tidak terlalu condong ke pola tertentu"

    tone = "lembut" if profile == "female" else "tegas"

    return f"secara umum, cara bicara kamu terdengar lebih {tone} dengan beberapa karakteristik yang menonjol pada pola bicara kamu"


# =========================
# 🧠 MAIN ENGINE
# =========================
def generate_feedback_v2(user_features, dataset_reference):
    """
    dataset_reference format:
    {
        "male": {...},
        "female": {...}
    }
    """

    male_ref = dataset_reference.get("male", {})
    female_ref = dataset_reference.get("female", {})

    # =========================
    # 🔍 HITUNG GAP (RELATIVE)
    # =========================
    diff_male = compute_diff(user_features, male_ref)
    diff_female = compute_diff(user_features, female_ref)

    # =========================
    # 🧠 DETECT PROFILE
    # =========================
    profile = detect_voice_profile(diff_male, diff_female)

    ref = female_ref if profile == "female" else male_ref
    diff = diff_female if profile == "female" else diff_male
    label = "perempuan" if profile == "female" else "laki-laki"

    # =========================
    # 🎯 FEATURE-BASED OUTPUT
    # =========================
    feature_feedback = {}

    pitch_fb = interpret_pitch(diff["pitch"], label)
    if pitch_fb:
        feature_feedback["pitch"] = pitch_fb

    energy_fb = interpret_energy(diff["energy"], label)
    if energy_fb:
        feature_feedback["energy"] = energy_fb

    pause_fb = interpret_pause(diff["pause"], label)
    if pause_fb:
        feature_feedback["pause"] = pause_fb

    duration_fb = interpret_duration(diff["duration"], label)
    if duration_fb:
        feature_feedback["duration"] = duration_fb

    # fallback biar ga kosong
    if not feature_feedback:
        feature_feedback["general"] = "karakter suara kamu cukup seimbang"

    # =========================
    # 🧾 SUMMARY
    # =========================
    summary = generate_summary(profile, feature_feedback)

    # =========================
    # 🧾 FINAL OUTPUT
    # =========================
    return {
        "voice_profile": f"cenderung {profile}",
        "features": feature_feedback,
        "summary": summary
    }