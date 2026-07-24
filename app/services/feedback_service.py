import math
import os
import google.generativeai as genai
from typing import Dict, Any, Optional

# Configure Google Generative AI
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

def safe_val(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or math.isnan(float(v)):
            return default
        return float(v)
    except Exception:
        return default

def relative_diff(user: float, ref: float) -> float:
    user = safe_val(user)
    ref = safe_val(ref)
    if ref == 0:
        return 0.0
    return (user - ref) / abs(ref)

def compute_diff(user: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, float]:
    return {
        "pitch": relative_diff(user.get("pitch_mean", 0), ref.get("pitch_mean", 0)),
        "energy": relative_diff(user.get("energy_mean", 0), ref.get("energy_mean", 0)),
        "pause": relative_diff(user.get("pause_ratio", 0), ref.get("pause_ratio", 0)),
        "duration": relative_diff(user.get("duration", 0), ref.get("duration", 0)),
    }

def magnitude(diff: Dict[str, float]) -> float:
    return sum(abs(v) for v in diff.values())

def generate_rule_based_feedback(user_features: Dict[str, Any], dataset_reference: Dict[str, Any]) -> Dict[str, Any]:
    male_ref = dataset_reference.get("male", {})
    female_ref = dataset_reference.get("female", {})

    diff_male = compute_diff(user_features, male_ref)
    diff_female = compute_diff(user_features, female_ref)

    profile = "female" if magnitude(diff_female) < magnitude(diff_male) else "male"
    diff = diff_female if profile == "female" else diff_male
    label = "perempuan" if profile == "female" else "laki-laki"

    feature_feedback: Dict[str, str] = {}

    if diff["pitch"] > 0.15:
        feature_feedback["pitch"] = f"intonasi kamu terdengar lebih tinggi dan ringan dibanding {label}"
    elif diff["pitch"] < -0.15:
        feature_feedback["pitch"] = f"intonasi kamu lebih dalam dan berat dibanding {label}"

    if diff["energy"] > 0.15:
        feature_feedback["energy"] = f"cara bicara kamu terdengar lebih tegas dan berenergi dibanding {label}"
    elif diff["energy"] < -0.15:
        feature_feedback["energy"] = f"cara bicara kamu lebih santai dan tidak terlalu menekan dibanding {label}"

    if diff["pause"] > 0.15:
        feature_feedback["pause"] = f"kamu cenderung lebih sering memberi jeda dibanding {label}"
    elif diff["pause"] < -0.15:
        feature_feedback["pause"] = f"bicara kamu lebih mengalir dan cepat tanpa banyak jeda dibanding {label}"

    if diff["duration"] > 0.2:
        feature_feedback["duration"] = f"durasi bicara kamu cenderung lebih panjang dibanding {label}"
    elif diff["duration"] < -0.2:
        feature_feedback["duration"] = f"durasi bicara kamu lebih singkat dibanding {label}"

    if not feature_feedback:
        feature_feedback["general"] = "karakter suara kamu cukup seimbang"

    tone = "lembut" if profile == "female" else "tegas"
    summary = f"secara umum, cara bicara kamu terdengar lebih {tone} dengan beberapa karakteristik yang menonjol pada pola bicara kamu"

    return {
        "voice_profile": f"cenderung {profile}",
        "features": feature_feedback,
        "summary": summary
    }

def generate_natural_llm_feedback(data: Dict[str, Any]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "Pelafalan dan intonasi Anda sudah cukup baik. Teruskan berlatih membaca kalimat Bahasa Indonesia secara konsisten!"

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = f"""
        Anda adalah guru Bahasa Indonesia untuk penutur asing (BIPA).
        Tugas Anda adalah menganalisis gaya bicara berdasarkan perbandingan dengan pola penutur laki-laki dan perempuan.

        Data:
        Scores: {data.get("scores")}
        Features: {data.get("features")}
        Rule-based feedback: {data.get("rule_based")}

        Fokus analisis:
        1. Apakah gaya bicara lebih mendekati laki-laki atau perempuan?
        2. Jelaskan karakteristiknya (apakah lebih tegas/kuat atau lebih halus/mengalir).
        3. Bandingkan kedua pola tersebut secara natural.
        4. Jangan menyebut angka, gunakan bahasa deskriptif.

        Jawaban harus:
        - natural
        - singkat (2–3 kalimat)
        - seperti guru manusia
        """

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("LLM feedback generation error:", e)
        return "Pelafalan Anda sudah cukup jelas. Perhatikan penekanan intonasi pada akhir kalimat."
