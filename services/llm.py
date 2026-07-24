import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDFrCEHVuinGtgcVoaINHQX7fWwpbIYoKA")

model = genai.GenerativeModel("gemini-flash-latest")

def generate_natural_feedback(data):
    prompt = f"""
    Anda adalah guru Bahasa Indonesia untuk penutur asing (BIPA).

    Tugas Anda adalah menganalisis gaya bicara berdasarkan perbandingan dengan pola penutur laki-laki dan perempuan.

    Data:
    Scores: {data["scores"]}
    Features: {data["features"]}
    Rule-based feedback: {data["rule_based"]}

    Fokus analisis:
    1. Apakah gaya bicara lebih mendekati laki-laki atau perempuan?
    2. Jelaskan karakteristiknya:
       - apakah lebih tegas / kuat
       - atau lebih halus / mengalir
    3. Bandingkan kedua pola tersebut secara natural
    4. Jangan menyebut angka, gunakan bahasa deskriptif

    Contoh gaya output:
    "Gaya bicara Anda cenderung lebih mendekati pola perempuan, dengan intonasi yang lebih halus dan mengalir. Namun, dibandingkan pola laki-laki yang biasanya lebih tegas, pengucapan Anda masih terasa kurang kuat di beberapa bagian."

    Jawaban harus:
    - natural
    - singkat (2–3 kalimat)
    - seperti guru manusia
    """

    response = model.generate_content(prompt)
    return response.text