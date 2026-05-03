import librosa
import tempfile
import os

def load_audio(file):
    """
    Accept:
    - UploadFile (FastAPI)
    - file-like object
    - file path (string)
    """

    # CASE 1: UploadFile
    if hasattr(file, "file"):
        data = file.file.read()

    # CASE 2: file-like (BufferedReader)
    elif hasattr(file, "read"):
        data = file.read()

    # CASE 3: path string
    elif isinstance(file, str):
        return librosa.load(file, sr=16000)

    else:
        raise ValueError("Unsupported file type")

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    audio_22k, sr_22k = librosa.load(tmp_path, sr=22050)
    audio_16k, sr_16k = librosa.load(tmp_path, sr=16000)

    # Optional cleanup
    os.remove(tmp_path)

    return {
        "audio_22k": audio_22k,
        "sr_22k": sr_22k,
        "audio_16k": audio_16k,
        "sr_16k": sr_16k
    }