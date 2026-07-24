import tempfile
import librosa
from typing import Dict, Any, Union

def load_audio(file: Union[Any, str]) -> Dict[str, Any]:
    """
    Accepts UploadFile, file-like object, or file path string.
    Returns resampled audio arrays for both 22050Hz and 16000Hz along with temporary path.
    """
    if hasattr(file, "file"):
        data = file.file.read()
    elif hasattr(file, "read"):
        data = file.read()
    elif isinstance(file, str):
        audio_22k, sr_22k = librosa.load(file, sr=22050)
        audio_16k, sr_16k = librosa.load(file, sr=16000)
        return {
            "audio_22k": audio_22k,
            "sr_22k": sr_22k,
            "audio_16k": audio_16k,
            "sr_16k": sr_16k,
            "temp_path": file
        }
    else:
        raise ValueError("Unsupported file format for audio loading")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    audio_22k, sr_22k = librosa.load(tmp_path, sr=22050)
    audio_16k, sr_16k = librosa.load(tmp_path, sr=16000)

    return {
        "audio_22k": audio_22k,
        "sr_22k": sr_22k,
        "audio_16k": audio_16k,
        "sr_16k": sr_16k,
        "temp_path": tmp_path
    }
