import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h",
    force_download=True
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h",
    force_download=True
)

def extract_embedding(audio, sr):
    audio_tensor = torch.tensor(audio)

    inputs = processor(audio_tensor, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # ambil mean embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding