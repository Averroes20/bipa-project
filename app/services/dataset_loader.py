from app.services import embedding
from app.services import audio_loader
from app.services import feature_extraction
import pickle
import os

DATASET_PATH = "dataset/native"
CACHE_PATH = "dataset/cache.pkl"

def preprocess_dataset():
    dataset = {
        "male": [],
        "female": [],
        "male_embeddings": [],
        "female_embeddings": []
    }

    for gender in ["male", "female"]:
        folder = os.path.join(DATASET_PATH, gender)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            audio, sr = audio_loader.load_audio(path)

            # 🎧 feature
            features = feature_extraction.extract_features(audio, sr)
            dataset[gender].append(features)

            # 🤖 embedding
            emb = embedding.extract_embedding(audio, sr)
            dataset[f"{gender}_embeddings"].append(emb)

    # cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dataset, f)

    return dataset