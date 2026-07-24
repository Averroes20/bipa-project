from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_similar(user_emb, dataset):
    scores = []

    for row in dataset:
        sim = cosine_similarity(
            np.array(user_emb).reshape(1, -1),
            np.array(row["embedding"]).reshape(1, -1)
        )[0][0]

        scores.append(sim)

    return scores