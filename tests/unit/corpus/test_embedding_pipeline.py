import pytest
import numpy as np
from unittest.mock import patch
from app.services.corpus.embedding_pipeline import CorpusEmbeddingPipeline

def test_extract_embedding_numpy_array():
    mock_audio = np.zeros(16000)
    mock_embedding = np.array([0.1, 0.2, 0.3])
    with patch("app.services.corpus.embedding_pipeline.extract_embedding", return_value=mock_embedding) as mock_extract:
        result = CorpusEmbeddingPipeline.extract_embedding(mock_audio, 16000)
        assert result == str(mock_embedding.tolist())
        mock_extract.assert_called_once_with(mock_audio, 16000)

def test_extract_embedding_list():
    mock_audio = np.zeros(16000)
    mock_embedding = [0.1, 0.2, 0.3]
    with patch("app.services.corpus.embedding_pipeline.extract_embedding", return_value=mock_embedding) as mock_extract:
        result = CorpusEmbeddingPipeline.extract_embedding(mock_audio, 16000)
        assert result == str(mock_embedding)
        mock_extract.assert_called_once_with(mock_audio, 16000)
