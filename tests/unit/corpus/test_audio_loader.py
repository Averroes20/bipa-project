import pytest
from unittest.mock import patch
from app.services.corpus.audio_loader import CorpusAudioLoader

def test_load_file_not_found():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Audio file not found: missing.wav"):
            CorpusAudioLoader.load("missing.wav")

def test_load_success():
    mock_result = {"audio_16k": [0.0], "audio_22k": [0.0], "temp_path": "/tmp/test.wav"}
    with patch("os.path.exists", return_value=True), \
         patch("app.services.corpus.audio_loader.AudioPreprocessingService.process", return_value=mock_result) as mock_process:
        result = CorpusAudioLoader.load("test.wav")
        assert result == mock_result
        mock_process.assert_called_once_with("test.wav")
