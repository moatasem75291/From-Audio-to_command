"""Transcription failures should be actionable and must not leak provider details."""

import io
import sys
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.audio import transcribe


class DownloadError(Exception):
    pass


class GatedError(DownloadError):
    pass


class MemoryErrorForTest(Exception):
    pass


def speech_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(np.full(16000, 4096, dtype="<i2").tobytes())
    return buffer.getvalue()


class AudioErrorTests(unittest.TestCase):
    def test_missing_dependencies_explains_install_and_restart(self):
        with patch.dict(sys.modules, {"librosa": None}):
            with self.assertRaises(RuntimeError) as result:
                transcribe(speech_wav())
        self.assertIn("requirements-audio.txt", str(result.exception))
        self.assertIn("restart", str(result.exception))

    def test_access_download_and_memory_failures_are_distinct(self):
        modules = {
            "librosa": SimpleNamespace(resample=lambda waveform, **kwargs: waveform),
            "torch": SimpleNamespace(OutOfMemoryError=MemoryErrorForTest),
            "huggingface_hub.errors": SimpleNamespace(
                GatedRepoError=GatedError, HfHubHTTPError=DownloadError,
            ),
        }
        for error, expected in [
            (GatedError("SECRET"), "same approved account"),
            (DownloadError("SECRET"), "internet connection"),
            (MemoryErrorForTest("SECRET"), "ASR_DEVICE=cpu"),
            (ImportError("SECRET"), "audio libraries"),
        ]:
            with self.subTest(error=type(error).__name__), patch.dict(sys.modules, modules), \
                    patch("backend.audio._load_model", side_effect=error):
                with self.assertRaises(RuntimeError) as result:
                    transcribe(speech_wav())
                self.assertIn(expected, str(result.exception))
                self.assertNotIn("SECRET", str(result.exception))


if __name__ == "__main__":
    unittest.main()
