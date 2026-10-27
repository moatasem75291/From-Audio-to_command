"""Validate short WAV recordings before lazily loading Cohere's Arabic ASR."""

import io
import os
import wave
from functools import lru_cache
from threading import Lock

from backend.config import ASR_MODEL, MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS
from backend.security import SafetyError

_INFERENCE_LOCK = Lock()


def validate_wav(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_AUDIO_BYTES:
        raise SafetyError("Upload a WAV recording of at most 10 MB.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            rate = audio.getframerate()
            frames = audio.getnframes()
            if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
                raise SafetyError("Use an uncompressed 16-bit PCM WAV recording.")
            if audio.getnchannels() not in {1, 2} or not 8000 <= rate <= 48000:
                raise SafetyError("Use mono/stereo WAV audio at 8–48 kHz.")
            if not 0 < frames / rate <= MAX_AUDIO_SECONDS:
                raise SafetyError("Record between 0 and 30 seconds of audio.")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * audio.getnchannels() * 2:
                raise SafetyError("The WAV recording is incomplete.")
            return pcm, rate, audio.getnchannels()
    except (wave.Error, EOFError) as error:
        raise SafetyError("This is not a valid PCM WAV recording.") from error


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(ASR_MODEL, trust_remote_code=False)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        ASR_MODEL, device_map=os.getenv("ASR_DEVICE", "auto"),
        trust_remote_code=False, use_safetensors=True,
    )
    model.eval()
    return processor, model


def transcribe(data):
    pcm, rate, channels = validate_wav(data)
    try:
        import librosa
        import numpy as np
        import torch
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise RuntimeError(
            "Audio dependencies are missing or incompatible. Run: "
            "python -m pip install -r requirements-audio.txt, then restart Streamlit."
        ) from error

    try:
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        waveform = waveform.reshape(-1, channels).mean(axis=1)
        if float(np.sqrt(np.mean(waveform ** 2))) < 0.003:
            raise SafetyError("The recording is too quiet. Please record speech again.")
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)
        # Shared cached weights are used serially, never conversation state.
        with _INFERENCE_LOCK:
            processor, model = _load_model()
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", language="ar")
            chunk_index = inputs.get("audio_chunk_index")
            inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(
                output, skip_special_tokens=True, audio_chunk_index=chunk_index, language="ar",
            )
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str) or not text.strip():
            raise SafetyError("No speech was recognized. Try a clearer recording.")
        return text.strip()
    except SafetyError:
        raise
    except GatedRepoError as error:
        raise RuntimeError(
            "Hugging Face denied access to Cohere Transcribe Arabic. Accept or request access at "
            f"https://huggingface.co/{ASR_MODEL}, then start Streamlit with HF_TOKEN from "
            "the same approved account. Never paste the token into a message or source file."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Could not download the transcription model. Check Hugging Face access, "
            "HF_TOKEN, and the internet connection."
        ) from error
    except ImportError as error:
        raise RuntimeError(
            "The installed audio libraries do not support this model. "
            "Install requirements-audio.txt and restart Streamlit."
        ) from error
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            "Transcription ran out of memory. Set ASR_DEVICE=cpu before starting Streamlit "
            "to use system RAM instead of GPU memory."
        ) from error
    except Exception as error:
        # Do not expose model cache paths, tokens, or provider tracebacks in the UI.
        raise RuntimeError(
            "Transcription failed while loading or running the model. Check available "
            "memory, the model download, and the audio dependencies."
        ) from error
