"""Small, explicit configuration shared by the backend."""

import os
from dataclasses import dataclass

MAX_INPUT = 2000
MAX_FILE_BYTES = 16_384
MAX_WORKSPACE_FILES = 100
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
MAX_TOOL_CALLS = 3
QWEN_MODEL = "Qwen/Qwen3.5-2B"
ASR_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"


@dataclass(frozen=True)
class Settings:
    qwen_url: str = "http://127.0.0.1:8000/v1"
    qwen_api_key: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            qwen_url=os.getenv("QWEN_BASE_URL", cls.qwen_url).rstrip("/"),
            qwen_api_key=os.getenv("QWEN_API_KEY", ""),
        )
