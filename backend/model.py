"""Qwen service adapter; it has no access to tool execution."""

import json
import re
from urllib.parse import urlparse

import requests

from backend.config import QWEN_MODEL, Settings


class ModelError(RuntimeError):
    pass


def final_text(content):
    if not isinstance(content, str):
        return ""
    # Some servers put reasoning in content rather than a separate field.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)
    return content.strip()[:4000]


class QwenClient:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        url = urlparse(self.settings.qwen_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ModelError("Invalid QWEN_BASE_URL configuration.")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ModelError("Remote model services must use HTTPS.")

    def reply(self, messages, tools):
        payload = {
            "model": QWEN_MODEL, "messages": messages, "tools": tools,
            "tool_choice": "auto", "max_tokens": 1024,
            "temperature": 1.0, "top_p": 1.0, "presence_penalty": 2.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {}
        if self.settings.qwen_api_key:
            headers["Authorization"] = f"Bearer {self.settings.qwen_api_key}"
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.post(
                    self.settings.qwen_url + "/chat/completions", json=payload,
                    headers=headers, timeout=(5, 90), allow_redirects=False, stream=True,
                ) as response:
                    if response.status_code != 200:
                        raise ModelError("Qwen service rejected the request. Check its configuration.")
                    data = bytearray()
                    for chunk in response.iter_content(8192):
                        data.extend(chunk)
                        if len(data) > 131_072:
                            raise ModelError("Qwen response exceeded the size limit.")
            result = json.loads(data)
            message = result["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError
            return message
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelError("Could not read a valid Qwen response. Check the local model service.") from error
