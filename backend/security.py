"""Deterministic checks; model instructions are not a security boundary."""

import re

from backend.config import MAX_FILE_BYTES, MAX_INPUT


class SafetyError(ValueError):
    """A request falls outside this demo's supported capabilities."""


def validate_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT:
        raise SafetyError(f"Enter between 1 and {MAX_INPUT} characters.")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in text):
        raise SafetyError("Control characters are not allowed.")
    return text.strip()


def validate_arguments(name, arguments, schemas):
    if name not in schemas:
        raise SafetyError("This tool is not allowed.")
    if not isinstance(arguments, dict):
        raise SafetyError("Tool arguments must be an object.")
    parameters = schemas[name]["function"]["parameters"]
    properties = parameters["properties"]
    if set(arguments) != set(parameters["required"]):
        raise SafetyError("Missing or unexpected tool arguments.")
    for key, value in arguments.items():
        if not isinstance(value, str):
            raise SafetyError("Tool arguments must be text.")
        rule = properties[key]
        if "enum" in rule and value not in rule["enum"]:
            raise SafetyError("This application is not allowed.")
        if key == "filename":
            # A flat workspace makes traversal and directory permissions simple.
            if not re.fullmatch(r"[\w -]{1,64}\.txt", value, flags=re.UNICODE):
                raise SafetyError("Use a simple .txt filename without directory paths.")
        if key == "content" and len(value.encode("utf-8")) > MAX_FILE_BYTES:
            raise SafetyError("Text file content is too large.")
    return dict(arguments)
