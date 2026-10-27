# Fortitude project plan

## Goal
Build a small university demonstration of the voice-controlled Ubuntu assistant
described in `Fortitude..pdf`. Keep Streamlit separate from the core Python logic.
This is an assistant running on an existing OS, rather than a new operating system.

## Flow
1. Record or upload a short WAV clip, or type an Arabic/English message.
2. Cohere Transcribe Arabic converts audio to editable text.
3. Qwen3.5-2B receives the message, recent conversation, and tool definitions.
4. The backend validates each proposed tool call. Show its exact arguments and
   require a user click before executing it, including read-only tools.
5. Return the tool result to Qwen and display the final response.
6. Show live status and tool results; do not display private chain-of-thought.

## Simple modules
- `frontend/app.py`: Streamlit UI, session state, audio review, confirmations.
- `backend/assistant.py`: conversation flow and bounded tool calling.
- `backend/model.py`: Qwen chat-completions HTTP client.
- `backend/audio.py`: WAV validation and lazy Cohere model loading.
- `backend/tools.py`: tool schemas and small allowlisted handlers.
- `backend/security.py`: strict argument and workspace validation.
- `backend/history.py`: bounded, complete conversation turns.
- `backend/config.py`: environment settings and fixed limits.
- `tests/`: security and integration tests using fake models; no weight download.

## Initial features and boundaries
- System date/info, list/read/create text files in a dedicated session workspace.
- Launch an allowlisted calculator or text editor; close only processes launched
  by this session. No terminal, arbitrary shell command, browser, or arbitrary PID.
- Pending calls are validated again on confirmation and never run automatically.
- No deletion, overwrite, sudo, software installation, credentials access, network
  tools, power controls, or arbitrary code execution.
- Limits on audio size/duration, prompts, files, history, calls, and HTTP requests.
- Input and tool results are untrusted. Prompts help refusal; Python permissions
  and allowlists enforce the actual execution boundary, regardless of language.
- Session-only history, isolated workspaces, no transcript logging or persistence.
- Local single-user classroom use. This is not a hardened multi-user service;
  run unprivileged on loopback, ideally inside a disposable desktop VM.
- TTS and more OS controls are later extensions, after the core demo is tested.

## Implementation sequence (matching the PDF's six weeks)
1. Planning: define the architecture, tool scope, security boundary, and team roles.
2. Environment: Python virtualenv, dependencies, Ubuntu desktop applications.
3. Speech: validated WAV input, Cohere ASR, transcript review before submission.
4. NLP: Qwen service integration, tool schemas, bounded session history.
5. Actions: safe handlers, preview/confirm/cancel UI, model follow-up on results.
6. Integration: fake-model tests, adversarial tool arguments, UI smoke test,
   then manual Arabic/English model and Ubuntu desktop checks.

## Acceptance checks
- Text works without installing speech/model-server dependencies.
- Arabic audio is reviewable before any action is proposed.
- Confirm and cancel work across Streamlit reruns without duplicate execution.
- Unknown tools, extra keys, traversal, symlinks, and shell payloads cannot execute.
- Follow-up questions receive recent completed turns and actual tool results.
- Failures show useful generic errors without disclosing stack traces or secrets.
- Document clearly which checks require model access and a desktop/GPU.

## References
- Overview: `Fortitude..pdf`.
- https://cohere.com/blog/transcribe-arabic
- https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
  (Transformers >=5.4, gated weights, Arabic language option).
- https://huggingface.co/Qwen/Qwen3.5-2B
  (chat-completions service, `qwen3_coder` tool parser, non-thinking default).
