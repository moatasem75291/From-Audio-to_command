# Fortitude — From Audio to Command

A straightforward university project: a Streamlit voice assistant for Ubuntu,
with modular Python backend logic. The overview and six-week implementation plan
are in [PROJECT_PLAN.md](PROJECT_PLAN.md), based on `Fortitude..pdf`.

## Quick start

Use Python 3.10+ for the UI (Python 3.12 is recommended for the model environment).
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

Open `http://127.0.0.1:8501`. The UI runs without model weights installed, but
responses require a running Qwen service as described below. No API key is
required for a local unauthenticated service.

## Qwen generation service

The backend uses **Qwen/Qwen3.5-2B** through its chat-completions HTTP endpoint.
Run a compatible model server in a **separate environment**, on suitable GPU
hardware. This keeps GPU/server dependencies out of the small UI environment.
For example, after installing a Qwen3.5-compatible vLLM build:

```bash
vllm serve Qwen/Qwen3.5-2B \
  --host 127.0.0.1 --port 8000 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Use the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-2B) for compatible
server installation instructions; the card documents vLLM and SGLang tool parsing.
The example uses a smaller context window for the course demo. GPU memory needs
depend on server settings; CPU-only inference may be impractically slow.

Optional environment settings, exported **before** launching Streamlit:

```bash
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
# Set QWEN_API_KEY only if your service requires authentication.
```

`.env.example` lists the settings. The app does not load `.env` automatically.
Only use trusted model endpoints: conversation text and tool results are sent to
the configured endpoint. Remote endpoints must use HTTPS.

## Arabic audio

Audio uses [Cohere Transcribe Arabic](https://cohere.com/blog/transcribe-arabic),
specifically [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026).
The model card documents native Transformers support and requires accepting
Hugging Face access conditions before downloading the weights.

1. Accept the conditions on the model page using your Hugging Face account.
2. Install the optional speech dependencies in the UI environment:

   ```bash
   python -m pip install -r requirements-audio.txt
   ```

3. Set `HF_TOKEN` in the environment to a token authorized for that model, or
   authenticate with the Hugging Face CLI. Do not put tokens in source code.
4. Record or upload a 16-bit PCM WAV, 8–48 kHz, mono/stereo, at most 30 seconds
   and 10 MB. Click **Transcribe audio**, review/edit the transcript, then **Send**.

ASR loads weights lazily on first use, resamples to 16 kHz, and uses Arabic mode.
For a small GPU, set `export ASR_DEVICE=cpu` before starting Streamlit to use
system RAM. `ASR_DEVICE` defaults to `auto` for automatic device placement.
It may use substantial RAM/VRAM. If Qwen and ASR share a GPU, budget memory for
both. Very quiet audio is rejected; this simple check is not full speech detection.
Reviewing the transcript is essential because speech recognition can be wrong.

If transcription fails, the app distinguishes missing dependencies, denied model
access, download errors, and insufficient memory. A valid `HF_TOKEN` alone does
not grant access to gated weights: the same account must also have access on the
model page. Restart Streamlit after changing dependencies or environment settings.

## Available tools

| Tool | Behavior |
| --- | --- |
| `system_info` | OS name and current server-local date/time |
| `list_files` | List up to 100 workspace text files |
| `read_file` | Read a small UTF-8 `.txt` file |
| `create_file` | Create a new `.txt` file, never overwrite |
| `open_application` | Launch calculator or text editor |
| `close_application` | Terminate only a process launched by this session |

Each browser session gets a private temporary workspace. Files survive normal
Streamlit reruns, but are removed on **Clear session** or normal process cleanup.
Do not use them for permanent storage. Application launches require an Ubuntu
desktop with `/usr/bin/gnome-calculator` and `/usr/bin/gedit` installed. They open
on the machine running Streamlit. A desktop app that forwards to another process
may not remain controllable through its launcher process; this app never kills
an arbitrary existing process. Closing an editor can lose unsaved work, so review
the confirmation carefully.

Try: `اعرض معلومات النظام`, `Create a file named notes.txt containing Hello`,
`List my files`, or `افتح الآلة الحاسبة`. Every proposal shows the exact tool name
and arguments and needs **Confirm action**. **Cancel action** executes nothing
for that proposal. A later tool proposal requires another confirmation.

## Activity and conversation

The activity log displays real orchestration events: asking Qwen, waiting for
approval, invoking a tool, its result, and the final response. “Thinking” is a
status label, not a transcript of private model reasoning. The client requests
non-thinking mode and discards separate reasoning fields and `<think>` blocks.

History keeps up to six completed turns within a 24,000-character budget,
including matched tool calls/results. Whole turns are removed when limits are
exceeded; an oversized turn can be omitted. No chat database or transcript log
is created. Clear session resets the history, pending action, and workspace.

## Security boundary

The model proposes actions; Python validates and executes them. Security does
not depend on detecting particular English or Arabic attack phrases.

- Fixed tool and application allowlists, exact argument keys/types, and review
  before every invocation; unknown tools and unexpected arguments fail closed.
- No shell execution, generated Python, arbitrary executable arguments, arbitrary
  PIDs, sudo, deletion, overwrite, power controls, or network tools.
- Flat `.txt` workspace with traversal rejection, no-follow file descriptors,
  exclusive creation, regular-file checks, and hard-link rejection on reads.
- Workspace limited to 100 files of 16 KiB each; messages to 2,000 characters; at most three tool
  proposals per message; bounded history, audio, model output, and HTTP waits.
- Untrusted text is rendered as plain text. Model errors hide provider details.
  Local Streamlit binds to loopback with CORS/XSRF protection enabled.
- Harmful-request refusal is included in the system prompt. Prompt injection
  can still affect model text, so prompts are not claimed to guarantee safe
  conversation content. The allowlist limits actual system effects regardless.

Run as an ordinary user in a disposable Ubuntu desktop VM for the class demo.
This is a local single-user prototype, without accounts, authentication, or
production isolation. Do not expose it publicly or run it as root. The workspace
checks do not defend against an attacker who already controls the same OS account.
Clear sessions after demonstrations to release their files and resources.

## Structure and checks

```text
frontend/app.py       Streamlit UI, transcript review, session state, approvals
backend/assistant.py  Message/tool/confirmation flow
backend/audio.py      WAV checks and Cohere transcription
backend/model.py      Qwen HTTP adapter
backend/tools.py      Schemas and allowlisted actions
backend/security.py   Input and tool-argument checks
backend/history.py    Bounded complete turns
backend/config.py     Constants and environment settings
tests/                Model-free security, flow, and UI checks
```

Run the tests without downloading weights:

```bash
python -m unittest discover -s tests -v
```

The tests cover confirmations, cancellation, duplicate prevention, history,
malformed model responses, path/command injection, symlinks/hard links/special
files, audio validation, and the Streamlit flow using a fake model.
Real Arabic recognition quality, Qwen tool selection, GPU performance, microphone
permissions, and desktop application behavior still need manual end-to-end
checks on the intended Ubuntu machine.

TTS and broader OS control are deferred extensions from the PDF; the implemented
first version focuses on safe voice-to-action interaction.
