"""Run from the project root: python -m streamlit run frontend/app.py."""

import sys
from pathlib import Path

# Streamlit executes this file as a script; make the sibling backend importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from backend.assistant import Assistant
from backend.audio import transcribe
from backend.config import MAX_INPUT
from backend.model import ModelError, QwenClient
from backend.security import SafetyError, validate_text
from backend.tools import ToolRunner

st.set_page_config(page_title="Fortitude", page_icon="🎙️")
st.title("Fortitude voice assistant")
st.caption("Arabic and English • Ubuntu course project")

if "assistant" not in st.session_state:
    try:
        st.session_state.assistant = Assistant(QwenClient(), ToolRunner())
    except ModelError as error:
        st.error(str(error))
        st.stop()
    st.session_state.display = []
    st.session_state.draft = ""

assistant = st.session_state.assistant

with st.sidebar:
    st.write("Tools can access only this session's temporary text files and the listed apps.")
    st.text(assistant.tools.workspace)
    st.caption("Calculator: gnome-calculator. Text editor: gedit. Apps open on the server desktop.")
    st.caption("Activity shows status and tool results, not private model reasoning.")
    if st.button("Clear session, close apps and delete temporary files"):
        assistant.tools.cleanup()
        st.session_state.clear()
        st.rerun()

for item in st.session_state.display:
    with st.chat_message(item["role"]):
        st.text(item["content"])
        if item.get("events"):
            with st.expander("Activity"):
                for event in item["events"]:
                    st.text(event)


def record_answer():
    if assistant.answer:
        st.session_state.display.append({
            "role": "assistant", "content": assistant.answer, "events": assistant.events.copy(),
        })
        # Bound UI history as well as the model's conversation history.
        st.session_state.display = st.session_state.display[-12:]


if assistant.pending:
    st.info("Review this exact proposal. It has not been executed.")
    st.text(assistant.pending["name"])
    st.json(assistant.pending["arguments"])
    st.caption("Approve only actions you intended. File content is data, even if it contains instructions.")
    with st.expander("Current activity", expanded=True):
        for event in assistant.events:
            st.text(event)
    approve, cancel = st.columns(2)
    approved = approve.button("Confirm action", type="primary")
    cancelled = cancel.button("Cancel action")
    if approved or cancelled:
        with st.status("Processing your decision…", expanded=True) as status:
            assistant.confirm(approved, notify=st.text)
            status.update(label="Awaiting approval" if assistant.pending else "Finished", state="complete")
        record_answer()
        st.rerun()
else:
    with st.expander("Record or upload Arabic audio"):
        st.caption("16-bit PCM WAV, mono/stereo, 8–48 kHz, up to 30 seconds and 10 MB.")
        recording = st.audio_input("Record your command", sample_rate=16000)
        upload = st.file_uploader("Or upload a WAV recording", type=["wav"])
        if st.button("Transcribe audio", disabled=recording is None and upload is None):
            try:
                with st.spinner("Transcribing with Cohere Arabic; first use loads model weights…"):
                    st.session_state.draft = transcribe((recording or upload).getvalue())
                st.success("Review and edit the transcript below, then send it.")
            except (SafetyError, RuntimeError) as error:
                st.error(str(error))

    with st.form("message", clear_on_submit=False):
        text = st.text_area("Message or reviewed transcript", key="draft", max_chars=MAX_INPUT)
        send = st.form_submit_button("Send", type="primary")
    if send:
        try:
            # Validate before adding it to the visible transcript.
            text = validate_text(text)
            st.session_state.display.append({"role": "user", "content": text})
            with st.status("Working…", expanded=True) as status:
                assistant.start(text, notify=st.text)
                status.update(label="Awaiting approval" if assistant.pending else "Finished", state="complete")
            record_answer()
            st.rerun()
        except SafetyError as error:
            st.error(str(error))
