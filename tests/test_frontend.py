"""Exercise Streamlit reruns with a fake model; no inference service needed."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


class FrontendTests(unittest.TestCase):
    def test_review_confirm_followup_and_clear(self):
        proposal = {"tool_calls": [{"id": "call_ui", "type": "function", "function": {
            "name": "create_file", "arguments": json.dumps({
                "filename": "notes.txt", "content": "Hello from the UI",
            }),
        }}]}
        with patch("backend.model.QwenClient.reply", side_effect=[
            proposal, {"content": "Created notes.txt."},
        ]):
            app = AppTest.from_file(str(APP_PATH)).run()
            self.assertFalse(app.exception)
            app.text_area[0].input("Create notes.txt with a greeting")
            next(button for button in app.button if button.label == "Send").click().run()
            self.assertFalse(app.exception)
            assistant = app.session_state["assistant"]
            workspace = Path(assistant.tools.workspace)
            self.addCleanup(assistant.tools.cleanup)
            self.assertIsNotNone(assistant.pending)
            self.assertFalse((workspace / "notes.txt").exists())
            next(button for button in app.button if button.label == "Confirm action").click().run()
            self.assertFalse(app.exception)
            self.assertEqual((workspace / "notes.txt").read_text(), "Hello from the UI")
            self.assertIsNone(assistant.pending)
            self.assertTrue(any(text.value == "Created notes.txt." for text in app.text))
            next(button for button in app.button if button.label.startswith("Clear session")).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(workspace.exists())
            self.assertEqual(app.session_state["display"], [])
            app.session_state["assistant"].tools.cleanup()

    def test_cancel_does_not_launch_application(self):
        proposal = {"tool_calls": [{"id": "call_ui", "type": "function", "function": {
            "name": "open_application", "arguments": '{"app": "calculator"}',
        }}]}
        with patch("backend.model.QwenClient.reply", return_value=proposal), \
                patch("backend.tools.subprocess.Popen") as spawn:
            app = AppTest.from_file(str(APP_PATH)).run()
            self.addCleanup(app.session_state["assistant"].tools.cleanup)
            app.text_area[0].input("Open calculator")
            next(button for button in app.button if button.label == "Send").click().run()
            next(button for button in app.button if button.label == "Cancel action").click().run()
            self.assertFalse(app.exception)
            spawn.assert_not_called()
            self.assertIn("cancelled", app.session_state["assistant"].answer)


if __name__ == "__main__":
    unittest.main()
