"""Small allowlisted tools. No shell, eval, arbitrary paths, or arbitrary PIDs."""

import os
import platform
import stat
import subprocess
import tempfile
from datetime import datetime

from backend.config import MAX_FILE_BYTES, MAX_WORKSPACE_FILES
from backend.security import SafetyError, validate_arguments

APPLICATIONS = {
    "calculator": ["/usr/bin/gnome-calculator"],
    "text_editor": ["/usr/bin/gedit", "--standalone"],
}


def schema(name, description, properties=None):
    properties = properties or {}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
    }}


FILENAME = {"type": "string", "description": "Simple .txt filename, no directories."}
APP = {"type": "string", "enum": list(APPLICATIONS)}
TOOL_SCHEMAS = {item["function"]["name"]: item for item in [
    schema("system_info", "Get OS name and current local date/time."),
    schema("list_files", "List up to 100 text files in this session's workspace."),
    schema("read_file", "Read a small UTF-8 text file in the session workspace.",
           {"filename": FILENAME}),
    schema("create_file", "Create a new text file; never overwrite an existing file.",
           {"filename": FILENAME, "content": {"type": "string"}}),
    schema("open_application", "Open an allowlisted Ubuntu desktop application.", {"app": APP}),
    schema("close_application", "Close only the application process opened by this session.",
           {"app": APP}),
]}


class ToolRunner:
    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="fortitude-")
        self.workspace = self._directory.name
        self.processes = {}

    def validate(self, name, arguments):
        return validate_arguments(name, arguments, TOOL_SCHEMAS)

    def execute(self, name, arguments):
        arguments = self.validate(name, arguments)
        try:
            if name == "system_info":
                return {"os": platform.system(), "time": datetime.now().astimezone().isoformat()}
            if name == "list_files":
                with os.scandir(self.workspace) as entries:
                    files = []
                    for entry in entries:
                        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".txt"):
                            files.append(entry.name)
                        if len(files) == 100:
                            break
                return {"files": sorted(files)}
            if name in {"read_file", "create_file"}:
                return self._file(name, arguments)
            return self._application(name, arguments["app"])
        except (OSError, UnicodeError, subprocess.SubprocessError):
            return {"error": "Action failed. Check the file, application, or desktop availability."}

    def _file(self, name, arguments):
        # dir_fd + O_NOFOLLOW prevents following a substituted file symlink.
        directory = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if name == "create_file":
                with os.scandir(directory) as entries:
                    if sum(1 for _ in entries) >= MAX_WORKSPACE_FILES:
                        raise SafetyError("Workspace file limit reached. Clear the session to start again.")
            flags = os.O_NOFOLLOW | os.O_NONBLOCK
            flags |= os.O_WRONLY | os.O_CREAT | os.O_EXCL if name == "create_file" else os.O_RDONLY
            descriptor = os.open(arguments["filename"], flags, 0o600, dir_fd=directory)
            with os.fdopen(descriptor, "wb" if name == "create_file" else "rb") as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise SafetyError("Only ordinary, unlinked workspace files are supported.")
                if name == "create_file":
                    file.write(arguments["content"].encode("utf-8"))
                    return {"created": arguments["filename"]}
                data = file.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise SafetyError("File is too large to read.")
                return {"content": data.decode("utf-8")}
        finally:
            os.close(directory)

    def _application(self, name, app):
        process = self.processes.get(app)
        if name == "close_application":
            if process is None or process.poll() is not None:
                return {"error": "No running application owned by this session."}
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                return {"error": "Application did not close; close it manually on the desktop."}
            del self.processes[app]
            return {"closed": app}
        if process is not None and process.poll() is None:
            return {"status": "Already running", "app": app}
        self.processes[app] = subprocess.Popen(
            APPLICATIONS[app], shell=False, cwd=self.workspace,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"launched": app}

    def cleanup(self):
        for app in list(self.processes):
            self.execute("close_application", {"app": app})
        self._directory.cleanup()
