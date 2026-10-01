import datetime
import platform
import subprocess
import sys
from pathlib import Path


def capture_environment(python_executable=None):
    python_executable = python_executable or sys.executable
    try:
        frozen = subprocess.run(
            [python_executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
        )
        pip_freeze = frozen.stdout
    except (FileNotFoundError, OSError):
        pip_freeze = ""
    return {
        "python": sys.version.split()[0],
        "executable": python_executable,
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "pip_freeze": pip_freeze,
    }


def git_commit(root="."):
    try:
        recorded = (Path(root) / "REVISION").read_text().strip()
        if recorded:
            return recorded
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    return result.stdout.strip() or None


def render_environment(environment):
    lines = [
        f"python={environment['python']}",
        f"executable={environment['executable']}",
        f"platform={environment['platform']}",
        f"timestamp={environment['timestamp']}",
        "",
        environment["pip_freeze"],
    ]
    return "\n".join(lines)
