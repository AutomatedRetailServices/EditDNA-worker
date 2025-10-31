import os
import tempfile
import subprocess
from typing import List


def _run(cmd: List[str]) -> None:
    """
    Tiny shell runner that raises on error.
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {err.strip()}")


def make_tmpfile(suffix: str = ".mp4") -> str:
    """
    Create a secure temp file path we control.
    We close() immediately; caller writes later.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def download_video_to_local(url: str) -> str:
    """
    Download remote video URL (public S3 or presigned) into /tmp/...
    Uses curl because the pod already has curl in your start script.
    """
    local_path = make_tmpfile(suffix=".mp4")
    _run(["curl", "-L", "-o", local_path, url])
    return local_path
