# /workspace/EditDNA-worker/worker/video.py
from __future__ import annotations
import tempfile
import subprocess

def download_to_local(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    local_path = tmp.name
    tmp.close()

    cmd = ["curl", "-L", "-o", local_path, url]
    subprocess.check_call(cmd)
    return local_path
