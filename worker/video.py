# /workspace/EditDNA-worker/worker/video.py
from __future__ import annotations
import os, uuid, subprocess, shutil
from typing import Optional

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
DOWNLOAD_ROOT = os.getenv("DOWNLOAD_ROOT", "/tmp")

def make_temp_video_path(suffix: str = ".mp4") -> str:
    os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
    return os.path.join(DOWNLOAD_ROOT, f"vid_{uuid.uuid4().hex}{suffix}")

def download_to_local(src: str, dst: Optional[str] = None) -> str:
    if os.path.exists(src):
        return src
    if dst is None:
        dst = make_temp_video_path()
    # http(s)
    if src.startswith("http://") or src.startswith("https://"):
        subprocess.check_call(["curl", "-L", "-o", dst, src])
        return dst
    # s3 – if you have worker.s3
    if src.startswith("s3://"):
        from . import s3 as s3mod
        s3mod.download_file(src, dst)
        return dst
    # fallback copy
    shutil.copy(src, dst)
    return dst

def probe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            [
                FFPROBE_BIN,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception:
        return 0.0
