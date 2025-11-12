# worker/utils.py
import os
import json
import subprocess
from datetime import datetime, timezone

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

# Allow longer single-take spans by default
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 60.0)

def ensure_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ffprobe_duration(path: str) -> float:
    """
    Returns media duration in seconds using ffprobe.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ]
        out = subprocess.check_output(cmd).decode("utf-8", "ignore")
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", 0.0))
        return max(0.0, dur)
    except Exception:
        return 0.0
