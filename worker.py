import os
import tempfile
import subprocess
import requests
from typing import Dict, List


def task_nop():
    """Tiny test job to verify queue/worker wiring."""
    return {"echo": {"hello": "world"}}


def _download_to_tmp(url: str, session_id: str, idx: int) -> str:
    """Download a remote file to /tmp and return the local path."""
    # Keep the extension (helps ffmpeg input probing)
    ext = os.path.splitext(url.split("?")[0])[-1] or ".mov"
    local_path = f"/tmp/{session_id}_{idx}{ext}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local_path


def job_render(data: Dict) -> Dict:
    """
    Expects:
    {
      "session_id": "sess-demo-001",
      "files": ["https://.../a.mov", "https://.../b.mov", ...],
      "output_prefix": "editdna/outputs"
    }
    Returns:
      {"ok": True, "session_id": "...", "output": "/tmp/.../sess-demo-001.mp4"}
      or {"ok": False, "session_id": "...", "error": "ffmpeg failed: ..."}
    """
    session_id: str = data["session_id"]
    urls: List[str] = data["files"]
    output_prefix: str = data.get("output_prefix", "editdna/outputs")

    # Download all inputs first (avoid ffmpeg HTTPS whitelist issues)
    local_inputs: List[str] = []
    for i, u in enumerate(urls):
        local_inputs.append(_download_to_tmp(u, session_id, i))

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    concat_file = os.path.join(tmpdir, "concat.txt")
    with open(concat_file, "w") as f:
        for path in local_inputs:
            # Escape single quotes in paths for ffmpeg concat demuxer
            safe = path.replace("'", r"'\''")
            f.write(f"file '{safe}'\n")

    output_path = os.path.join(tmpdir, f"{session_id}.mp4")

    # Build ffmpeg command. We intentionally DO NOT pass analyzeduration/probesize
    # overrides here; default 5.1.7 values are usually fine after local download.
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-ignore_unknown", "1",  # ignore odd metadata tracks from iPhone files
        "-vf",
        (
            "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
        ),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return {"ok": True, "session_id": session_id, "output": output_path}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "session_id": session_id, "error": f"ffmpeg failed:\n{e.output.decode(errors='ignore')}"}

