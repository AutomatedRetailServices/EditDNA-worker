import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------
# Public RQ tasks (these names must stay exactly like this)
# ---------------------------------------------------------------------

def task_nop():
    # Small, fast job to verify the queue/worker plumbing.
    return {"echo": {"hello": "world"}}


def job_render(session_id: str, files: List[str], output_prefix: Optional[str] = "editdna/outputs"):
    """
    Concatenate remote HTTPS videos into a single vertical 1080x1920 MP4.
    - session_id: identifier for temporary work dir and output file name
    - files: list of HTTPS URLs (strings). May be iPhone HEVC/LPCM etc.
    - output_prefix: kept for compatibility; output is written under /tmp
    """
    # Ensure all entries are plain strings
    sources = [str(u) for u in files]

    # Work dir under /tmp to keep things ephemeral
    tmpdir = Path(tempfile.mkdtemp(prefix=f"editdna-sess-{session_id}-"))
    concat_txt = tmpdir / "concat.txt"
    out_mp4 = tmpdir / f"{session_id}.mp4"

    try:
        _write_concat_txt(concat_txt, sources)
        _ffmpeg_concat_to_mp4(concat_txt, out_mp4, portrait=True)
        return {"ok": True, "session_id": session_id, "output": str(out_mp4)}
    except subprocess.CalledProcessError as e:
        # Surface full ffmpeg stdout/stderr for debugging in /jobs
        return {"ok": False, "session_id": session_id, "error": f"ffmpeg failed:\n{e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)}"}
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _write_concat_txt(path: Path, urls: Iterable[str]) -> None:
    """
    Create a concat demuxer file list with HTTPS sources.
    Properly escape single quotes; do not append stray characters.
    """
    with path.open("w", encoding="utf-8") as f:
        for u in urls:
            # escape single quotes for ffmpeg file list
            esc = str(u).replace("'", r"'\'\''")
            f.write(f"file '{esc}'\n")


def _ffmpeg_concat_to_mp4(concat_txt: Path, out_mp4: Path, portrait: bool = True) -> None:
    """
    Run ffmpeg concat demuxer with protocol whitelist for HTTPS.
    Re-encodes video to H.264 + AAC; handles rotation/padding to 1080x1920 when portrait=True.
    """
    vf = []
    if portrait:
        vf = [
            "-vf",
            "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
        ]

    cmd = [
        "ffmpeg",
        "-y",
        # Make remote probing robust, and allow https/tls/tcp
        "-protocol_whitelist", "file,crypto,data,https,tls,tcp",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-safe", "0",
        "-f", "concat",
        "-i", str(concat_txt),
        "-ignore_unknown", "1",
        *vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    # Run and capture stderr for error diagnostics
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    # Optional: return code already checked; if you want logs:
    # print(proc.stderr.decode("utf-8", errors="ignore"))
