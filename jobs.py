# jobs.py — ffmpeg helpers used by worker.py
import os
import tempfile
import subprocess
import shutil
from typing import List, Dict, Any

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def _write_concat_file(tmpdir: str, files: List[str]) -> str:
    """
    Write an ffconcat-style list file for the concat demuxer.
    We avoid f-strings with backslashes to prevent SyntaxError issues.
    """
    list_path = os.path.join(tmpdir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            # escape single quotes for concat syntax
            safe = str(p).replace("'", "'\\''")
            f.write("file '{}'\n".format(safe))
    return list_path


def _run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n---\n{proc.stdout}\n---")


def _make_out(tmpdir: str, session_id: str) -> str:
    return os.path.join(tmpdir, f"{session_id}.mp4")


def job_render(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    """
    Simple vertical 1080x1920 render using concat demuxer.
    Returns local temp output path for now (you can S3-upload later).
    """
    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    try:
        concat_txt = _write_concat_file(tmpdir, files)
        out_path = _make_out(tmpdir, session_id)

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-safe", "0",
            "-f", "concat",
            "-i", concat_txt,
            "-ignore_unknown", "1",
            "-vf", "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        _run_ffmpeg(cmd)

        return {
            "ok": True,
            "session_id": session_id,
            "output_local": out_path,
            "output_prefix": output_prefix,
            "inputs": files,
        }
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}
    finally:
        # Keep tmpdir for debugging; comment next line to persist files
        shutil.rmtree(tmpdir, ignore_errors=True)


def job_render_chunked(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    """
    Placeholder: same behavior as job_render for now.
    """
    return job_render(session_id, files, output_prefix)
