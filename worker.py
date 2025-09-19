# worker.py — robust ffmpeg worker (concat HTTPS S3 clips → MP4)

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union


def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]


def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    """Write ffmpeg concat demuxer list file. Quotes and escapes are handled."""
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")


def _run_ffmpeg(cmd: List[str]) -> None:
    """Run ffmpeg and raise with full stderr on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed (exit={proc.returncode}):\n{out}\n{err}")


def _ffmpeg_concat_to_mp4(
    files: Sequence[Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    files = _to_str_list(files)
    output_path = str(output_path)

    with tempfile.TemporaryDirectory(prefix="editdna-") as tmpdir:
        concat_file = Path(tmpdir) / "concat.txt"
        _write_concat_txt(files, concat_file)

        vf_args: List[str] = []
        if portrait:
            vf_args = [
                "-vf",
                "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
            ]

        cmd: List[str] = [
            "ffmpeg",
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            # allow ffmpeg concat demuxer to pull HTTPS clips
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat",
            "-i", str(concat_file),
            "-ignore_unknown",          # ✅ no stray "1"
            *vf_args,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        _run_ffmpeg(cmd)


def task_nop() -> dict:
    return {"echo": {"hello": "world"}}


def _coerce_payload(*args, **kwargs) -> dict:
    """
    Accept either a single dict payload or legacy positional args.
    Returns a normalized payload dict.
    """
    if args and isinstance(args[0], dict):
        p = dict(args[0])
    else:
        p = {
            "session_id": kwargs.get("session_id"),
            "files": kwargs.get("files"),
            "output_prefix": kwargs.get("output_prefix"),
            "portrait": kwargs.get("portrait"),
        }
        # legacy tuple args: (session_id, files, output_prefix)
        if args:
            if p["session_id"] is None and len(args) >= 1:
                p["session_id"] = args[0]
            if p["files"] is None and len(args) >= 2:
                p["files"] = args[1]
            if p["output_prefix"] is None and len(args) >= 3:
                p["output_prefix"] = args[2]

    # defaults
    p.setdefault("session_id", "session")
    p.setdefault("files", [])
    p.setdefault("output_prefix", "editdna/outputs")
    p.setdefault("portrait", True)

    # pass-through (future knobs, harmless if unused)
    for k in ["mode", "max_duration", "take_top_k", "min_clip_seconds", "max_clip_seconds", "drop_silent", "drop_black"]:
        p.setdefault(k, None if k not in ("drop_silent", "drop_black") else True)

    return p


def job_render(*args, **kwargs) -> dict:
    """Main RQ job: concat many HTTPS/S3 clips to a single MP4."""
    p = _coerce_payload(*args, **kwargs)

    session_id = str(p.get("session_id") or "session")
    files = _to_str_list(p.get("files") or [])
    portrait = bool(p.get("portrait", True))

    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / f"{session_id}.mp4"

    try:
        _ffmpeg_concat_to_mp4(files, output_path, portrait=portrait)
        return {"ok": True, "session_id": session_id, "output": str(output_path)}
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}


def job_render_chunked(*args, **kwargs) -> dict:
    """Compat stub: currently same as job_render."""
    return job_render(*args, **kwargs)
