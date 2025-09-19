# worker.py — full replacement

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union


def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]


def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")


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
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat",
            "-i", str(concat_file),
            "-ignore_unknown",
            *vf_args,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg failed:\n{err}") from e


def task_nop() -> dict:
    return {"echo": {"hello": "world"}}


def job_render(*args, **kwargs) -> dict:
    # Accept either a single dict payload (preferred) or legacy positional args
    if args and isinstance(args[0], dict):
        payload = dict(args[0])
        session_id = payload.get("session_id") or payload.get("sid") or "session"
        files = payload.get("files") or []
        output_prefix = payload.get("output_prefix") or "editdna/outputs"
        portrait = bool(payload.get("portrait", True))
    else:
        session_id = kwargs.get("session_id")
        files = kwargs.get("files")
        output_prefix = kwargs.get("output_prefix", "editdna/outputs")
        portrait = bool(kwargs.get("portrait", True))
        if args:
            if session_id is None and len(args) >= 1:
                session_id = args[0]
            if files is None and len(args) >= 2:
                files = args[1]
            if len(args) >= 3:
                output_prefix = args[2]

    session_id = str(session_id or "session")
    files = _to_str_list(files or [])

    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / f"{session_id}.mp4"

    try:
        _ffmpeg_concat_to_mp4(files, output_path, portrait=portrait)
        return {"ok": True, "session_id": session_id, "output": str(output_path)}
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}


def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
