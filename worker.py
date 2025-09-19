from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

# RQ job context for progress updates
try:
    from rq import get_current_job
except Exception:  # pragma: no cover
    get_current_job = None  # type: ignore


def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]


def _update_progress(stage: str, current: int = 0, total: int = 0, extra: dict | None = None) -> None:
    """Write progress into job.meta so /jobs returns something useful while running."""
    if get_current_job is None:
        return
    try:
        job = get_current_job()
        if job is None:
            return
        meta = job.meta or {}
        meta["progress"] = {"stage": stage, "current": current, "total": total}
        if extra:
            meta["progress"].update(extra)
        job.meta = meta
        job.save_meta()
    except Exception:
        # progress is best-effort; never crash the job for this
        pass


def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            # ffmpeg concat demuxer: one file per line
            f.write(f"file '{esc}'\n")


def _ffprobe_streams(url: str) -> tuple[bool, str]:
    """
    Quick sanity check on each input. Returns (ok, raw_output_or_error).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
        url,
    ]
    try:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        return True, out.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout or str(e)
    except subprocess.TimeoutExpired:
        return False, "ffprobe timeout"


def _ffmpeg_concat_to_mp4(
    files: Sequence[Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    files = _to_str_list(files)
    output_path = str(output_path)

    with tempfile.TemporaryDirectory(prefix="editdna-") as tmpdir:
        tmp = Path(tmpdir)
        concat_file = tmp / "concat.txt"

        # Validate inputs (lightweight)
        _update_progress("validating_inputs", 0, len(files))
        good_files: List[str] = []
        for i, u in enumerate(files, 1):
            ok, _raw = _ffprobe_streams(u)
            _update_progress("validating_inputs", i, len(files), {"last_checked": u, "ok": ok})
            if ok:
                good_files.append(u)

        if not good_files:
            raise RuntimeError("No valid inputs after ffprobe check.")

        _write_concat_txt(good_files, concat_file)

        vf_args: List[str] = []
        if portrait:
            vf_args = [
                "-vf",
                "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
            ]

        # Build ffmpeg command
        cmd: List[str] = [
            "ffmpeg",
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-rw_timeout", "30000000",  # 30s per network read
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

        _update_progress("ffmpeg_start")
        try:
            # Run ffmpeg; capture logs for debugging
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _update_progress("ffmpeg_done")
        except subprocess.CalledProcessError as e:
            log = e.stdout or str(e)
            raise RuntimeError(f"ffmpeg failed:\n{log}") from e


def task_nop() -> dict:
    return {"echo": {"hello": "world"}}


def job_render(*args, **kwargs) -> dict:
    """
    Accepts either:
      - payload dict as first arg (web enqueues like queue.enqueue('worker.job_render', payload))
      - or classic positional: (session_id, files, output_prefix)
      - or kwargs: session_id=..., files=..., output_prefix=..., portrait=...
    Reports progress via job.meta.
    """
    # Normalize inputs
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

    # Initial progress snapshot
    _update_progress("queued", 0, len(files), {"session_id": session_id})

    try:
        _update_progress("rendering", 0, len(files))
        _ffmpeg_concat_to_mp4(files, output_path, portrait=portrait)
        _update_progress("uploaded_or_ready", len(files), len(files), {"local_output": str(output_path)})
        return {"ok": True, "session_id": session_id, "output": str(output_path), "output_prefix": output_prefix}
    except Exception as e:
        _update_progress("error", 0, len(files), {"message": str(e)})
        return {"ok": False, "session_id": session_id, "error": str(e)}


def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
