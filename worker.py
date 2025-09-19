# worker.py — full replacement with S3 upload

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

# S3 helpers (expects env: AWS_REGION, S3_BUCKET, creds)
from s3_utils import upload_file as s3_upload


def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]


def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    """
    Build concat-demuxer list file:
      file '...'
      file '...'
    """
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")


def _ffmpeg_concat_to_mp4(
    files: Sequence[Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    """
    Concatenate files using ffmpeg concat demuxer and re-encode to H.264 + AAC MP4.
    Supports https inputs by whitelisting protocols.
    """
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
            "-ignore_unknown",            # important: no trailing value
            *vf_args,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg failed:\n{err}") from e


def task_nop() -> dict:
    return {"echo": {"hello": "world"}}


def _unpack_payload(*args, **kwargs):
    """
    Accepts either:
      - job_render({"session_id":..., "files":[...], "output_prefix": "...", "portrait": True})
      - job_render(session_id, files, output_prefix, portrait=True)
      - job_render(session_id=session_id, files=[...], output_prefix="...", portrait=True)
    """
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
            if len(args) >= 3 and output_prefix == "editdna/outputs":
                output_prefix = args[2]

    return str(session_id or "session"), _to_str_list(files or []), str(output_prefix), bool(portrait)


def job_render(*args, **kwargs) -> dict:
    session_id, files, output_prefix, portrait = _unpack_payload(*args, **kwargs)

    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / f"{session_id}.mp4"

    try:
        _ffmpeg_concat_to_mp4(files, output_path, portrait=portrait)

        # Upload to S3 and return the S3 URI
        s3_uri = s3_upload(str(output_path), key_prefix=output_prefix, content_type="video/mp4")

        # Optional: clean up tmp output now that it’s in S3
        try:
            output_path.unlink(missing_ok=True)
            # Remove directory if empty
            if not any(workdir.iterdir()):
                workdir.rmdir()
        except Exception:
            pass

        return {
            "ok": True,
            "session_id": session_id,
            "output": s3_uri,           # <-- final S3 location
            "inputs": files,
        }
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}


def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
