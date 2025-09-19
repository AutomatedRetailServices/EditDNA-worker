from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

from s3_utils import upload_file as s3_upload, new_session_id
import mimetypes

def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]

def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")

def _run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        raise RuntimeError(err) from e

def _vf_portrait_args(enabled: bool) -> List[str]:
    if not enabled:
        return []
    return [
        "-vf",
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ]

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
            *_vf_portrait_args(portrait),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        _run_ffmpeg(cmd)

def _ffmpeg_single_to_mp4(
    src: Union[str, os.PathLike],
    dst: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
        "-i", str(src),
        "-ignore_unknown",
        *_vf_portrait_args(portrait),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    _run_ffmpeg(cmd)

def _presign(url_s3: str) -> str:
    """
    Quick pre-sign via boto3 using s3_utils' upload target; we rely on upload_file()
    returning s3://bucket/key. Re-create the HTTPS form to sign.
    """
    # We keep pre-signing inside s3_utils in your repo; if you already return
    # output_url from there, you can delete this helper. For now we just re-use
    # boto3 client from s3_utils through upload call side-effects.
    # (Kept minimal: app already shows working pre-signed URL.)
    return url_s3  # app already constructs output_url in s3_utils.upload_file()

def task_nop() -> dict:
    return {"echo": {"hello": "world"}}

def job_render(*args, **kwargs) -> dict:
    # Accept both dict payload and positional signature
    if args and isinstance(args[0], dict):
        payload = dict(args[0])
        session_id = payload.get("session_id") or payload.get("sid") or new_session_id()
        files = payload.get("files") or []
        output_prefix = payload.get("output_prefix") or "editdna/outputs"
        portrait = bool(payload.get("portrait", True))
        mode = (payload.get("mode") or "concat").lower()
    else:
        session_id = kwargs.get("session_id") or new_session_id()
        files = kwargs.get("files") or []
        output_prefix = kwargs.get("output_prefix", "editdna/outputs")
        portrait = bool(kwargs.get("portrait", True))
        mode = (kwargs.get("mode") or "concat").lower()
        if args:
            if session_id is None and len(args) >= 1: session_id = args[0]
            if not files and len(args) >= 2: files = args[1]
            if len(args) >= 3: output_prefix = args[2]

    session_id = str(session_id)
    files = _to_str_list(files or [])

    # Workdir
    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "split":
            outputs: List[dict] = []
            for idx, src in enumerate(files, start=1):
                local_out = workdir / f"{session_id}-{idx:03d}.mp4"
                _ffmpeg_single_to_mp4(src, local_out, portrait=portrait)
                # Upload each
                url_s3 = s3_upload(str(local_out), key_prefix=output_prefix, content_type="video/mp4")
                outputs.append(
                    {
                        "index": idx,
                        "source": src,
                        "output_s3": url_s3,
                        # If your s3_utils already returns a presigned URL, replace next line
                        "output_url": None,
                    }
                )
            return {
                "ok": True,
                "session_id": session_id,
                "mode": "split",
                "outputs": outputs,
                "count": len(outputs),
            }

        # default: concat
        final_local = workdir / f"{session_id}.mp4"
        _ffmpeg_concat_to_mp4(files, final_local, portrait=portrait)
        url_s3 = s3_upload(str(final_local), key_prefix=output_prefix, content_type="video/mp4")
        return {
            "ok": True,
            "session_id": session_id,
            "mode": "concat",
            "output_s3": url_s3,
            # If your s3_utils.upload_file adds a presigned HTTPS in return value,
            # you can include it here. Otherwise the web app adds it to response.
            "inputs": files,
        }

    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": f"ffmpeg failed:\n{e}"}

def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
