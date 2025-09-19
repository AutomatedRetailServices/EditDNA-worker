# worker.py — supports "concat" and a simple "best" mode with S3 upload

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Union

import subprocess

from s3_utils import upload_file as s3_upload, presigned_url, parse_s3_url, S3_BUCKET

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

# ------------------------------
# Small helpers
# ------------------------------
def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]

def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}\n---\n{out}\n---") from e

def _vf_portrait(enabled: bool) -> List[str]:
    if not enabled:
        return []
    return [
        "-vf",
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ]

# ------------------------------
# ffmpeg building blocks
# ------------------------------
def _ffmpeg_concat_to_mp4(
    files: Sequence[Union[str, os.PathLike]],
    dst: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    files = _to_str_list(files)
    dst = str(dst)
    with tempfile.TemporaryDirectory(prefix="editdna-") as tmpdir:
        concat_file = Path(tmpdir) / "concat.txt"
        _write_concat_txt(files, concat_file)

        cmd: List[str] = [
            FFMPEG,
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat",
            "-i", str(concat_file),
            "-ignore_unknown",
            * _vf_portrait(portrait),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            dst,
        ]
        _run(cmd)

def _ffmpeg_trim_one(
    src: str,
    dst: Union[str, os.PathLike],
    start: float,
    dur: float,
    portrait: bool = True,
) -> None:
    dst = str(dst)
    cmd: List[str] = [
        FFMPEG,
        "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", src,
        "-t", f"{max(0.05, dur):.3f}",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
        "-ignore_unknown",
        * _vf_portrait(portrait),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dst,
    ]
    _run(cmd)

def _probe_loudness_db(src: str, sample_seconds: float = 6.0) -> float:
    """
    Use ffmpeg volumedetect to estimate mean loudness in dB (higher is louder).
    We only analyze the first few seconds to keep it fast.
    """
    cmd = [
        FFMPEG,
        "-t", f"{sample_seconds:.2f}",
        "-i", src,
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    # volumedetect prints to stderr
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        text = proc.stderr
    except subprocess.CalledProcessError as e:
        text = (e.stderr or "") + (e.stdout or "")

    # Look for "mean_volume: -23.4 dB"
    m = re.search(r"mean_volume:\s*([\-0-9\.]+)\s*dB", text)
    if not m:
        # fallback if not found
        return -math.inf
    return float(m.group(1))

# ------------------------------
# Public RQ tasks
# ------------------------------
def task_nop() -> dict:
    return {"echo": {"hello": "world"}}

def _upload_and_presign(local_path: str, key_prefix: str) -> Tuple[str, str]:
    s3_uri = s3_upload(local_path, key_prefix=key_prefix, content_type="video/mp4")
    bucket, key = parse_s3_url(s3_uri)
    # parse_s3_url may return (None, key) if it can’t infer bucket; default to env
    bkt = bucket or S3_BUCKET
    url = presigned_url(bkt, key, expires=3600)
    return s3_uri, url

def _assemble_best(
    files: List[str],
    session_id: str,
    portrait: bool,
    *,
    max_duration: int | None,
    take_top_k: int | None,
    min_clip_seconds: float | None,
    max_clip_seconds: float | None,
) -> str:
    """
    'Best' heuristic:
      - rank by loudness (volumedetect on first ~6s)
      - keep top_k
      - trim each clip between min/max seconds
      - stop once we hit max_duration
    Returns path to a local temp mp4 ready to upload.
    """
    # Defaults
    top_k = max(1, int(take_top_k or 8))
    min_s = float(min_clip_seconds or 2.5)
    max_s = float(max_clip_seconds or 10.0)
    total_cap = int(max_duration or 60)

    # rank by loudness
    scored: List[Tuple[float, str]] = []
    for src in files:
        loud = _probe_loudness_db(src)
        scored.append((loud, src))
    scored.sort(reverse=True, key=lambda t: t[0])
    chosen = [s for _, s in scored[:top_k]]

    with tempfile.TemporaryDirectory(prefix=f"editdna-best-{session_id}-") as tmpdir:
        parts: List[str] = []
        remain = total_cap
        for src in chosen:
            if remain <= 0:
                break
            clip_len = min(max_s, remain)
            if clip_len < min_s and parts:
                break  # we're out of budget; keep what we have
            clip_len = max(min_s, min(clip_len, max_s))

            out_seg = str(Path(tmpdir) / f"seg-{len(parts):03d}.mp4")
            _ffmpeg_trim_one(src, out_seg, start=0.0, dur=clip_len, portrait=portrait)
            parts.append(out_seg)
            remain -= int(clip_len)

        # fall back to concat everything if nothing made it
        if not parts:
            parts = chosen[:1] or files[:1]

        final_path = str(Path(tmpdir) / f"{session_id}.mp4")
        _ffmpeg_concat_to_mp4(parts, final_path, portrait=portrait)

        # Move to a stable temp dir so caller can upload after context closes
        stable_dir = Path(f"/tmp/editdna-{session_id}")
        stable_dir.mkdir(parents=True, exist_ok=True)
        stable_path = stable_dir / f"{session_id}.mp4"
        Path(final_path).replace(stable_path)
        return str(stable_path)

def job_render(*args, **kwargs) -> dict:
    """
    Accepts either a single dict payload (from FastAPI) or positional
    (session_id, files, output_prefix).
    """
    if args and isinstance(args[0], dict):
        payload = dict(args[0])
        session_id = payload.get("session_id") or payload.get("sid") or "session"
        files = _to_str_list(payload.get("files") or [])
        output_prefix = payload.get("output_prefix") or "editdna/outputs"
        portrait = bool(payload.get("portrait", True))
        mode = (payload.get("mode") or "concat").lower()
        max_duration = payload.get("max_duration")
        take_top_k = payload.get("take_top_k")
        min_clip_seconds = payload.get("min_clip_seconds")
        max_clip_seconds = payload.get("max_clip_seconds")
        drop_silent = payload.get("drop_silent")
        drop_black = payload.get("drop_black")
    else:
        # legacy signature
        session_id = kwargs.get("session_id") or (args[0] if args else "session")
        files = _to_str_list(kwargs.get("files") or (args[1] if len(args) > 1 else []))
        output_prefix = kwargs.get("output_prefix", "editdna/outputs")
        portrait = bool(kwargs.get("portrait", True))
        mode = (kwargs.get("mode") or "concat").lower()
        max_duration = kwargs.get("max_duration")
        take_top_k = kwargs.get("take_top_k")
        min_clip_seconds = kwargs.get("min_clip_seconds")
        max_clip_seconds = kwargs.get("max_clip_seconds")
        drop_silent = kwargs.get("drop_silent")
        drop_black = kwargs.get("drop_black")

    session_id = str(session_id or "session")

    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "best":
            local_final = _assemble_best(
                files,
                session_id,
                portrait,
                max_duration=max_duration,
                take_top_k=take_top_k,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
            )
            s3_uri, url = _upload_and_presign(local_final, key_prefix=output_prefix)
            return {
                "ok": True,
                "session_id": session_id,
                "mode": "best",
                "output_s3": s3_uri,
                "output_url": url,
                "inputs": files,
            }

        if mode == "split":
            # Export each as its own portrait mp4 and upload
            outputs = []
            for idx, src in enumerate(files):
                seg_path = str(workdir / f"{session_id}-{idx:03d}.mp4")
                _ffmpeg_trim_one(src, seg_path, start=0.0, dur=max_clip_seconds or 10.0, portrait=portrait)
                s3_uri, url = _upload_and_presign(seg_path, key_prefix=output_prefix)
                outputs.append({"index": idx, "source": src, "output_s3": s3_uri, "output_url": url})
            return {"ok": True, "session_id": session_id, "mode": "split", "outputs": outputs, "count": len(outputs)}

        # default: concat in given order
        final_local = str(workdir / f"{session_id}.mp4")
        _ffmpeg_concat_to_mp4(files, final_local, portrait=portrait)
        s3_uri, url = _upload_and_presign(final_local, key_prefix=output_prefix)
        return {
            "ok": True,
            "session_id": session_id,
            "mode": "concat",
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": files,
        }

    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}

def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
