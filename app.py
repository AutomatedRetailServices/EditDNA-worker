# worker.py — RQ tasks used by the web app (full file)

import os
import subprocess
import tempfile
import shutil
import uuid
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
import boto3

# --------- tiny helper: safe single-quote for ffmpeg concat ---------
def _sq(path: str) -> str:
    # escape single quotes for ffmpeg concat (POSIX)
    return path.replace("'", "'\\''")


# --------- downloads ---------
def _download_http(url: str, dst_path: str) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _download_s3(s3_url: str, dst_path: str) -> None:
    # s3://bucket/key
    p = urlparse(s3_url)
    bucket = p.netloc
    key = p.path.lstrip("/")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, dst_path)


def _download_to_tmp(urls: List[str], workdir: str) -> List[str]:
    local_paths = []
    for i, u in enumerate(urls):
        ext = os.path.splitext(urlparse(u).path)[1] or ".mov"
        dst = os.path.join(workdir, f"part_{i:03d}{ext}")
        if u.startswith("s3://"):
            _download_s3(u, dst)
        else:
            _download_http(u, dst)
        local_paths.append(dst)
    return local_paths


# --------- ffmpeg concat render ---------
def _render_concat(session_id: str, files: List[str], workdir: str) -> str:
    # 1) write concat file (ffconcat)
    concat_txt = os.path.join(workdir, "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in files:
            f.write(f"file '{_sq(p)}'\n")

    # 2) output path
    out_path = os.path.join(workdir, f"{session_id}.mp4")

    # 3) ffmpeg args
    # NOTE: -ignore_unknown is a flag (NO value). Passing "1" causes the
    # error you saw: “Unable to find a suitable output format for '1'”.
    args = [
        "ffmpeg",
        "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-safe", "0",
        "-f", "concat",
        "-i", concat_txt,
        "-ignore_unknown",
        "-vf",
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]

    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stdout.decode('utf-8', errors='ignore')}")
    return out_path


# --------- upload (optional to S3 if OUTPUT_S3_URL like s3://bucket/prefix is set) ---------
def _maybe_upload_to_s3(local_path: str, output_prefix: str) -> str:
    """
    If output_prefix is an s3:// URL, upload there. Otherwise return local path.
    Accepts either 'editdna/outputs' (returns local path) or 's3://bucket/prefix'.
    """
    if output_prefix.startswith("s3://"):
        p = urlparse(output_prefix)
        bucket = p.netloc
        prefix = p.path.lstrip("/")
        key = f"{prefix.rstrip('/')}/{os.path.basename(local_path)}"
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    return local_path


# =========================  PUBLIC TASKS  =========================

def task_nop() -> Dict[str, Any]:
    # simple smoke test
    return {"echo": {"hello": "world"}}


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload:
      {
        "session_id": "sess-123",
        "files": ["https://.../a.mov", "s3://bucket/b.mov"],
        "output_prefix": "editdna/outputs" OR "s3://my-bucket/editdna/outputs"
      }
    """
    session_id = payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}"
    files_urls = payload.get("files") or []
    output_prefix = payload.get("output_prefix") or "editdna/outputs"

    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    try:
        # download all inputs to local files
        local_inputs = _download_to_tmp(files_urls, workdir)

        # render
        out_local = _render_concat(session_id, local_inputs, workdir)

        # maybe upload
        final_uri = _maybe_upload_to_s3(out_local, output_prefix)

        return {"ok": True, "session_id": session_id, "output": final_uri}
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}
    finally:
        # keep temp dir for debugging by commenting next line if you want
        shutil.rmtree(workdir, ignore_errors=True)


def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    # for now just call the same implementation
    return job_render(payload)
