# jobs.py — OpenAI analyze + robust chunked video render for EditDNA.ai
import os
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
import uuid

import requests

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client = None
try:
    from openai import OpenAI  # type: ignore
    if OPENAI_API_KEY:
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            _client = None
except Exception:
    OpenAI = None  # type: ignore
    _client = None

_S3_BUCKET = os.getenv("S3_BUCKET", "")
_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

def _s3_client():
    if not _S3_BUCKET:
        return None
    if boto3 is None:
        raise RuntimeError("boto3 not available but S3_BUCKET is set")
    return boto3.client("s3", region_name=_AWS_REGION)

def analyze_session(session_id: str,
                    product_link: str,
                    features_csv: str,
                    tone: str = "casual") -> Dict[str, Any]:
    features_csv = (features_csv or "").strip()
    features = [p.strip() for p in features_csv.split(",") if p.strip()] if features_csv else []
    feat_str = ", ".join(features) if features else "its key features"

    prompt = (
        f"You are a marketing writer. Create a short {tone or 'neutral'} promo script "
        f"for product {product_link or '(no link)'}. Focus on {feat_str}. "
        f"Keep it 3–5 sentences, engaging, and suitable for voiceover."
    )
    stub = {
        "session_id": session_id,
        "engine": "stub",
        "script": (
            f"[STUB] {tone} promo for {product_link or '(no link)'} highlighting "
            f"{feat_str}. Keep it upbeat and concise!"
        ),
    }
    if not _client:
        return stub
    try:
        r = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You write concise, friendly promo scripts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=220,
        )
        text = (r.choices[0].message.content or "").strip()
        return {"session_id": session_id, "engine": "openai", "script": text}
    except Exception as e:
        out = dict(stub)
        out["error"] = str(e)
        return out

def _basename_from_url(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return name or f"file-{uuid.uuid4().hex}"

def _download_to(path: str, url: str) -> None:
    if url.startswith("s3://"):
        cli = _s3_client()
        if cli is None:
            raise RuntimeError("S3 not configured (missing S3_BUCKET/AWS creds)")
        rest = url[5:]
        bucket, key = rest.split("/", 1) if "/" in rest else (rest, "")
        cli.download_file(bucket, key, path)
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _upload_s3(path: str, key_prefix: Optional[str], content_type: Optional[str]) -> str:
    cli = _s3_client()
    if cli is None:
        return path
    key_prefix = (key_prefix or "").strip("/")
    key = f"{key_prefix}/{os.path.basename(path)}" if key_prefix else os.path.basename(path)
    extra = {"ContentType": content_type} if content_type else None
    cli.upload_file(path, _S3_BUCKET, key, ExtraArgs=extra or {})
    return f"s3://{_S3_BUCKET}/{key}"

def _ffmpeg_path() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")

def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\n---\n{}\n---".format(proc.returncode, " ".join(cmd), proc.stdout)
        )

def _sorted_by_name(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda p: os.path.basename(p).lower())

def render_from_files(session_id: str,
                      files: List[str],
                      output_prefix: str = "editdna/outputs") -> Dict[str, Any]:
    if not files or not isinstance(files, list):
        return {"ok": False, "session_id": session_id, "error": "No input files"}

    ffmpeg = _ffmpeg_path()
    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    clips_dir = os.path.join(workdir, "clips")
    interm_dir = os.path.join(workdir, "interm")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(interm_dir, exist_ok=True)

    try:
        local_paths: List[str] = []
        for u in files:
            name = _basename_from_url(u)
            dst = os.path.join(clips_dir, name)
            _download_to(dst, u)
            local_paths.append(dst)

        ordered = _sorted_by_name(local_paths)

        interm_paths: List[str] = []
        for idx, src in enumerate(ordered):
            outi = os.path.join(interm_dir, f"part_{idx:04d}.mp4")
            vf = (
                "scale=w=720:h=1280:force_original_aspect_ratio=decrease,"
                "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
                "format=yuv420p,setsar=1"
            )
            cmd = [
                ffmpeg, "-y",
                "-nostdin", "-hide_banner", "-loglevel", "info",
                "-analyzeduration", "100M", "-probesize", "100M",
                "-i", src,
                "-vf", vf,
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-max_muxing_queue_size", "9999",
                "-threads", "1",
                outi,
            ]
            _run(cmd)
            interm_paths.append(outi)

        concat_txt = os.path.join(workdir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in interm_paths:
                esc = p.replace("'", "'\\''")
                f.write(f"file '{esc}'\n")

        out_path = os.path.join(workdir, f"{session_id}.mp4")
        cmd2 = [
            ffmpeg, "-y",
            "-nostdin", "-hide_banner", "-loglevel", "info",
            "-safe", "0",
            "-f", "concat", "-i", concat_txt,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        _run(cmd2)

        uri = _upload_s3(out_path, key_prefix=f"{output_prefix.strip('/')}/{session_id}",
                         content_type="video/mp4")

        return {"ok": True, "session_id": session_id, "inputs": len(files), "output": uri}
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
