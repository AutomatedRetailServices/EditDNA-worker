# app.py — RunPod worker: proxy video + upload to S3 + presigned URL
from __future__ import annotations
import os, tempfile, subprocess, uuid, shutil, pathlib
from typing import Optional
import boto3, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------ Config (env) ------------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"
S3_BUCKET = os.getenv("S3_BUCKET")  # REQUIRED
S3_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs")
S3_ACL = os.getenv("S3_ACL", "private")  # "private" or "public-read"
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "3600"))  # seconds

# local working dirs
TMP_DIR = "/root/tmp"
OUT_DIR = "/root/proxies"
pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ------------ FastAPI ------------
app = FastAPI(title="editdna", version="1.3.1")

class ProcessIn(BaseModel):
    input_url: str = Field(..., description="HTTP/S URL of a video file")
    mode: str = Field(default="best")
    portrait: bool = Field(default=True)
    max_duration: int = Field(default=15, ge=1, le=600)
    output_prefix: Optional[str] = Field(default=None, description="Optional S3 prefix override")

def _download_to_tmp(url: str) -> str:
    # stream download with requests (no curl dependency)
    fn = f"src_{uuid.uuid4().hex}.mp4"
    dst = os.path.join(TMP_DIR, fn)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst

def _probe_ok(path: str) -> bool:
    try:
        p = subprocess.run(
            [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        return p.returncode == 0 and os.path.getsize(path) > 0
    except Exception:
        return False

def _make_proxy(src: str, portrait: bool, max_seconds: int) -> tuple[str, str]:
    """Return (path, encoder_used)."""
    # scale+pad to 1080x1920 if portrait, else 1920x1080
    if portrait:
        vf = 'scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2,fps=24'
    else:
        vf = 'scale=1920:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1920:1080:(1920-iw)/2:(1080-ih)/2,fps=24'

    # Try libx264 first (present in this image). If it ever fails, fall back to native h264.
    out_path = os.path.join(OUT_DIR, f"proxy_{uuid.uuid4().hex}.mp4")
    base_cmd = [
        FFMPEG, "-y",
        "-ss", "0",
        "-i", src,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        "-t", str(int(max_seconds)),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-ar", "48000", "-ac", "1",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ]

    p = subprocess.run(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, "libx264"

    # fallback encoder
    fallback_cmd = base_cmd.copy()
    i = fallback_cmd.index("-c:v")
    fallback_cmd[i+1] = "h264"
    p2 = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"FFmpeg failed\n\nSTDERR:\n{p2.stderr}")
    return out_path, "h264"

def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def _s3_key_for(output_prefix: Optional[str], filename: str) -> str:
    prefix = (output_prefix or S3_PREFIX).strip("/")
    return f"{prefix}/{filename}"

def _upload_and_presign(local_path: str, key: str) -> tuple[str, Optional[str]]:
    s3 = _s3_client()
    extra = {"ContentType": "video/mp4"}
    if S3_ACL:
        extra["ACL"] = S3_ACL
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs=extra)
    s3_url = f"s3://{S3_BUCKET}/{key}"

    url = None
    if S3_ACL == "private":
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=PRESIGN_EXPIRES,
        )
    else:
        # public-read -> no presign necessary (but still return https form)
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return s3_url, url

@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "version": "1.3.1", "time": int(__import__("time").time())}

@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": FFMPEG, "ffprobe": FFPROBE, "s3_bucket": S3_BUCKET or "(unset)"}

@app.post("/process")
def process_video(p: ProcessIn):
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET not configured on worker")

    tmp_src = out_path = None
    try:
        tmp_src = _download_to_tmp(p.input_url)
        if not _probe_ok(tmp_src):
            raise HTTPException(status_code=400, detail="Downloaded file is not a valid video")

        out_path, encoder = _make_proxy(tmp_src, p.portrait, p.max_duration)

        # upload to S3
        fname = os.path.basename(out_path)
        key = _s3_key_for(p.output_prefix, fname)
        s3_url, http_url = _upload_and_presign(out_path, key)

        return {
            "ok": True,
            "session_id": "session",
            "mode": p.mode,
            "inputs": [p.input_url],
            "encoder": encoder,
            "output_path": out_path,
            "s3_bucket": S3_BUCKET,
            "s3_key": key,
            "s3_url": s3_url,
            "url": http_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        for f in (tmp_src,):
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
