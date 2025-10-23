# app.py — RunPod worker: process -> render proxy -> upload to S3
import os, re, json, time, shutil, uuid, subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ---- optional: use your s3_utils if available ----
S3_UTILS = None
try:
    import s3_utils  # must provide: upload_file(local_path, key, bucket=..., region=..., content_type=..., acl=...)
    S3_UTILS = s3_utils
except Exception:
    S3_UTILS = None

try:
    import boto3  # fallback uploader
except Exception:
    boto3 = None

# --------- env / paths ----------
FFMPEG_BIN  = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

OUT_DIR = Path(os.getenv("OUT_DIR", "/root/outputs")).resolve()
TMP_DIR = Path(os.getenv("TMP_DIR", "/root/tmp")).resolve()
PROXY_DIR = Path(os.getenv("PROXY_DIR", "/root/proxies")).resolve()

# S3 config
S3_BUCKET   = os.getenv("S3_BUCKET")  # REQUIRED if you want upload
AWS_REGION  = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
S3_PREFIX   = os.getenv("S3_PREFIX", "editdna/outputs")
PRESIGN_SEC = int(os.getenv("PRESIGN_EXPIRES", "3600"))
S3_ACL      = os.getenv("S3_ACL", "private")  # "private" or "public-read"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
PROXY_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="editdna", version="1.3.1")

# ---------- Models ----------
class ProcessIn(BaseModel):
    input_url: str
    mode: str = Field(default="best")
    portrait: bool = True
    max_duration: int = 15  # seconds

# ---------- Helpers ----------
def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout

def ffmpeg_has(name: str) -> bool:
    try:
        out = run([FFMPEG_BIN, "-hide_banner", "-encoders"])
        return (re.search(rf"\b{name}\b", out) is not None)
    except Exception:
        return False

def pick_video_encoder() -> tuple[list[str], str]:
    """
    Returns (video_args, encoder_name) tuned for broad compatibility.
    """
    if ffmpeg_has("libx264"):
        # software x264 — always available on our image, highest compatibility
        return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"], "libx264")
    if ffmpeg_has("libopenh264"):
        # fallback if x264 somehow missing
        return (["-c:v", "libopenh264", "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "5000k"], "libopenh264")
    # worst-case generic
    return (["-c:v", "h264"], "h264")

def _download_to_tmp(url: str) -> Path:
    # use ffmpeg to download and normalize container quickly
    stem = Path(url.split("/")[-1]).stem or f"src_{uuid.uuid4().hex}"
    tmp_path = TMP_DIR / f"src_{uuid.uuid4().hex}_{stem}.mp4"
    # copy/stream into mp4 (no re-encode) if possible
    run([FFMPEG_BIN, "-y", "-i", url, "-c", "copy", "-movflags", "+faststart", str(tmp_path)])
    return tmp_path

def _make_proxy(src: Path, portrait: bool, tmax: int) -> Path:
    enc_args, enc_name = pick_video_encoder()
    proxy = PROXY_DIR / f"proxy_{uuid.uuid4().hex}.mp4"
    # letterbox to 1080x1920 if portrait else 1920x1080
    if portrait:
        scale_pad = 'scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2'
        out_w, out_h = 1080, 1920
    else:
        scale_pad = 'scale=1920:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1920:1080:(1920-iw)/2:(1080-ih)/2'
        out_w, out_h = 1920, 1080

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", "0", "-i", str(src),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        "-t", str(int(tmax)),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"{scale_pad},fps=24",
        *enc_args,
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(proxy),
    ]
    log = run(cmd)
    print(f"[ffmpeg] using {enc_name}\n{log[:4000]}")
    return proxy

def _upload_s3(local: Path) -> dict:
    """
    Returns dict with bucket/key/s3_url/https_url/presigned_url
    """
    if not S3_BUCKET:
        return {}

    key = f"{S3_PREFIX.strip('/')}/{int(time.time())}_{local.name}"
    if S3_UTILS is not None:
        info = S3_UTILS.upload_file(
            str(local), key,
            bucket=S3_BUCKET, region=AWS_REGION,
            content_type="video/mp4", acl=S3_ACL
        )
        # presign if private
        if S3_ACL != "public-read":
            try:
                presigned = S3_UTILS.presigned_url(key, bucket=S3_BUCKET, region=AWS_REGION, expires=PRESIGN_SEC)
            except Exception:
                presigned = None
        else:
            presigned = info.get("https_url")
        return {
            **info,
            "presigned_url": presigned
        }

    # Fallback inline if s3_utils not importable
    if boto3 is None:
        return {}
    s3 = boto3.client("s3", region_name=AWS_REGION)
    extra = {"ContentType": "video/mp4"}
    if S3_ACL:
        extra["ACL"] = S3_ACL
    with open(local, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, key, ExtraArgs=extra)
    https_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    presigned = https_url
    if S3_ACL != "public-read":
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=PRESIGN_SEC
        )
    return {
        "bucket": S3_BUCKET,
        "key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
        "presigned_url": presigned,
        "region": AWS_REGION,
    }

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "version": "1.3.1", "time": int(time.time())}

@app.post("/process")
def process(req: ProcessIn):
    try:
        src = _download_to_tmp(req.input_url)
        proxy = _make_proxy(src, req.portrait, req.max_duration)

        # here you’d normally do the semantic pipeline; for this demo we just publish the proxy
        out_name = f"outputs_{uuid.uuid4().hex}.mp4"
        final_path = OUT_DIR / out_name
        shutil.copy2(proxy, final_path)

        s3_info = _upload_s3(final_path)
        return {
            "ok": True,
            "session_id": "session",
            "mode": req.mode,
            "inputs": [req.input_url],
            "output_path": str(final_path),
            "s3": s3_info or None,
            "output_url_hint": (s3_info.get("presigned_url") if s3_info else None),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
