# app.py — minimal RunPod worker API with robust FFmpeg wrapper
from __future__ import annotations
import os, tempfile, subprocess, shutil, time, json
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="editdna")

FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

def _check_bin(path: str) -> str:
    if shutil.which(path):
        return shutil.which(path)  # absolute path if it was just 'ffmpeg'
    if os.path.exists(path):
        return path
    raise RuntimeError(f"Binary not found: {path}")

FFMPEG = _check_bin(FFMPEG)
FFPROBE = _check_bin(FFPROBE)

def _has_encoder(name: str) -> bool:
    try:
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True
        )
        return name in p.stdout
    except Exception:
        return False

def _ffmpeg_err_tail(cp: subprocess.CalledProcessError, n: int = 60) -> str:
    err = (cp.stderr or "").splitlines()[-n:]
    return "\n".join(err) or str(cp)

class ProcessPayload(BaseModel):
    input_url: str
    mode: Optional[str] = "best"
    portrait: Optional[bool] = True
    max_duration: Optional[int] = 20
    output_prefix: Optional[str] = "editdna/test"

@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "version": "1.3.1", "time": int(time.time())}

@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": FFMPEG, "ffprobe": FFPROBE}

@app.post("/process")
def process_video(p: ProcessPayload):
    # temp files
    os.makedirs("/root/tmp", exist_ok=True)
    os.makedirs("/root/proxies", exist_ok=True)

    src_name = os.path.basename(p.input_url.split("?")[0]) or "input.mp4"
    tmp_in = os.path.join("/root/tmp", f"src_{src_name}")
    out_path = os.path.join("/root/proxies", f"proxy_{int(time.time())}.mp4")

    # 1) download with curl
    try:
        subprocess.run(["curl", "-L", "-o", tmp_in, p.input_url], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"curl failed: {e.stderr.decode('utf-8','ignore') if e.stderr else str(e)}")

    # 2) pick encoder (libx264 first, then libopenh264)
    enc_name = None
    if _has_encoder("libx264"):
        enc_name = "libx264"
        enc_flags = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", "-g", "48"]
    elif _has_encoder("libopenh264"):
        enc_name = "libopenh264"
        enc_flags = ["-c:v", "libopenh264", "-b:v", "2500k", "-maxrate", "2500k", "-bufsize", "5000k",
                     "-pix_fmt", "yuv420p", "-g", "48"]
    else:
        # last resort: FFmpeg native h264 encoder (very basic, but exists everywhere)
        enc_name = "h264"
        enc_flags = ["-c:v", "h264", "-pix_fmt", "yuv420p", "-g", "48"]

    # 3) build filter
    if p.portrait:
        vf = "scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2,fps=24"
    else:
        vf = "scale=1920:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1920:1080:(1920-iw)/2:(1080-ih)/2,fps=24"

    dur = max(1, min(int(p.max_duration or 20), 300))

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-y", "-ss", "0", "-i", tmp_in,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        "-t", str(dur),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf,
        *enc_flags,
        "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        out_path
    ]

    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed (encoder={enc_name}).\n{_ffmpeg_err_tail(e)}")

    # success
    return {
        "ok": True,
        "session_id": "session",
        "mode": p.mode,
        "inputs": [p.input_url],
        "encoder": enc_name,
        "output_path": out_path,
        "output_url_hint": None
    }
