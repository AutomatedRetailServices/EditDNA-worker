 # app.py — FastAPI entry for EditDNA worker (RunPod)
import os
import time
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import subprocess

app = FastAPI(title="editdna")

# --------------------------------------------------------
# HEALTH ENDPOINTS
# --------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "version": "1.3.1", "time": int(time.time())}

@app.get("/health")
def health():
    return {"ok": True, "service": "editdna", "status": "ready"}

# --------------------------------------------------------
# PROCESS ENDPOINT
# --------------------------------------------------------
class ProcessPayload(BaseModel):
    input_url: Optional[str] = None
    files: Optional[List[str]] = None
    mode: str = "best"
    portrait: bool = True
    max_duration: int = 20
    output_prefix: Optional[str] = "editdna/test"

@app.post("/process")
def process_video(payload: ProcessPayload):
    src = payload.input_url or (payload.files[0] if payload.files else None)
    if not src:
        raise HTTPException(status_code=400, detail="No input_url or files provided")

    # define tmp and output paths
    tmp_in = f"/root/tmp/src_{os.path.basename(src)}"
    os.makedirs("/root/tmp", exist_ok=True)
    os.makedirs("/root/proxies", exist_ok=True)

    # download input
    subprocess.run(["curl", "-L", "-o", tmp_in, src], check=True)

    # build output path
    out_path = f"/root/proxies/proxy_{int(time.time())}.mp4"

    # use libx264 (works everywhere)
    cmd = [
        "ffmpeg", "-y", "-ss", "0", "-i", tmp_in,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        "-t", str(payload.max_duration),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(1080-iw)/2:(1920-ih)/2,fps=24",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-ar", "48000", "-ac", "1",
        "-shortest", "-movflags", "+faststart", out_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {e}")

    return {
        "ok": True,
        "input_url": src,
        "output_path": out_path,
        "output_url_hint": None
    }
