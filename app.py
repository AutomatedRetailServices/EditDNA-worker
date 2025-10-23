# app.py — EditDNA worker API with robust ffmpeg encoder selection
from __future__ import annotations

import os
import json
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

APP_VERSION = "1.3.2"

# ---------- Paths ----------
HOME = Path.home()
OUT_DIR = HOME / "outputs"
PROXY_DIR = HOME / "proxies"
TMP_DIR = HOME / "tmp"
for d in (OUT_DIR, PROXY_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- FFmpeg paths ----------
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

def _run_subproc(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print("[$]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check, text=True)

def _has_encoder(name: str) -> bool:
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False).stdout
        return name in out
    except Exception:
        return False

def _pick_h264_encoder() -> Dict[str, List[str]]:
    """
    Decide best H.264 encoder and its safe flags for this ffmpeg build.
    Returns dict with 'vcodec' and 'vflags'.
    """
    # Prefer libx264 if present
    if _has_encoder("libx264"):
        print("[ffmpeg] using libx264", flush=True)
        return {
            "vcodec": ["-c:v", "libx264"],
            "vflags": ["-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
        }
    # Fallback to the builtin 'h264' encoder (no -preset support)
    if _has_encoder(" h264 "):  # note spaces in listing lines
        print("[ffmpeg] libx264 not found; falling back to builtin h264 encoder", flush=True)
        return {
            "vcodec": ["-c:v", "h264"],
            "vflags": ["-global_header", "-pix_fmt", "yuv420p"]
        }
    # Last resort: try mpeg4 so the pipeline still runs
    print("[ffmpeg] no H.264 encoder available; falling back to mpeg4 (compat)", flush=True)
    return {
        "vcodec": ["-c:v", "mpeg4"],
        "vflags": ["-qscale:v", "5", "-pix_fmt", "yuv420p"]
    }

H264 = _pick_h264_encoder()

print(f"[boot] FFMPEG={FFMPEG}", flush=True)
print(f"[boot] FFPROBE={FFPROBE}", flush=True)

# ---------- Models ----------
class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True
    mode: Optional[str] = "concat"  # "best"|"first"|"concat"
    max_duration: Optional[int] = None
    take_top_k: Optional[int] = None
    min_clip_seconds: Optional[float] = None
    max_clip_seconds: Optional[float] = None
    drop_silent: Optional[bool] = True
    drop_black: Optional[bool] = True
    with_captions: Optional[bool] = False

class ProcessRequest(BaseModel):
    session_id: Optional[str] = "session"
    mode: Optional[str] = "best"
    input_url: Optional[str] = None
    files: Optional[List[str | HttpUrl]] = None
    portrait: Optional[bool] = True
    max_duration: Optional[int] = 60
    output_prefix: Optional[str] = "editdna/outputs"

# ---------- Helpers ----------
def _safe_name(url_or_path: str) -> str:
    parsed = urlparse(url_or_path)
    base = os.path.basename(parsed.path) or "input"
    base = base.replace(" ", "_")
    return base

def _run_or_raise(cmd: List[str]) -> None:
    proc = _run_subproc(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\n{proc.stdout}")

def _http_base() -> Optional[str]:
    host_env = os.getenv("PUBLIC_BASE")
    if host_env:
        return host_env.rstrip("/")
    return None

# ---------- Proxy pipeline ----------
def build_proxy(input_url: str, portrait: bool = True, max_seconds: Optional[int] = None) -> Path:
    src_name = _safe_name(input_url)
    local_src = TMP_DIR / f"src_{uuid.uuid4().hex}_{src_name}"
    if str(input_url).startswith("http"):
        urlretrieve(input_url, local_src)
    else:
        _run_or_raise(["/bin/sh", "-lc", f"cp {shlex.quote(str(input_url))} {shlex.quote(str(local_src))}"])

    if portrait:
        scale_filter = 'scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2'
    else:
        scale_filter = 'scale=960:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=960:540:(960-iw)/2:(540-ih)/2'

    limit = ["-t", str(int(max_seconds))] if max_seconds and max_seconds > 0 else []

    proxy_path = PROXY_DIR / f"proxy_{uuid.uuid4().hex}.mp4"
    cmd = [
        FFMPEG, "-y", "-ss", "0", "-i", str(local_src),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
        *limit,
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"{scale_filter},fps=24",
        # video codec + flags (auto-selected)
        *H264["vcodec"], *H264["vflags"],
        # audio
        "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(proxy_path)
    ]
    _run_or_raise(cmd)
    return proxy_path

def pick_good_takes(proxies: List[Path], mode: str = "best", take_top_k: Optional[int] = 1) -> List[Path]:
    if not proxies:
        return []
    if mode == "concat":
        return proxies
    if mode in ("best", "first"):
        k = take_top_k or 1
        return proxies[:k]
    return proxies[:1]

def render_concat(takes: List[Path], output_prefix: str = "editdna/outputs") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{Path(output_prefix).name}_{uuid.uuid4().hex}.mp4"

    if len(takes) == 1:
        _run_or_raise([FFMPEG, "-y", "-i", str(takes[0]), "-c", "copy", "-movflags", "+faststart", str(out_file)])
        return out_file

    list_txt = TMP_DIR / f"concat_{uuid.uuid4().hex}.txt"
    with list_txt.open("w") as f:
        for p in takes:
            f.write(f"file '{p.as_posix()}'\n")
    _run_or_raise([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", "-movflags", "+faststart", str(out_file)])
    return out_file

# ---------- Routes ----------
app = FastAPI(title="editdna", version=APP_VERSION)

@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "editdna", "version": APP_VERSION, "time": int(datetime.utcnow().timestamp())})

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "version": APP_VERSION,
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "encoder": " ".join(H264["vcodec"])
    })

@app.post("/process")
def process(req: ProcessRequest) -> JSONResponse:
    try:
        session_id = req.session_id or "session"
        files = req.files or ([req.input_url] if req.input_url else [])
        if not files:
            raise HTTPException(status_code=400, detail="Provide input_url or files[]")

        proxies = [build_proxy(str(f), portrait=bool(req.portrait), max_seconds=req.max_duration) for f in files]
        takes = pick_good_takes(proxies, mode=req.mode or "best", take_top_k=1)
        out_path = render_concat(takes, output_prefix=req.output_prefix or "editdna/outputs")
        base = _http_base()

        return JSONResponse({
            "ok": True,
            "session_id": session_id,
            "mode": req.mode or "best",
            "inputs": [str(f) for f in files],
            "output_path": out_path.as_posix(),
            "output_url_hint": (f"{base}/{out_path.relative_to(HOME).as_posix()}" if base else None),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
