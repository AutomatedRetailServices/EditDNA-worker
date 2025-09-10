import os, json, uuid, time, math, tempfile, subprocess
from pathlib import Path
from typing import List, Optional, Any, Dict
from urllib.parse import urlparse

import boto3
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse

# ----------------------------
# Config & helpers
# ----------------------------
APP_NAME = "EditDNA Worker"
ROOT = Path("/tmp/s2c_sessions")
ROOT.mkdir(parents=True, exist_ok=True)

AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE")  # optional, e.g. https://cdn.yourdomain.com

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET env var is required")

s3 = boto3.client("s3", region_name=AWS_REGION)

def sid_dir(session_id: str) -> Path:
    d = ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())

def short_id() -> str:
    return uuid.uuid4().hex[:8]

def parse_s3_key_from_url(url: str) -> str:
    """
    Accepts AWS-style URL:
      https://<bucket>.s3.<region>.amazonaws.com/<key>
    Returns key.
    """
    parsed = urlparse(url)
    # path begins with '/', remove it
    key = parsed.path.lstrip("/")
    return key

def ffmpeg(cmd: List[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{proc.stderr.decode('utf-8', errors='ignore')}")
    return proc

# ----------------------------
# Schemas
# ----------------------------
class ProcessUrlsBody(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class AutoManifestBody(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720                    # target height, keep aspect
    max_total_sec: int = 12
    max_segments_per_file: int = 1      # default 1 segment per file for draft

class StitchBody(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: Dict[str, Any]

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": time.time()}

@app.post("/process_urls")
def process_urls(body: ProcessUrlsBody):
    """
    Register S3 file URLs (client uploaded directly to S3).
    Creates session.json with file list & metadata.
    """
    if not body.urls:
        raise HTTPException(status_code=422, detail="No URLs provided")

    session_id = uuid.uuid4().hex
    sd = sid_dir(session_id)

    files = []
    for u in body.urls:
        files.append({"file_id": short_id(), "url": u})

    session = {
        "session_id": session_id,
        "created_at": time.time(),
        "tone": body.tone,
        "product_link": body.product_link,
        "features_csv": body.features_csv,
        "files": files,
    }
    save_json(sd / "session.json", session)

    return {"ok": True, "session_id": session_id, "files": files}

@app.post("/automanifest")
def automanifest(body: AutoManifestBody):
    """
    Build a quick draft manifest:
      - Take N files from the session
      - 1 short segment per file (or up to max_segments_per_file later)
      - Clip length is max_total_sec / number_of_files (clamped 2..6 sec)
    """
    sd = sid_dir(body.session_id)
    session = load_json(sd / "session.json")
    files = session.get("files", [])
    if not files:
        raise HTTPException(status_code=400, detail="No files in session")

    # how many seconds per segment
    n = len(files)
    if n <= 0:
        raise HTTPException(status_code=400, detail="No input files")

    per = max(2.0, min(6.0, body.max_total_sec / float(n)))
    total = 0.0
    segments = []

    for f in files:
        if total + per > body.max_total_sec + 0.25:
            break
        segments.append({
            "file_id": f["file_id"],
            "start": 0.0,
            "end": round(per, 3)
        })
        total += per

    manifest = {
        "segments": segments,
        "fps": body.fps,
        "scale": body.scale
    }
    # Save last manifest draft for inspection
    save_json(sd / "draft_manifest.json", manifest)

    return {"ok": True, "session_id": body.session_id, "filename": body.filename, "manifest": manifest}

@app.post("/stitch")
def stitch_video(body: StitchBody):
    """
    Download needed sources from S3, clip each segment to temp mp4, then concat.
    Output saved to /tmp/s2c_sessions/{session}/finals/{filename}
    """
    sd = sid_dir(body.session_id)
    session = load_json(sd / "session.json")
    files = session.get("files", [])

    # Build lookup: file_id -> url
    by_id = {f["file_id"]: f["url"] for f in files}

    manifest = body.manifest
    segments = manifest.get("segments", [])
    if not segments:
        raise HTTPException(status_code=422, detail="Manifest has no segments")

    fps = int(manifest.get("fps", 30))
    scale_h = int(manifest.get("scale", 720))

    work = sd / "work"
    outd = sd / "finals"
    work.mkdir(parents=True, exist_ok=True)
    outd.mkdir(parents=True, exist_ok=True)

    temp_clips = []

    # Download sources (only once per unique file_id)
    local_cache: Dict[str, Path] = {}
    for seg in segments:
        fid = seg["file_id"]
        if fid in local_cache:
            continue
        url = by_id.get(fid)
        if not url:
            raise HTTPException(status_code=400, detail=f"Unknown file_id {fid}")

        # Try S3 SDK (faster in-region), else fall back to HTTP GET
        local_path = work / f"src_{fid}.mov"
        try:
            key = parse_s3_key_from_url(url)
            # If the URL’s bucket equals our S3_BUCKET, use SDK; if not, use HTTP
            host_bucket = urlparse(url).netloc.split(".")[0]  # <bucket>.s3...
            if host_bucket == S3_BUCKET:
                s3.download_file(S3_BUCKET, key, str(local_path))
            else:
                # fall back to HTTP
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
        except Exception:
            # HTTP fallback if SDK path fails for any reason
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)

        local_cache[fid] = local_path

    # Make per-segment trimmed clips (re-encode to unify settings)
    for idx, seg in enumerate(segments):
        fid = seg["file_id"]
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        dur = max(0.05, end - start)

        src = local_cache[fid]
        clip = work / f"clip_{idx:03d}.mp4"

        # scale: keep aspect, target height = scale_h
        vf = f"scale=-2:{scale_h},fps={fps},format=yuv420p"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start}",
            "-i", str(src),
            "-t", f"{dur}",
            "-vf", vf,
            "-r", f"{fps}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip),
        ]
        ffmpeg(cmd)
        temp_clips.append(clip)

    # Concat them
    list_file = work / "list.txt"
    list_file.write_text("".join([f"file '{p.as_posix()}'\n" for p in temp_clips]))

    out_path = outd / body.filename
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-r", f"{fps}",
        "-movflags", "+faststart",
        str(out_path),
    ]
    ffmpeg(cmd_concat)

    # Prepare a public URL if caller configured S3_PUBLIC_BASE (optional future use)
    resp = {
        "ok": True,
        "download_path": f"/download/{body.session_id}/{body.filename}",
        "session_id": body.session_id,
        "filename": body.filename,
    }
    if S3_PUBLIC_BASE:
        resp["public_url"] = f"{S3_PUBLIC_BASE}/sessions/{body.session_id}/finals/{body.filename}"
    return resp

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sid_dir(session_id)
    file_path = sd / "finals" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename, media_type="video/mp4")
