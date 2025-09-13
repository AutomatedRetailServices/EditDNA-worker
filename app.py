# app.py  — EditDNA Web API (with /analyze + /classify placeholders)

import os, uuid, json, shutil, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue, Connection
from rq.job import Job

# ==============================
# App setup & ENV
# ==============================
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-mvp-stitch+placeholders"

# Sessions live on ephemeral disk in Render; fine for MVP
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis / RQ
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
# IMPORTANT: binary-safe (avoid UTF-8 decode issues you saw earlier)
rconn = redis.from_url(REDIS_URL, decode_responses=False)

# Optional public CDN base (for nicer links)
# e.g. https://script2clipshop-video-automatedretailservices.s3.us-east-1.amazonaws.com
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

# ==============================
# Helpers
# ==============================
def sess_dir(session_id: str) -> Path:
    p = SESS_ROOT / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())

def run(cmd: list):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def _public_or_download(session_id: str, fname: str) -> dict:
    if S3_PUBLIC_BASE:
        # If later you push the file to S3, return that URL instead.
        # For now we still serve from /download, but we include where it *would* live.
        return {
            "public_base_hint": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}",
            "download_path": f"/download/{session_id}/{fname}",
        }
    return {"download_path": f"/download/{session_id}/{fname}"}

def _download_to_tmp(url: str, dst: Path):
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        with dst.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _ffmpeg_trim(input_path: Path, start: float, end: float, out_path: Path, scale: int, fps: int):
    dur = max(0.1, float(end) - float(start))
    run([
        "ffmpeg", "-y",
        "-ss", f"{float(start):.3f}",
        "-t", f"{dur:.3f}",
        "-i", str(input_path),
        "-vf", f"scale={int(scale)}:-2:flags=lanczos",
        "-r", str(int(fps)),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path)
    ])

def _concat_mp4s(parts: List[Path], out_path: Path):
    lst = out_path.with_suffix(".txt")
    lst.write_text("\n".join([f"file '{p.as_posix()}'" for p in parts]))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)])
    lst.unlink(missing_ok=True)

def _safe_name(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or "final.mp4"

# ==============================
# Health
# ==============================
@app.get("/health")
def health():
    try:
        # ping once (non-blocking)
        rconn.ping()
        ok_redis = True
    except Exception:
        ok_redis = False
    return {"ok": True, "service": "editdna-worker", "version": VERSION, "redis": ok_redis}

# ==============================
# 1) Register input URLs
# ==============================
class ProcessURLsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = "casual"
    product_link: Optional[str] = ""
    features_csv: Optional[str] = ""

@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    url_map: Dict[str, str] = {}
    for u in body.urls:
        fid = uuid.uuid4().hex[:8]
        files_meta.append({"file_id": fid, "source": "url"})
        url_map[fid] = u

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "urls": url_map,  # file_id -> URL (public/presigned)
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
        "created_at": int(time.time()),
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}

# ==============================
# 2) Auto-manifest (simple heuristic)
# ==============================
class AutoManifestIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    max_total_sec: float = 12.0
    max_segments_per_file: int = 1

@app.post("/automanifest")
def automanifest(body: AutoManifestIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    files = session.get("files") or []
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")

    segments = []
    total = 0.0
    per_seg = max(2.0, min(8.0, body.max_total_sec / max(1, len(files))))
    for f in files:
        if total >= body.max_total_sec:
            break
        take = min(per_seg, body.max_total_sec - total)
        segments.append({"file_id": f["file_id"], "start": 0.0, "end": round(take, 3)})
        total += take

    manifest = {"segments": segments, "fps": body.fps, "scale": body.scale}
    out = {"ok": True, "session_id": body.session_id, "filename": _safe_name(body.filename), "manifest": manifest}
    save_json(sd / "manifest.json", out)
    return out

# ==============================
# 3) Stitch core (sync helper)
# ==============================
def _stitch_do(*, session_id: str, urls: Dict[str, str], manifest: dict, filename: str) -> dict:
    sd = sess_dir(session_id)
    work = sd / "work"
    work.mkdir(parents=True, exist_ok=True)

    parts: List[Path] = []
    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 720))

    for idx, seg in enumerate(manifest.get("segments", [])):
        fid = seg["file_id"]
        src_url = urls.get(fid)
        if not src_url:
            raise RuntimeError(f"missing url for file_id {fid}")
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)
        part_out = work / f"part_{idx:03d}.mp4"
        _ffmpeg_trim(cache_vid, seg.get("start", 0.0), seg.get("end", 3.0), part_out, scale=scale, fps=fps)
        parts.append(part_out)

    final_name = _safe_name(filename)
    final_path = sd / final_name
    _concat_mp4s(parts, final_path)

    # In a later step, upload to S3 and return the actual S3 URL here.
    return {"ok": True, **_public_or_download(session_id, final_name)}

# RQ worker job target
def stitch_core_from_session(session_id: str, filename: str, manifest: dict) -> dict:
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise RuntimeError("no URLs registered for this session")
    return _stitch_do(session_id=session_id, urls=urls, manifest=manifest, filename=filename)

# ==============================
# 4) Async enqueue
# ==============================
class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    with Connection(rconn):
        q = Queue("default")
        job = q.enqueue(stitch_core_from_session, body.session_id, _safe_name(body.filename), body.manifest)
    # Also persist a lightweight mirror for convenience (optional).
    # Not strictly required because /jobs reads straight from RQ by id.
    return {"ok": True, "job_id": job.get_id()}

# ==============================
# 5) Job status
# ==============================
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        with Connection(rconn):
            job = Job.fetch(job_id, connection=rconn)
    except Exception as e:
        # common when asking too fast: "No such job"
        return {"ok": False, "error": f"rq_lookup_failed: {str(e)}"}

    if job.is_queued:
        return {"job_id": job_id, "status": "queued", "result": None}
    if job.is_started:
        return {"job_id": job_id, "status": "started", "result": None}
    if job.is_finished:
        return {"job_id": job_id, "status": "finished", "result": job.result}
    if job.is_failed:
        err = None
        try:
            err = job.exc_info
        except Exception:
            err = "failed"
        return {"job_id": job_id, "status": "failed", "result": None, "error": err}

    return {"job_id": job_id, "status": "unknown", "result": None}

# ==============================
# 6) Download (local)
# ==============================
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")

# ==============================
# NEW: 7) Analyze (placeholder)
# ==============================
class AnalyzeIn(BaseModel):
    session_id: str

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    """
    Placeholder analysis:
    - lists the registered file_ids + source URLs
    - picks the 'best_file_id' as the first one
    - returns a fake per-file 'score'
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls: Dict[str, str] = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    file_ids = list(urls.keys())
    best_file_id = file_ids[0]

    analysis = {
        "session_id": body.session_id,
        "best_file_id": best_file_id,
        "files": [
            {"file_id": fid, "url": urls[fid], "score": 0.5 if i else 0.8}
            for i, fid in enumerate(file_ids)
        ],
        "notes": "Placeholder analysis. Replace with real model later."
    }

    save_json(sd / "analysis.json", analysis)
    return {"ok": True, **analysis}

# ==============================
# NEW: 8) Classify (placeholder)
# ==============================
class ClassifyIn(BaseModel):
    session_id: str

@app.post("/classify")
def classify(body: ClassifyIn):
    """
    Placeholder classification:
    - returns static tags + a trivial heuristic on filename
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls: Dict[str, str] = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    tags = ["talking_head", "product_demo"]
    # silly hint: if any URL contains 'IMG' assume phone_camera
    if any("IMG" in u or "img" in u for u in urls.values()):
        tags.append("phone_camera")

    out = {"ok": True, "session_id": body.session_id, "tags": tags, "notes": "Placeholder classification."}
    save_json(sd / "classification.json", out)
    return out
