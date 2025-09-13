import os, uuid, json, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
import redis
from rq import Queue
from rq.job import Job

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ================== Config & Globals ==================
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-rq-stitch"

# Where we keep transient session data/files on disk
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis/RQ (IMPORTANT: binary connection, NO decode_responses)
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required")
redis_conn = redis.from_url(REDIS_URL)        # binary
q = Queue("default", connection=redis_conn)

# Optional: public S3 base to return nice links (bucket must be public-read)
# Example: https://script2clipshop-video-automatedretailservices.s3.us-east-1.amazonaws.com
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

# ================== Helpers ==================
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

def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

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
    try:
        lst.unlink()
    except Exception:
        pass

def _public_or_download(session_id: str, fname: str) -> Dict[str, Any]:
    if S3_PUBLIC_BASE:
        # You are serving purely local files; public S3 base is only for when
        # you upload results to S3. If later you add S3 upload here, return that URL.
        return {"public_url_hint": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}"}
    return {"download_path": f"/download/{session_id}/{fname}"}

# ================== Health ==================
@app.get("/health")
def health():
    try:
        # light ping to ensure connection object is usable
        redis_conn.ping()
        r_ok = True
    except Exception:
        r_ok = False
    return {"ok": True, "service": "editdna-worker", "version": VERSION, "redis": r_ok}

# ================== Models ==================
class ProcessURLsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = "casual"
    product_link: Optional[str] = ""
    features_csv: Optional[str] = ""

class AutoManifestIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    max_total_sec: float = 12.0
    max_segments_per_file: int = 1

class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict  # {"segments":[{file_id,start,end},...], "fps":30, "scale":720}

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict
    fps: int = 30
    scale: int = 720

# ================== Core endpoints ==================
@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    """
    Registers external video URLs for a new session.
    """
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    url_map = {}
    for u in body.urls:
        fid = uuid.uuid4().hex[:8]
        files_meta.append({"file_id": fid, "source": "url"})
        url_map[fid] = u

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "urls": url_map,               # file_id -> URL (public or presigned)
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
        "created_at": int(time.time())
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}

@app.post("/automanifest")
def automanifest(body: AutoManifestIn):
    """
    Creates a simple manifest: short bite from each file until max_total_sec.
    """
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
    out = {"ok": True, "session_id": body.session_id, "filename": body.filename, "manifest": manifest}
    save_json(sd / "manifest.json", out)
    return out

def _stitch_do(*, urls: Dict[str, str], session_id: str, manifest: dict, filename: str) -> Dict[str, Any]:
    """
    Worker-friendly inner function that downloads, trims and concatenates parts.
    Returns a dict with either public_url_hint or local download path.
    """
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

    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "final.mp4"
    final_path = sd / safe_name
    _concat_mp4s(parts, final_path)

    # (If later you upload to S3, swap this to return that real URL.)
    return {"ok": True, **_public_or_download(session_id, safe_name)}

@app.post("/stitch")
def stitch(body: StitchIn):
    """
    Synchronous stitch — good for short drafts.
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    return _stitch_do(urls=urls, session_id=body.session_id, manifest=body.manifest, filename=body.filename)

# ---------- Worker function (must be module-level so RQ can import it) ----------
def stitch_core_from_session(session_id: str, filename: str, manifest: dict) -> Dict[str, Any]:
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise RuntimeError("no URLs in session")
    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    """
    Enqueue a background stitch job via RQ.
    """
    # store a bit of context in job.meta for nicer /jobs/{id}
    meta = {
        "type": "stitch",
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": body.manifest,
        "created_at": int(time.time()),
    }
    job = q.enqueue(stitch_core_from_session, body.session_id, body.filename, body.manifest, meta=meta)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/test")
def jobs_test():
    """
    Tiny test job to verify the worker loop.
    """
    job = q.enqueue(lambda x, y: x + y, 2, 3)
    return {"job_id": job.id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """
    Lookup RQ job status/result using the SAME binary Redis connection.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as e:
        # If it isn't in Redis yet (or wrong ID), report queued.
        return {"job_id": job_id, "status": "queued", "result": None, "note": f"lookup_failed: {e}"}

    status = job.get_status()
    meta = job.meta or {}
    payload = {
        "job_id": job.id,
        "type": meta.get("type"),
        "session_id": meta.get("session_id"),
        "filename": meta.get("filename"),
        "manifest": meta.get("manifest"),
        "created_at": meta.get("created_at"),
        "status": status,
        "result": job.result if status == "finished" else None,
    }
    # surface exceptions if any
    if status == "failed" and job.exc_info:
        payload["error"] = job.exc_info
    return payload

# ================== Local Download (if not using S3 public link) ==================
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
