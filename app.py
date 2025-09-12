import os, uuid, json, shutil, subprocess, time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue
from rq.job import Job

# ========= App setup =========
app = FastAPI(title="EditDNA Worker API")
VERSION = "1.2.0-async-stitch"

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis (for async jobs)
REDIS_URL = os.getenv("REDIS_URL", "")
r: Optional[redis.Redis] = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

# Optional public CDN base (for nicer links)
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")


# ========= Helpers =========
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
        return {"public_url": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}"}
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

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


# ========= Health =========
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-worker",
        "version": VERSION,
        "redis": bool(r)
    }


# ========= S3 URL (no direct upload) flow we’re using now =========
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
    url_map = {}
    for u in body.urls:
        fid = uuid.uuid4().hex[:8]
        files_meta.append({"file_id": fid, "source": "url"})
        url_map[fid] = u

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "urls": url_map,  # file_id -> URL (presigned or public)
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}


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

    # Simple heuristic: one short bite from each file until max_total_sec
    segments = []
    total = 0.0
    # target per segment so we don't exceed the cap
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


class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict  # {"segments":[...], "fps":30, "scale":720}

@app.post("/stitch")
def stitch(body: StitchIn):
    """
    Synchronous stitch (works for short drafts). Worker will call this for big jobs.
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    work = sd / "work"
    work.mkdir(parents=True, exist_ok=True)

    parts = []
    fps = int(body.manifest.get("fps", 30))
    scale = int(body.manifest.get("scale", 720))
    for idx, seg in enumerate(body.manifest.get("segments", [])):
        fid = seg["file_id"]
        src_url = urls.get(fid)
        if not src_url:
            raise HTTPException(status_code=400, detail=f"missing url for file_id {fid}")
        # cache original
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)
        # trim/encode consistent part
        part_out = work / f"part_{idx:03d}.mp4"
        _ffmpeg_trim(cache_vid, seg.get("start", 0.0), seg.get("end", 3.0), part_out, scale=scale, fps=fps)
        parts.append(part_out)

    safe_name = "".join(c for c in body.filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "final.mp4"
    final_path = sd / safe_name
    _concat_mp4s(parts, final_path)

    return {"ok": True, **_public_or_download(body.session_id, safe_name)}


# ========= Async layer: enqueue + status =========
class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict
    fps: int = 30
    scale: int = 720

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    if not r:
        raise HTTPException(status_code=500, detail="REDIS_URL not configured")
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "type": "stitch",
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": body.manifest,
        "fps": body.fps,
        "scale": body.scale,
        "created_at": int(time.time()),
        "status": "queued",
    }
    r.set(_job_key(job_id), json.dumps(job))
    r.lpush("jobs", json.dumps(job))
    return {"ok": True, "job_id": job_id}

@app.get("/job_status")
def job_status(job_id: str):
    if not r:
        raise HTTPException(status_code=500, detail="REDIS_URL not configured")
    raw = r.get(_job_key(job_id))
    if not raw:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, **json.loads(raw)}


# ========= Local download (fallback if no S3 public base) =========
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")


# ========= Redis Test Routes (for Step 4) =========
def get_q():
    url = os.environ["REDIS_URL"]
    return Queue("default", connection=redis.from_url(url))

def add(a, b):
    time.sleep(2)
    return a + b

@app.get("/jobs/test")
def jobs_test():
    q = get_q()
    job = q.enqueue(add, 2, 3)
    return {"job_id": job.id}

@app.get("/jobs/{job_id}")
def jobs_status(job_id: str):
    q = get_q()
    try:
        job = Job.fetch(job_id, connection=q.connection)
        return {
            "job_id": job.id,
            "status": job.get_status(),
            "result": job.result
        }
    except Exception as e:
        return {"error": str(e)}
