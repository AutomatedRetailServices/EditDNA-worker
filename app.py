# app.py  — EditDNA Web API (jobs + stitching)
# v1.4.0 — pass URLs in the job payload so worker doesn't need local session

import os, uuid, json, shutil, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue

# ========= App setup =========
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-rq-stitch-inline-urls"

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis / RQ
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL missing")
r_conn: redis.Redis = redis.from_url(REDIS_URL, decode_responses=False)
rq = Queue("default", connection=r_conn)

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
        # You can later upload the file to S3 and return S3 URL here.
        # For now we expose local download endpoint.
        pass
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


# ========= Health =========
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-worker",
        "version": VERSION,
        "redis": True
    }


# ========= S3 URL (no direct upload) flow =========
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
        "urls": url_map,  # file_id -> URL (presigned/public)
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


# ========= Core stitch used by both sync and worker =========
def _stitch_do(*, urls: Dict[str, str], session_id: str, manifest: Dict[str, Any], filename: str) -> Dict[str, Any]:
    sd = sess_dir(session_id)
    work = sd / "work"
    work.mkdir(parents=True, exist_ok=True)

    parts = []
    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 720))
    for idx, seg in enumerate(manifest.get("segments", [])):
        fid = seg["file_id"]
        src_url = urls.get(fid)
        if not src_url:
            raise HTTPException(status_code=400, detail=f"missing url for file_id {fid}")
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)
        part_out = work / f"part_{idx:03d}.mp4"
        _ffmpeg_trim(cache_vid, seg.get("start", 0.0), seg.get("end", 3.0), part_out, scale=scale, fps=fps)
        parts.append(part_out)

    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "final.mp4"
    final_path = sd / safe_name
    _concat_mp4s(parts, final_path)

    return {"ok": True, **_public_or_download(session_id, safe_name)}


# ========= Sync stitch (small jobs) =========
class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict

@app.post("/stitch")
def stitch(body: StitchIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs for session")

    return _stitch_do(urls=urls, session_id=body.session_id, manifest=body.manifest, filename=body.filename)


# ========= Async layer: enqueue + status =========
def stitch_core_from_job(payload: dict) -> Dict[str, Any]:
    """
    Executed inside the RQ worker. It receives everything it needs:
    - session_id (for temp dir)
    - filename
    - manifest
    - urls (file_id -> url)  <— critical so worker doesn't need local session.json
    """
    return _stitch_do(
        urls=payload["urls"],
        session_id=payload["session_id"],
        manifest=payload["manifest"],
        filename=payload["filename"],
    )

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    # load session (on the web container) just to grab URLs
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs for session")

    job_payload = {
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": body.manifest,
        "urls": urls,  # <— embed URLs so worker has everything
    }
    job = rq.enqueue(stitch_core_from_job, job_payload, job_id=uuid.uuid4().hex)
    out = {
        "ok": True,
        "job_id": job.id,
        "type": "stitch",
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": body.manifest,
        "created_at": int(time.time()),
        "status": "queued",
        "rq_id": job.id,
    }
    # also store a small JSON so /jobs can show details even before completion
    r_conn.set(f"jobmeta:{job.id}", json.dumps(out).encode("utf-8"))
    return out

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    # RQ state
    job = rq.fetch_job(job_id)
    base = {}
    meta_raw = r_conn.get(f"jobmeta:{job_id}")
    if meta_raw:
        try:
            base = json.loads(meta_raw.decode("utf-8"))
        except Exception:
            base = {"job_id": job_id}
    else:
        base = {"job_id": job_id}

    if not job:
        # maybe finished long ago; return what we have
        return {**base, "status": base.get("status", "unknown"), "result": base.get("result")}

    if job.is_finished:
        res = job.result
        base.update({"status": "finished", "result": res})
        r_conn.set(f"jobmeta:{job_id}", json.dumps(base).encode("utf-8"))
        return base
    if job.is_failed:
        err = job.exc_info or "failed"
        base.update({"status": "failed", "result": None, "error": err})
        r_conn.set(f"jobmeta:{job_id}", json.dumps(base).encode("utf-8"))
        return base
    if job.get_status() == "started":
        base.update({"status": "started", "result": None})
        return base
    base.update({"status": "queued", "result": None})
    return base


# ========= Local download (fallback) =========
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
