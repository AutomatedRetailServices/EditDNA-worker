import os, uuid, json, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
import redis
from rq import Queue
from rq.job import Job

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===================== App setup =====================
VERSION = "1.3.3-rq-stitch"
app = FastAPI(title="EditDNA Web API", version=VERSION)

# CORS (set CORS_ORIGINS="https://yourapp.bubbleapps.io,*" if needed)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Working directory for sessions
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis & RQ  (NO decode_responses so we can handle bytes safely)
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not set")
_redis = redis.from_url(REDIS_URL)  # important: leave default (bytes)

def get_q() -> Queue:
    return Queue("default", connection=_redis)

# Optional public CDN base (so responses can return clickable URLs)
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")


# ===================== Helpers =====================
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

def run_ffmpeg(cmd: list) -> str:
    """Run ffmpeg (or any shell cmd) and raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def _download_to_tmp(url: str, dst: Path):
    # Stream download to avoid RAM spikes
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        with dst.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _safe_name(name: str) -> str:
    name = name.strip() or "final.mp4"
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", "."))

def _public_or_download(session_id: str, fname: str) -> Dict[str, str]:
    if S3_PUBLIC_BASE:
        # If you later upload outputs to S3, return that URL instead.
        return {"public_url": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}"}
    return {"download_path": f"/download/{session_id}/{fname}"}


# ===================== Core stitch function =====================
def stitch_core(session_id: str, manifest: Dict[str, Any], filename: str = "final.mp4") -> Dict[str, Any]:
    """
    Downloads/caches sources, trims segments with consistent fps/scale,
    and concatenates into a final mp4.
    """
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")

    session = load_json(meta_path)
    urls: Dict[str, str] = session.get("urls") or {}
    if not urls:
        raise RuntimeError("no URLs registered for this session")

    work = sd / "work"
    work.mkdir(parents=True, exist_ok=True)

    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 1080))
    segments = manifest.get("segments", [])
    if not segments:
        raise RuntimeError("manifest has no segments")

    parts: List[Path] = []
    for idx, seg in enumerate(segments):
        fid = seg["file_id"]
        src_url = urls.get(fid)
        if not src_url:
            raise RuntimeError(f"missing url for file_id {fid}")

        # cache original once
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)

        # trim/normalize each part
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 2.0))
        dur = max(0.1, end - start)

        part_out = work / f"part_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(cache_vid),
            "-vf", f"scale={scale}:-2:flags=lanczos",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", os.getenv("FFMPEG_THREADS", "1"),
            str(part_out),
        ]
        run_ffmpeg(cmd)
        parts.append(part_out)

    # concat
    safe_name = _safe_name(filename or "final.mp4")
    final_path = sd / safe_name
    concat_list = work / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.as_posix()}'" for p in parts]))

    run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(final_path),
    ])

    return {"ok": True, **_public_or_download(session_id, safe_name)}


# ===================== Schemas =====================
class ProcessURLsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = "casual"
    product_link: Optional[str] = ""
    features_csv: Optional[str] = ""

class AutoManifestIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 1080
    max_total_sec: float = 12.0
    max_segments_per_file: int = 1

class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: Dict[str, Any]

class StitchAsyncIn(StitchIn):
    pass


# ===================== Routes =====================
@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": VERSION, "redis": True}

@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    """
    Register external/public/presigned video URLs for this session.
    """
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
        "urls": url_map,
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
        "created_at": int(time.time()),
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}

@app.post("/automanifest")
def automanifest(body: AutoManifestIn):
    """
    Simple heuristic: share max_total_sec across files.
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
    out = {
        "ok": True,
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": manifest,
    }
    save_json(sd / "manifest.json", out)
    return out

@app.post("/stitch")
def stitch(body: StitchIn):
    """Synchronous stitch (ok for very small drafts)."""
    try:
        return stitch_core(body.session_id, body.manifest, body.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    """Enqueue background stitch; the worker processes it."""
    q = get_q()
    job = q.enqueue(stitch_core, body.session_id, body.manifest, body.filename)
    return {"ok": True, "job_id": job.get_id()}

# ---- Jobs: test + status ----
def add(a: int, b: int) -> int:
    time.sleep(1.5)
    return a + b

@app.get("/jobs/test")
def jobs_test():
    q = get_q()
    job = q.enqueue(add, 2, 3)
    return {"job_id": job.get_id()}

@app.get("/jobs/{job_id}")
def jobs_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=get_q().connection)
        out = {"job_id": job.id, "status": job.get_status(), "result": job.result}
        if job.is_failed:
            exc = job.exc_info
            if isinstance(exc, bytes):
                try:
                    exc = exc.decode("utf-8", errors="replace")
                except Exception:
                    exc = str(exc)
            elif not isinstance(exc, str):
                exc = str(exc)
            info = exc.splitlines()
            out["error"] = "\n".join(info[-25:]) if info else "job failed (see worker logs)"
        return out
    except Exception as e:
        return {"ok": False, "error": f"lookup_failed: {e}"}

# Local download (only if you didn’t upload final.mp4 to S3)
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
