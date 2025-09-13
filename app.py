import os, uuid, json, shutil, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue
import boto3

# ================== App setup ==================
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-s3out"

# Session working directory (worker + web both write here, but we no longer depend on it for downloads)
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis / RQ
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL not set")
r = redis.from_url(REDIS_URL, decode_responses=True)
q = Queue(connection=r, default_timeout=60 * 60)  # up to 60 min job

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"

# AWS / S3
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET not set")

# Optional: if you prefer returning pretty URLs
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com").rstrip("/")

s3 = boto3.client("s3", region_name=AWS_REGION)

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

def run(cmd: list) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def _download_to_tmp(url: str, dst: Path):
    # Stream download (works with public or presigned URLs)
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

def _s3_upload(local_path: Path, key: str) -> str:
    """
    Upload a local file to s3://S3_BUCKET/key and return its public URL.
    Your bucket currently has a public read policy, so the URL will be directly usable.
    """
    s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"{S3_PUBLIC_BASE}/{key}"

# ================== Health ==================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-worker",
        "version": VERSION,
        "redis": True
    }

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

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: Dict[str, Any]
    fps: int = 30
    scale: int = 720

# ================== Endpoints ==================
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
        "urls": url_map,             # file_id -> URL (public/presigned)
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
        "created_at": int(time.time())
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}

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

    # Simple heuristic: take a short bite from each file until cap
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

def _stitch_do(*, urls: Dict[str, str], session_id: str, manifest: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Core stitcher:
      - downloads needed source chunks
      - trims and concatenates with ffmpeg
      - uploads the final to S3
      - returns {"ok": True, "s3_url": "..."}
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

        # cache original (download once)
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)

        # trim/encode consistent part
        part_out = work / f"part_{idx:03d}.mp4"
        _ffmpeg_trim(cache_vid, seg.get("start", 0.0), seg.get("end", 3.0), part_out, scale=scale, fps=fps)
        parts.append(part_out)

    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "final.mp4"
    final_path = sd / safe_name
    _concat_mp4s(parts, final_path)

    # Upload final to S3 at sessions/<session_id>/<filename>
    s3_key = f"sessions/{session_id}/{safe_name}"
    s3_url = _s3_upload(final_path, s3_key)

    # optional: clean up local to save space
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True, "s3_url": s3_url, "s3_key": s3_key}

def stitch_core_from_session(session_id: str, filename: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function executed by the worker (RQ).
    """
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")

    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise RuntimeError("no URLs registered for this session")

    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    # enqueue RQ job
    job = q.enqueue(stitch_core_from_session, body.session_id, body.filename, body.manifest)
    # store a little status doc in Redis so /jobs/{id} can read it even before Redis RQ result is ready
    init = {"job_id": job.id, "status": "queued", "result": None}
    r.set(_job_key(job.id), json.dumps(init))
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def jobs(job_id: str):
    # Try RQ job first
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        job = None

    if job and job.get_status() == "finished":
        # finished: job.result is a dict from stitch_core_from_session()
        res = {"job_id": job_id, "status": "finished", "result": job.result}
        r.set(_job_key(job_id), json.dumps(res))
        return res
    elif job and job.get_status() == "failed":
        # include traceback string
        tb = None
        try:
            tb = job.meta.get("traceback") if job.meta else None
        except Exception:
            pass
        res = {"job_id": job_id, "status": "failed", "result": None, "error": tb or "failed"}
        r.set(_job_key(job_id), json.dumps(res))
        return res

    # fallback: whatever we last wrote
    raw = r.get(_job_key(job_id))
    if raw:
        return json.loads(raw)

    return {"ok": False, "error": "job_not_found"}

# Kept for backward compatibility (not used once we return S3 URLs)
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
