# app.py
import os, uuid, json, time, shutil, subprocess
from pathlib import Path
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Redis / RQ
import redis
from rq import Queue, Worker, Connection
from rq.job import Job

# Optional S3 upload
try:
    import boto3
except Exception:
    boto3 = None

# =========================
# App & storage setup
# =========================
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-rq-pass-urls+s3"

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Env
REDIS_URL = os.getenv("REDIS_URL", "").strip()
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# binary-safe connection (avoid utf-8 decode issues)
rconn = redis.from_url(REDIS_URL, decode_responses=False)

# S3 (optional but recommended for final outputs)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")  # e.g. https://bucket.s3.us-east-1.amazonaws.com
_s3_client = None
if S3_BUCKET and boto3:
    _s3_client = boto3.client("s3", region_name=AWS_REGION)

# =========================
# Helpers
# =========================
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

def _safe_name(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or "final.mp4"

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

def _upload_final_to_s3(session_id: str, filename: str, local_path: Path) -> Optional[str]:
    """
    Upload final MP4 to s3://S3_BUCKET/sessions/<session_id>/<filename>
    Returns public URL if S3_PUBLIC_BASE is configured; else None.
    """
    if not (_s3_client and S3_BUCKET):
        return None
    key = f"sessions/{session_id}/{filename}"
    _s3_client.upload_file(
        Filename=str(local_path),
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"}  # needs bucket policy allowing public-read
    )
    if S3_PUBLIC_BASE:
        return f"{S3_PUBLIC_BASE}/{key}"
    return None

# =========================
# Models
# =========================
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
    manifest: dict  # {"segments":[{file_id, start, end}], "fps":30, "scale":720}

class StitchAsyncIn(StitchIn):
    pass

# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-web",
        "version": VERSION,
        "redis": True,
        "s3_bucket": bool(S3_BUCKET),
        "s3_public": bool(S3_PUBLIC_BASE),
    }

# =========================
# Core business endpoints
# =========================
@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    """
    Register a session with a list of public/presigned URLs.
    We keep the mapping (file_id -> URL) on *web* side so we can pass it to the worker later.
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
    }
    save_json(sd / "session.json", session_json)
    return {"ok": True, "session_id": session_id, "files": files_meta}

@app.post("/automanifest")
def automanifest(body: AutoManifestIn):
    """
    Simple heuristic: take ~even bite from each file up to max_total_sec.
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

def _stitch_do(session_id: str, urls: Dict[str, str], manifest: dict, filename: str) -> dict:
    """
    Actual stitching logic (used by both sync and async worker).
    Downloads sources, trims, concatenates, then optionally uploads final to S3.
    Returns either public_url (preferred) or a local download path.
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

        # Cache original source file
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)

        # Produce trimmed/normalized clip
        part_out = work / f"part_{idx:03d}.mp4"
        _ffmpeg_trim(cache_vid, seg.get("start", 0.0), seg.get("end", 3.0), part_out, scale=scale, fps=fps)
        parts.append(part_out)

    safe_name = _safe_name(filename)
    final_path = sd / safe_name
    _concat_mp4s(parts, final_path)

    # Try uploading to S3 (if configured)
    public_url = _upload_final_to_s3(session_id, safe_name, final_path)
    if public_url:
        return {"ok": True, "public_url": public_url}

    # Fallback to local download route (on the SAME machine that produced the file)
    # NOTE: If the job ran on the worker machine, this URL will not work.
    return {"ok": True, "download_path": f"/download/{session_id}/{safe_name}"}

@app.post("/stitch")
def stitch_sync(body: StitchIn):
    """
    Synchronous stitch. Use only for very short reels.
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    try:
        result = _stitch_do(
            session_id=body.session_id,
            urls=urls,
            manifest=body.manifest,
            filename=_safe_name(body.filename),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------- RQ worker function (runs on worker box) -------------
def stitch_core(session_id: str, filename: str, manifest: dict, urls: Dict[str, str]) -> dict:
    """
    Worker entrypoint. It receives URLs directly (no local disk read),
    stitches, then uploads final to S3 if configured.
    """
    return _stitch_do(
        session_id=session_id,
        urls=urls,
        manifest=manifest,
        filename=_safe_name(filename),
    )

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    """
    Enqueue background job. IMPORTANT: we read session.json on *web*,
    extract the URLs, and pass them into the RQ job so the worker does
    not need to access web's local disk.
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    with Connection(rconn):
        q = Queue("default")
        job = q.enqueue(
            stitch_core,
            body.session_id,
            _safe_name(body.filename),
            body.manifest,
            urls,  # pass URLs explicitly
        )
    return {"ok": True, "job_id": job.get_id()}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """
    Query job by ID. Returns status + result or error trace if failed.
    """
    try:
        job = Job.fetch(job_id, connection=rconn)
    except Exception:
        return {"ok": False, "error": "job_not_found"}

    status = job.get_status()
    out = {
        "job_id": job_id,
        "status": status,
    }
    if status == "finished":
        try:
            out["result"] = job.result
        except Exception:
            out["result"] = None
    elif status == "failed":
        out["result"] = None
        out["error"] = getattr(job, "exc_info", "") or "unknown_error"
    return out

# =========================
# Local download (fallback)
# =========================
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
