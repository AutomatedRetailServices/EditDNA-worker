import os
import io
import json
import time
import uuid
import shutil
import pathlib
import subprocess
from typing import List, Dict, Any, Optional

import boto3
import requests
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from rq import Queue

# ------------------------------------------------------------------------------
# ENV & clients
# ------------------------------------------------------------------------------

REDIS_URL        = os.getenv("REDIS_URL", "")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET", "")
S3_PUBLIC_BASE   = os.getenv("S3_PUBLIC_BASE") or os.getenv("S3_PUBLIC_BUCKET")
S3_URL_MODE      = (os.getenv("S3_URL_MODE") or "auto").lower()  # auto | public | presigned

if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not S3_BUCKET:
    print("[WARN] AWS/S3 env vars not fully set. Uploads may fail.")

# Binary-safe redis (RQ passes pickles)
r_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=r_conn)

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

TMP_ROOT = pathlib.Path("/tmp/s2c_sessions")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Redis helpers
# ------------------------------------------------------------------------------

def _jset(key: str, obj: Any, ex: Optional[int] = None):
    r_conn.set(key, json.dumps(obj).encode("utf-8"), ex=ex)

def _jget(key: str) -> Optional[Any]:
    raw = r_conn.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _session_key(session_id: str) -> str:
    return f"session:{session_id}:files"

def _analysis_key(session_id: str) -> str:
    return f"session:{session_id}:analysis"

def _manifest_key(session_id: str) -> str:
    return f"session:{session_id}:manifest"

def _jobs_key(job_id: str) -> str:
    return f"job:{job_id}"

def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip().replace(" ", "_")

# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------

def _download_to(tmp_dir: pathlib.Path, url: str, out_name: str) -> pathlib.Path:
    out_path = tmp_dir / out_name
    with requests.get(url, stream=True, timeout=60) as res:
        res.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(res.raw, f)
    return out_path

def _probe_duration(path: pathlib.Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(out)
    except Exception:
        return 0.0

def _ffmpeg_trim(src: pathlib.Path, dst: pathlib.Path, start: float, end: float, scale: int, fps: int):
    dur = max(0.0, end - start)
    if dur <= 0:
        raise RuntimeError("Invalid segment duration")
    vf = f"scale=-2:{scale}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{dur:.3f}",
        "-r", str(fps),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def _ffmpeg_concat(tsv: List[pathlib.Path], out_path: pathlib.Path):
    list_file = out_path.parent / f"concat_{uuid.uuid4().hex}.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in tsv:
                f.write(f"file '{p.as_posix()}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        if list_file.exists():
            list_file.unlink(missing_ok=True)

def _upload_final_to_s3(session_id: str, filename: str, local_path: pathlib.Path) -> str:
    key = f"sessions/{session_id}/{filename}"
    _s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs={})
    if S3_URL_MODE == "presigned":
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=900,
        )
    if S3_PUBLIC_BASE:
        base = S3_PUBLIC_BASE.rstrip("/")
        return f"{base}/{key}"
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=900,
    )

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class ProcessUrlsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class AutoManifestIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    max_total_sec: int = 12
    max_segments_per_file: int = 1

class StitchIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    manifest: Dict[str, Any]

class AnalyzeIn(BaseModel):
    session_id: str
    script_text: Optional[str] = None

# ------------------------------------------------------------------------------
# Core jobs (these functions MUST stay at module top-level so RQ can import app.X)
# ------------------------------------------------------------------------------

def analyze_core_from_session(session_id: str, script_text: Optional[str] = None) -> Dict[str, Any]:
    files = _jget(_session_key(session_id)) or {}
    if not files:
        raise RuntimeError("session not found")

    work_dir = TMP_ROOT / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    results = {}
    for file_id, url in files.items():
        cache_name = f"{file_id}.src"
        src_path = cache_dir / cache_name
        if not src_path.exists():
            _download_to(cache_dir, url, cache_name)
        dur = _probe_duration(src_path)
        score = 0.8 if dur >= 1.5 else 0.4
        results[file_id] = {"duration": dur, "score": score}

    _jset(_analysis_key(session_id), {"script_used": bool(script_text), "results": results}, ex=60*60*24)
    return {"ok": True, "analyzed": len(results)}

def stitch_core(session_id: str, filename: str, manifest: Dict[str, Any], fps: int, scale: int) -> Dict[str, Any]:
    files = _jget(_session_key(session_id)) or {}
    if not files:
        raise RuntimeError("session not found")

    work_dir = TMP_ROOT / session_id
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    tmp_segments: List[pathlib.Path] = []
    try:
        for idx, seg in enumerate(manifest.get("segments", [])):
            file_id = seg["file_id"]
            start = float(seg.get("start", 0))
            end   = float(seg.get("end", start + 1))
            src_url = files.get(file_id)
            if not src_url:
                raise RuntimeError(f"file_id not found: {file_id}")

            cache_name = f"{file_id}.src"
            cache_path = cache_dir / cache_name
            if not cache_path.exists():
                _download_to(cache_dir, src_url, cache_name)

            out_piece = work_dir / f"piece_{idx:03d}.mp4"
            _ffmpeg_trim(cache_path, out_piece, start, end, scale, fps)
            tmp_segments.append(out_piece)

        safe_name = _safe_name(filename or "final.mp4")
        final_path = work_dir / safe_name
        _ffmpeg_concat(tmp_segments, final_path)
        public_url = _upload_final_to_s3(session_id, safe_name, final_path)
        return {"ok": True, "public_url": public_url}
    finally:
        for p in tmp_segments:
            p.unlink(missing_ok=True)

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-web", "version": "1.5.0-nop", "redis": True}

# --- simple NOP enqueue to prove worker consumption ---------------------------

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(lambda: "NOP")  # trivial job
    _jset(_jobs_key(job.id), {"type": "nop", "created_at": int(time.time())}, ex=60*30)
    return {"ok": True, "job_id": job.id}

# --- Session: store URLs ------------------------------------------------------

@app.post("/process_urls")
def process_urls(inp: ProcessUrlsIn):
    if not inp.urls:
        raise HTTPException(400, "urls required")
    session_id = uuid.uuid4().hex
    files_map = {}
    for url in inp.urls:
        fid = uuid.uuid4().hex[:8]
        files_map[fid] = url.strip()
    _jset(_session_key(session_id), files_map, ex=60*60*24)
    return {"ok": True, "session_id": session_id, "files": [{"file_id": fid, "source": "url"} for fid in files_map.keys()]}

# --- Analyze (async) ----------------------------------------------------------

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    job = q.enqueue(analyze_core_from_session, inp.session_id, inp.script_text, job_timeout=60*30)
    _jset(_jobs_key(job.id), {"type": "analyze", "session_id": inp.session_id, "created_at": int(time.time())}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

# --- Auto-manifest (very simple) ----------------------------------------------

@app.post("/manifest")
def manifest(inp: AutoManifestIn):
    files = _jget(_session_key(inp.session_id)) or {}
    if not files:
        raise HTTPException(404, "session not found")

    # naive equal split
    per = max(0.4, float(inp.max_total_sec) / max(1, len(files)))
    segments = []
    total = 0.0
    for fid in list(files.keys()):
        if total >= inp.max_total_sec:
            break
        take = min(2.0, per)  # keep 2.0s caps for now
        segments.append({"file_id": fid, "start": 0.0, "end": float(take)})
        total += take

    manifest = {"segments": segments, "fps": inp.fps, "scale": inp.scale}
    _jset(_manifest_key(inp.session_id), manifest, ex=60*60*24)
    return {"ok": True, "session_id": inp.session_id, "manifest": manifest, "filename": inp.filename}

# --- Stitch (async) -----------------------------------------------------------

@app.post("/stitch")
def stitch(inp: StitchIn):
    manifest = inp.manifest or _jget(_manifest_key(inp.session_id))
    if not manifest:
        raise HTTPException(400, "manifest required (call /manifest first)")
    fps = int(manifest.get("fps", inp.fps))
    scale = int(manifest.get("scale", inp.scale))
    job = q.enqueue(stitch_core, inp.session_id, inp.filename, manifest, fps, scale, job_timeout=60*60)
    _jset(_jobs_key(job.id), {
        "type": "stitch", "session_id": inp.session_id, "filename": inp.filename,
        "manifest": manifest, "created_at": int(time.time())
    }, ex=60*60*12)
    return {"ok": True, "job_id": job.id, "session_id": inp.session_id}

# --- Jobs ---------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    # Import here to avoid version/import errors at import time
    try:
        from rq.job import Job  # type: ignore
    except Exception:
        return {"job_id": job_id, "status": "unknown", "error": "rq.job.Job import failed"}
    try:
        job = Job.fetch(job_id, connection=r_conn)
    except Exception:
        meta = _jget(_jobs_key(job_id)) or {}
        return {"job_id": job_id, "status": "queued", **({"meta": meta} if meta else {})}
    status = job.get_status()
    resp: Dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "finished":
        resp["result"] = job.result
    elif status == "failed":
        resp["result"] = None
        tb = getattr(job, "exc_info", None)
        if tb:
            resp["error"] = tb.splitlines()[-1] if isinstance(tb, str) else "failed"
    return resp

# --- Debug: inspect a session -------------------------------------------------

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    return {
        "files": _jget(_session_key(session_id)),
        "analysis": _jget(_analysis_key(session_id)),
        "manifest": _jget(_manifest_key(session_id)),
    }

# --- Optional local file serve ------------------------------------------------

@app.get("/download/{session_id}/{filename}")
def download_local(session_id: str, filename: str):
    path = (TMP_ROOT / session_id / _safe_name(filename)).resolve()
    if not path.exists():
        raise HTTPException(404, "file not found")
    def _iter():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    return StreamingResponse(_iter(), media_type="video/mp4")

# root 404
@app.get("/")
def root():
    return PlainTextResponse("Not Found", status_code=404)
