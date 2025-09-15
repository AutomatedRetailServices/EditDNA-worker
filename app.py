import os
import io
import json
import time
import uuid
import shutil
import tempfile
import pathlib
import subprocess
from typing import List, Dict, Any, Optional

import boto3
import requests
import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field
from rq import Queue
from rq.job import Job

# ------------------------------------------------------------------------------
# ENV & global clients
# ------------------------------------------------------------------------------

REDIS_URL        = os.getenv("REDIS_URL", "")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET", "")
S3_PUBLIC_BASE   = os.getenv("S3_PUBLIC_BASE") or os.getenv("S3_PUBLIC_BUCKET")  # support either name
S3_URL_MODE      = os.getenv("S3_URL_MODE", "auto").lower()  # auto | public | presigned

if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY or not S3_BUCKET:
    # We only fail when we actually try to upload; warn now for clarity.
    print("[WARN] AWS/S3 env vars are not fully set. Uploads may fail.")

# Binary-safe redis (avoid UTF-8 decode problems with RQ payloads)
r_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=r_conn)

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION or "us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# All temp work under a stable root so /download can access by session
TMP_ROOT = pathlib.Path("/tmp/s2c_sessions")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Small helpers (Redis keys, JSON)
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
# Download / Upload helpers
# ------------------------------------------------------------------------------

def _download_to(tmp_dir: pathlib.Path, url: str, out_name: str) -> pathlib.Path:
    out_path = tmp_dir / out_name
    with requests.get(url, stream=True, timeout=60) as res:
        res.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(res.raw, f)
    return out_path

def _probe_duration(path: pathlib.Path) -> float:
    """Return duration in seconds using ffprobe (if available)."""
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
    """
    Trim using -ss/-to and scale/fps. We re-encode for uniform output.
    """
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
    """
    Concat with concat demuxer: needs an intermediate list file.
    """
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
    """
    Upload to s3://<bucket>/sessions/<session_id>/<filename>
    Returns a URL:
      - public URL if bucket allows public read and S3_URL_MODE in {public, auto with base}
      - presigned URL otherwise (15 min)
    """
    key = f"sessions/{session_id}/{filename}"
    extra: Dict[str, Any] = {}
    # DO NOT set ACL on ObjectOwnership=BucketOwnerEnforced buckets
    _s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra)

    # build URL
    if S3_URL_MODE == "presigned":
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=900,
        )
    # public or auto
    base = S3_PUBLIC_BASE
    if base:
        base = base.rstrip("/")
        return f"{base}/{key}"
    # fallback to presigned
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

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    manifest: Dict[str, Any]

class AnalyzeIn(BaseModel):
    session_id: str
    script_text: Optional[str] = None  # future: compare transcription to this

class ChooseBestIn(BaseModel):
    session_id: str
    target_sec: int = 12
    max_segments_per_file: int = 1
    fps: int = 30
    scale: int = 720

# ------------------------------------------------------------------------------
# Core jobs (run on worker)
# ------------------------------------------------------------------------------

def stitch_core(session_id: str, filename: str, manifest: Dict[str, Any], fps: int, scale: int) -> Dict[str, Any]:
    """
    Build final video per manifest and upload to S3. Returns {"ok": True, "public_url": "..."}.
    Manifest format:
      { "segments": [ { "file_id": "abcd", "start": 0.0, "end": 2.4 }, ... ] }
    """
    files = _jget(_session_key(session_id)) or {}
    if not files:
        raise RuntimeError("session not found")

    work_dir = TMP_ROOT / session_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # cache inputs
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Prepare trimmed clips
    tmp_segments: List[pathlib.Path] = []
    try:
        for idx, seg in enumerate(manifest.get("segments", [])):
            file_id = seg["file_id"]
            start = float(seg.get("start", 0))
            end   = float(seg.get("end", start + 1))
            src_url = files.get(file_id)
            if not src_url:
                raise RuntimeError(f"file_id not found in session: {file_id}")

            cache_name = f"{file_id}.src"
            cache_path = cache_dir / cache_name
            if not cache_path.exists():
                _download_to(cache_dir, src_url, cache_name)

            out_piece = work_dir / f"piece_{idx:03d}.mp4"
            _ffmpeg_trim(cache_path, out_piece, start, end, scale, fps)
            tmp_segments.append(out_piece)

        # concat
        safe_name = _safe_name(filename or "final.mp4")
        final_path = work_dir / safe_name
        _ffmpeg_concat(tmp_segments, final_path)

        public_url = _upload_final_to_s3(session_id, safe_name, final_path)
        result = {"ok": True, "public_url": public_url}
        return result
    finally:
        # keep work_dir for /download if needed; we already uploaded
        for p in tmp_segments:
            p.unlink(missing_ok=True)

def analyze_core_from_session(session_id: str, script_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Stub analysis:
      - Downloads each file once (cache)
      - Uses ffprobe to get duration
      - Produces a simple score: longer than 1.5s -> score 0.8, else 0.4
    Stores results in Redis for the session.
    """
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
        # simple rule score (placeholder for STT + LLM)
        score = 0.8 if dur >= 1.5 else 0.4
        results[file_id] = {"duration": dur, "score": score}

    _jset(_analysis_key(session_id), {"script_used": bool(script_text), "results": results}, ex=60*60*24)
    return {"ok": True, "analyzed": len(results)}

def choose_best_core(session_id: str, target_sec: int, max_segments_per_file: int, fps: int, scale: int) -> Dict[str, Any]:
    """
    Build an auto manifest using analysis results (fallback to equal slices).
    Policy:
      - Sort files by score desc (duration if no analysis)
      - Take up to target_sec total, at most max_segments_per_file segments per file
      - Each picked segment is [0, min(2.4, remaining)]
    Stores manifest in Redis and returns it.
    """
    files = _jget(_session_key(session_id)) or {}
    if not files:
        raise RuntimeError("session not found")

    analysis = _jget(_analysis_key(session_id)) or {}
    scored = []
    for fid, url in files.items():
        info = (analysis.get("results") or {}).get(fid) or {}
        score = float(info.get("score", 0.5))
        dur   = float(info.get("duration", 2.0))
        scored.append((fid, score, dur))
    scored.sort(key=lambda t: t[1], reverse=True)

    remaining = float(target_sec)
    segments = []
    for fid, _score, dur in scored:
        if remaining <= 0:
            break
        count = max_segments_per_file
        while count > 0 and remaining > 0:
            piece = min(2.4, remaining, max(0.4, dur))  # keep >=0.4s if possible
            segments.append({"file_id": fid, "start": 0.0, "end": float(min(piece, dur))})
            remaining -= float(piece)
            count -= 1

    manifest = {"segments": segments, "fps": fps, "scale": scale}
    _jset(_manifest_key(session_id), manifest, ex=60*60*24)
    return {"ok": True, "manifest": manifest}

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": "1.4.0-analyze-scaffold", "redis": True}

# --- Session: store URLs -------------------------------------------------------

@app.post("/process_urls")
def process_urls(inp: ProcessUrlsIn):
    if not inp.urls:
        raise HTTPException(400, "urls required")
    # create a new session id
    session_id = uuid.uuid4().hex
    files_map = {}
    for url in inp.urls:
        fid = uuid.uuid4().hex[:8]
        files_map[fid] = url.strip()
    _jset(_session_key(session_id), files_map, ex=60*60*24)  # keep 24h

    return {
        "ok": True,
        "session_id": session_id,
        "files": [{"file_id": fid, "source": "url"} for fid in files_map.keys()]
    }

# --- Auto manifest (simple) ----------------------------------------------------

@app.post("/automanifest")
def automanifest(inp: AutoManifestIn):
    files = _jget(_session_key(inp.session_id)) or {}
    if not files:
        raise HTTPException(404, "session not found")

    # simple equal-split up to max_total_sec across first N files
    per = max(0.4, float(inp.max_total_sec) / max(1, len(files)))
    segments = []
    for fid in list(files.keys()):
        for _ in range(inp.max_segments_per_file):
            if len(segments) * per >= inp.max_total_sec:
                break
            segments.append({"file_id": fid, "start": 0.0, "end": float(min(2.4, per))})

    manifest = {"segments": segments, "fps": inp.fps, "scale": inp.scale}
    _jset(_manifest_key(inp.session_id), manifest, ex=60*60*24)
    return {"ok": True, "session_id": inp.session_id, "filename": inp.filename, "manifest": manifest}

# --- Analyze (stub) ------------------------------------------------------------

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    # queue job
    job: Job = q.enqueue(analyze_core_from_session, inp.session_id, inp.script_text, job_timeout=60*30)
    _jset(_jobs_key(job.id), {"type": "analyze", "session_id": inp.session_id, "created_at": int(time.time())}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

# --- Choose best (auto manifest using analysis) --------------------------------

@app.post("/choose_best")
def choose_best(inp: ChooseBestIn):
    job: Job = q.enqueue(
        choose_best_core,
        inp.session_id, inp.target_sec, inp.max_segments_per_file, inp.fps, inp.scale,
        job_timeout=60*30,
    )
    _jset(_jobs_key(job.id), {"type": "choose_best", "session_id": inp.session_id, "created_at": int(time.time())}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

# --- Stitch (async) ------------------------------------------------------------

@app.post("/stitch_async")
def stitch_async(inp: StitchAsyncIn):
    # Allow caller to pass a manifest, or fallback to last saved manifest for this session
    manifest = inp.manifest or _jget(_manifest_key(inp.session_id))
    if not manifest:
        raise HTTPException(400, "manifest required (or call /automanifest or /choose_best first)")

    job: Job = q.enqueue(
        stitch_core,
        inp.session_id,
        inp.filename,
        manifest,
        int(inp.manifest.get("fps", inp.fps)),
        int(inp.manifest.get("scale", inp.scale)),
        job_timeout=60*60,  # allow long ffmpeg
    )
    _jset(_jobs_key(job.id), {
        "type": "stitch",
        "session_id": inp.session_id,
        "filename": inp.filename,
        "manifest": manifest,
        "created_at": int(time.time()),
    }, ex=60*60*12)

    return {"ok": True, "job_id": job.id}

# --- Jobs ----------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r_conn)
    except Exception:
        # still return what we can
        meta = _jget(_jobs_key(job_id)) or {}
        return {"job_id": job_id, "status": "queued", **({"meta": meta} if meta else {})}

    status = job.get_status()
    resp: Dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "finished":
        resp["result"] = job.result
    elif status == "failed":
        resp["result"] = None
        # include traceback message (short)
        tb = getattr(job, "exc_info", None)
        if tb:
            resp["error"] = tb.splitlines()[-1] if isinstance(tb, str) else "failed"
    return resp

# --- Download (optional local file serve) --------------------------------------

@app.get("/download/{session_id}/{filename}")
def download_local(session_id: str, filename: str):
    """
    If the file exists in /tmp we stream it. Otherwise 404.
    (Most users will click the S3 URL returned by /jobs when stitch finishes.)
    """
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

# root 404 (Render checks /)
@app.get("/")
def root():
    return PlainTextResponse("Not Found", status_code=404)
