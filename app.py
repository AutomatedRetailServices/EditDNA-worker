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
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from rq import Queue, Job

# ------------------------------------------------------------------------------
# ENV & global clients
# ------------------------------------------------------------------------------

REDIS_URL        = os.getenv("REDIS_URL", "")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET", "")
S3_PUBLIC_BASE   = os.getenv("S3_PUBLIC_BASE") or os.getenv("S3_PUBLIC_BUCKET")
S3_URL_MODE      = os.getenv("S3_URL_MODE", "auto").lower()

if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

r_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=r_conn)

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION or "us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

TMP_ROOT = pathlib.Path("/tmp/s2c_sessions")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Helpers
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
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15).decode("utf-8").strip()
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
    extra: Dict[str, Any] = {}
    _s3.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra)

    if S3_URL_MODE == "presigned":
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=900,
        )
    base = S3_PUBLIC_BASE
    if base:
        base = base.rstrip("/")
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
    manifest: Optional[Dict[str, Any]] = None

class AnalyzeIn(BaseModel):
    session_id: str

class ChooseBestIn(BaseModel):
    session_id: str
    target_sec: int = 12
    max_segments_per_file: int = 1
    fps: int = 30
    scale: int = 720

# ------------------------------------------------------------------------------
# Core jobs
# ------------------------------------------------------------------------------

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
            if end == 0:
                end = float("inf")
            src_url = files.get(file_id)
            if not src_url:
                raise RuntimeError(f"file_id not found in session: {file_id}")

            cache_name = f"{file_id}.src"
            cache_path = cache_dir / cache_name
            if not cache_path.exists():
                _download_to(cache_dir, src_url, cache_name)

            duration = _probe_duration(cache_path)
            if end == float("inf"):
                end = duration

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
        try:
            cache_name = f"{file_id}.src"
            src_path = cache_dir / cache_name
            if not src_path.exists():
                _download_to(cache_dir, url, cache_name)

            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(src_path)
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, _ = proc.communicate(timeout=15)
                dur = float(out.decode("utf-8").strip() or 0.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                dur = 0.0

            score = 0.8 if dur >= 1.5 else 0.4
            results[file_id] = {"duration": dur, "score": score}
        except Exception as e:
            results[file_id] = {"duration": 0.0, "score": 0.0, "error": str(e)}

    _jset(_analysis_key(session_id), {"results": results}, ex=60*60*24)
    return {"ok": True, "analyzed": len(results)}

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": "1.6.6", "redis": True}

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
    return {"ok": True, "session_id": session_id, "files": [{"file_id": fid} for fid in files_map.keys()]}

@app.post("/automanifest")
def automanifest(inp: AutoManifestIn):
    files = _jget(_session_key(inp.session_id)) or {}
    if not files:
        raise HTTPException(404, "session not found")
    per = max(0.4, float(inp.max_total_sec) / max(1, len(files)))
    segments = []
    for fid in list(files.keys()):
        for _ in range(inp.max_segments_per_file):
            if len(segments) * per >= inp.max_total_sec:
                break
            segments.append({"file_id": fid, "start": 0.0, "end": float(min(2.4, per))})
    manifest = {"segments": segments, "fps": inp.fps, "scale": inp.scale}
    _jset(_manifest_key(inp.session_id), manifest, ex=60*60*24)
    return {"ok": True, "manifest": manifest}

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    job: Job = q.enqueue(analyze_core_from_session, inp.session_id, job_timeout=60*5)
    _jset(_jobs_key(job.id), {"type": "analyze", "session_id": inp.session_id}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

@app.post("/choose_best")
def choose_best(inp: ChooseBestIn):
    job: Job = q.enqueue(analyze_core_from_session, inp.session_id, job_timeout=60*5)
    _jset(_jobs_key(job.id), {"type": "choose_best", "session_id": inp.session_id}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

@app.post("/stitch")
def stitch(inp: StitchAsyncIn):
    manifest = inp.manifest or _jget(_manifest_key(inp.session_id))
    if not manifest:
        raise HTTPException(400, "manifest required")
    job: Job = q.enqueue(stitch_core, inp.session_id, inp.filename, manifest, inp.fps, inp.scale, job_timeout=60*30)
    _jset(_jobs_key(job.id), {"type": "stitch", "session_id": inp.session_id}, ex=60*60*12)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r_conn)
    except Exception:
        meta = _jget(_jobs_key(job_id)) or {}
        return {"job_id": job_id, "status": "queued", "meta": meta}
    status = job.get_status()
    resp: Dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "finished":
        resp["result"] = job.result
    elif status == "failed":
        tb = getattr(job, "exc_info", None)
        resp["error"] = tb.splitlines()[-1] if isinstance(tb, str) else "failed"
    return resp

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

@app.get("/")
def root():
    return PlainTextResponse("Not Found", status_code=404)
