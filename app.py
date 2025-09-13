# app.py
import os, uuid, json, shutil, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue, Job

# ==============================
# Config & setup
# ==============================
app = FastAPI(title="EditDNA Web API")
VERSION = "1.4.0-mvp-analyze-classify-script"

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required")

# IMPORTANT: binary-safe (no implicit utf-8) to avoid decode errors from RQ
rconn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=rconn)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

# ==============================
# Helpers
# ==============================
def sess_dir(session_id: str) -> Path:
    p = SESS_ROOT / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_json_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def _safe_json_dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def _download_to_tmp(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        with dst.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _ffmpeg_trim(input_path: Path, start: float, end: float, out_path: Path, scale: int, fps: int):
    dur = max(0.1, float(end) - float(start))
    _run([
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
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)])
    lst.unlink(missing_ok=True)

def _public_or_download(session_id: str, fname: str) -> Dict[str, str]:
    if S3_PUBLIC_BASE:
        # if later you upload to S3, keep this format consistent
        return {"public_url": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}"}
    return {"download_path": f"/download/{session_id}/{fname}"}

def _ffprobe_json(url_or_path: str) -> Dict[str, Any]:
    # Ask ffprobe for format & streams, return parsed json
    out = _run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        url_or_path
    ])
    return json.loads(out)

# ==============================
# Models
# ==============================
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
    manifest: dict  # {"segments":[...], "fps":30, "scale":720}

class StitchAsyncIn(BaseModel):
    session_id: str
    filename: str = "final.mp4"
    manifest: dict

class AnalyzeIn(BaseModel):
    session_id: str

class ClassifyIn(BaseModel):
    session_id: str

class ScriptIn(BaseModel):
    session_id: str
    tone: Optional[str] = "casual"
    product_link: Optional[str] = ""
    features_csv: Optional[str] = ""

# ==============================
# Health
# ==============================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-web",
        "version": VERSION,
        "redis": True,
        "region": AWS_REGION,
        "bucket": S3_BUCKET,
        "s3_public": bool(S3_PUBLIC_BASE),
    }

# ==============================
# Session: register URLs
# ==============================
@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta, url_map = [], {}
    for u in body.urls:
        fid = uuid.uuid4().hex[:8]
        files_meta.append({"file_id": fid, "source": "url"})
        url_map[fid] = u

    session = {
        "session_id": session_id,
        "files": files_meta,
        "urls": url_map,
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
        "analysis": {},   # file_id -> {duration,width,height,...}
        "best_file_id": ""  # filled by /classify
    }
    _safe_json_dump(sd / "session.json", session)
    return {"ok": True, "session_id": session_id, "files": files_meta}

# ==============================
# Analyze: ffprobe each URL
# ==============================
@app.post("/analyze")
def analyze(body: AnalyzeIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = _safe_json_load(meta_path)
    urls: Dict[str, str] = session.get("urls", {})

    analysis: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for fid, url in urls.items():
        try:
            info = _ffprobe_json(url)
            # defaults
            duration = None
            width = height = None

            # duration
            if "format" in info and "duration" in info["format"]:
                try:
                    duration = float(info["format"]["duration"])
                except Exception:
                    duration = None

            # streams (pick first video)
            for s in info.get("streams", []):
                if s.get("codec_type") == "video":
                    width = s.get("width")
                    height = s.get("height")
                    break

            analysis[fid] = {
                "duration": duration,
                "width": width,
                "height": height
            }
        except Exception as e:
            errors[fid] = str(e)

    session["analysis"] = analysis
    if errors:
        session.setdefault("analyze_errors", {}).update(errors)

    _safe_json_dump(meta_path, session)
    return {"ok": True, "session_id": body.session_id, "analysis": analysis, "errors": errors}

# ==============================
# Classify: pick "best" file
# Heuristic: prefer highest resolution, then longest duration
# ==============================
@app.post("/classify")
def classify(body: ClassifyIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = _safe_json_load(meta_path)

    analysis: Dict[str, Dict[str, Any]] = session.get("analysis", {})
    if not analysis:
        raise HTTPException(status_code=400, detail="run /analyze first")

    def score(rec: Dict[str, Any]) -> float:
        w = rec.get("width") or 0
        h = rec.get("height") or 0
        dur = rec.get("duration") or 0.0
        # simple: pixels count + a small weight for duration
        return (w * h) + (dur * 10_000)

    best_id = None
    best_score = -1.0
    for fid, rec in analysis.items():
        s = score(rec)
        if s > best_score:
            best_score = s
            best_id = fid

    session["best_file_id"] = best_id or ""
    _safe_json_dump(meta_path, session)
    return {"ok": True, "session_id": body.session_id, "best_file_id": best_id, "score": best_score}

# ==============================
# Script: lightweight template
# ==============================
@app.post("/script")
def script_gen(body: ScriptIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = _safe_json_load(meta_path)

    tone = (body.tone or session.get("tone") or "casual").lower()
    product = body.product_link or session.get("product_link") or ""
    features = [x.strip() for x in (body.features_csv or session.get("features_csv") or "").split(",") if x.strip()]

    beats = [
        {"t": 0, "line": "Hook: Tired of boring videos?"},
        {"t": 2, "line": "Show product in action."},
        {"t": 4, "line": "Highlight key benefit."},
        {"t": 6, "line": "Social proof / quick testimonial."},
        {"t": 8, "line": "Call to action: Try it now."},
    ]

    if features:
        beats[2]["line"] = f"Top features: {', '.join(features[:3])}."
    if product:
        beats[-1]["line"] = f"Call to action: Learn more at {product}"

    script = {
        "tone": tone,
        "beats": beats
    }
    session["script"] = script
    _safe_json_dump(meta_path, session)
    return {"ok": True, "session_id": body.session_id, "script": script}

# ==============================
# Auto-manifest (basic; can use best_file_id if present)
# ==============================
@app.post("/automanifest")
def automanifest(body: AutoManifestIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = _safe_json_load(meta_path)
    files = session.get("files") or []
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")

    # If we already chose a best file, bias towards it.
    best = session.get("best_file_id")
    order = files
    if best:
        order = sorted(files, key=lambda f: 0 if f["file_id"] == best else 1)

    segments = []
    total = 0.0
    per_seg = max(2.0, min(8.0, body.max_total_sec / max(1, len(files))))
    for f in order:
        if total >= body.max_total_sec:
            break
        take = min(per_seg, body.max_total_sec - total)
        segments.append({"file_id": f["file_id"], "start": 0.0, "end": round(take, 3)})
        total += take

    manifest = {"segments": segments, "fps": body.fps, "scale": body.scale}
    out = {"ok": True, "session_id": body.session_id, "filename": body.filename, "manifest": manifest}
    _safe_json_dump(sd / "manifest.json", out)
    return out

# ==============================
# Stitch (sync)
# ==============================
def _stitch_do(*, urls: Dict[str, str], session_id: str, manifest: dict, filename: str) -> Dict[str, Any]:
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
    return {"ok": True, **_public_or_download(session_id, safe_name)}

@app.post("/stitch")
def stitch(body: StitchIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = _safe_json_load(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered")

    return _stitch_do(urls=urls, session_id=body.session_id, manifest=body.manifest, filename=body.filename)

# ==============================
# Worker entry (called by RQ)
# ==============================
def stitch_core_from_session(session_id: str, filename: str, manifest: dict) -> dict:
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")
    session = _safe_json_load(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise RuntimeError("no URLs registered")
    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)

# ==============================
# Stitch (async) + Jobs
# ==============================
@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    # enqueue
    job: Job = q.enqueue(stitch_core_from_session, body.session_id, body.filename, body.manifest)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=rconn)
    except Exception:
        return {"ok": False, "error": "job_not_found", "job_id": job_id}

    status = job.get_status(refresh=True)
    result = None
    if status == "finished":
        # job.result is already JSON (dict) from _stitch_do; ensure it is serializable
        result = job.result
    elif status == "failed":
        # try to extract traceback or error string
        tb = getattr(job, "exc_info", None)
        err = None
        if tb:
            try:
                err = tb.decode() if isinstance(tb, (bytes, bytearray)) else str(tb)
            except Exception:
                err = str(tb)
        result = {"error": err or "failed"}

    # Also return the arguments we enqueued for easy debugging
    payload = {
        "job_id": job_id,
        "type": "stitch",
        "session_id": job.kwargs.get("session_id") if job.kwargs else None,
        "filename": job.kwargs.get("filename") if job.kwargs else None,
        "manifest": job.kwargs.get("manifest") if job.kwargs else None,
        "created_at": int(job.created_at.timestamp()) if job.created_at else None,
        "status": status,
        "rq_id": job.id,
        "result": result,
    }
    return payload

# ==============================
# Download (local fallback)
# ==============================
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
