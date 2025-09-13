import os, uuid, json, time, shutil, subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import requests
import redis
from rq import Queue

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# ============== App setup ==============
app = FastAPI(title="EditDNA Web API")
VERSION = "1.3.1-s3-logging"

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis / RQ
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
r = redis.from_url(REDIS_URL, decode_responses=True)
q = Queue("default", connection=r)

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"

# S3
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET  = os.getenv("S3_BUCKET", "")  # MUST match exact bucket name
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

if not S3_BUCKET:
    raise RuntimeError("Missing S3_BUCKET")
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(s3={"addressing_style": "virtual"})
)

# ============== Helpers ==============
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
    print(f"[DL] → {url} -> {dst}")
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        with dst.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _ffmpeg_trim(input_path: Path, start: float, end: float, out_path: Path, scale: int, fps: int):
    dur = max(0.1, float(end) - float(start))
    print(f"[FFMPEG trim] {input_path} [{start:.3f}-{end:.3f}] -> {out_path} (scale={scale}, fps={fps})")
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
    print(f"[FFMPEG concat] {len(parts)} parts -> {out_path}")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)])
    lst.unlink(missing_ok=True)

def _s3_key_for(session_id: str, fname: str) -> str:
    return f"sessions/{session_id}/{fname}"

def _s3_upload_and_link(local_path: Path, session_id: str, filename: str) -> Dict[str, Any]:
    """
    Uploads to S3 and returns a dict that includes:
      - bucket
      - key
      - public_url (if S3_PUBLIC_BASE)
      - s3_path (s3://bucket/key)
    Also prints verbose logs to Render logs.
    """
    key = _s3_key_for(session_id, filename)
    print(f"[S3 PUT] bucket={S3_BUCKET} key={key} local={local_path}")
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"}
        )
    except ClientError as e:
        print(f"[S3 ERROR] upload failed: {e}")
        raise

    s3_path = f"s3://{S3_BUCKET}/{key}"
    public_url = f"{S3_PUBLIC_BASE}/{key}" if S3_PUBLIC_BASE else ""
    print(f"[S3 OK] {s3_path} public_url={public_url or '(none)'}")
    out = {"bucket": S3_BUCKET, "key": key, "s3_path": s3_path}
    if public_url:
        out["public_url"] = public_url
    else:
        out["download_path"] = f"/download/{session_id}/{filename}"
    return out

# ============== Models ==============
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

# ============== Health ==============
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "editdna-worker",
        "version": VERSION,
        "redis": True
    }

# ============== Sessions from public URLs ==============
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
        "urls": url_map,  # file_id -> URL
        "tone": body.tone,
        "features_csv": body.features_csv,
        "product_link": body.product_link,
    }
    save_json(sd / "session.json", session_json)
    print(f"[SESSION] created {session_id} with {len(files_meta)} files")
    return {"ok": True, "session_id": session_id, "files": files_meta}

# ============== Auto manifest (simple heuristic) ==============
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
    print(f"[AUTOMANIFEST] session={body.session_id} segments={len(segments)}")
    return out

# ============== Core stitch (sync) ==============
def _stitch_do(*, urls: Dict[str, str], session_id: str, manifest: dict, filename: str) -> Dict[str, Any]:
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

    # Upload to S3 + return link (with verbose logs)
    out = _s3_upload_and_link(final_path, session_id=session_id, filename=safe_name)

    # Optional: keep some disk clean
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True, **out}

@app.post("/stitch")
def stitch(body: StitchIn):
    # sync stitch (short jobs)
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        raise HTTPException(status_code=400, detail="no URLs registered for this session")

    print(f"[STITCH sync] session={body.session_id} → {body.filename}")
    return _stitch_do(urls=urls, session_id=body.session_id, manifest=body.manifest, filename=body.filename)

# ============== RQ Background jobs ==============
def stitch_core_from_session(session_id: str, manifest: dict, filename: str) -> Dict[str, Any]:
    """
    RQ worker calls this function. We keep prints so they appear in worker logs.
    """
    print(f"[RQ] stitch_core_from_session session={session_id} filename={filename}")
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        print(f"[RQ] ERROR: session not found: {session_id}")
        raise RuntimeError("session not found")
    session = load_json(meta_path)
    urls = session.get("urls") or {}
    if not urls:
        print(f"[RQ] ERROR: no URLs map in session {session_id}")
        raise RuntimeError("no urls in session")
    # Do the job and return the same dict as sync stitch
    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)

@app.post("/stitch_async")
def stitch_async(body: StitchIn):
    job_id = uuid.uuid4().hex
    job_payload = {
        "job_id": job_id,
        "type": "stitch",
        "session_id": body.session_id,
        "filename": body.filename,
        "manifest": body.manifest,
        "created_at": int(time.time()),
        "status": "queued",
    }
    r.set(_job_key(job_id), json.dumps(job_payload))
    print(f"[RQ ENQUEUE] job_id={job_id} session={body.session_id} filename={body.filename}")
    # enqueue
    rq_job = q.enqueue(stitch_core_from_session, body.session_id, body.manifest, body.filename, job_id=job_id)
    # store RQ id for debugging
    job_payload["rq_id"] = rq_job.id
    r.set(_job_key(job_id), json.dumps(job_payload))
    return {"ok": True, "job_id": job_id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    raw = r.get(_job_key(job_id))
    if not raw:
        # old jobs (pre-RQ) could be missing; return a friendly error
        return {"ok": False, "error": "job_not_found"}
    job = json.loads(raw)
    # If finished stored result in Redis
    if job.get("status") in ("finished", "failed"):
        return job
    # Try to peek RQ state to reflect updated status
    try:
        from rq.job import Job
        rq_id = job.get("rq_id")
        if rq_id:
            j = Job.fetch(rq_id, connection=r)
            if j.is_finished:
                job["status"] = "finished"
                job["result"] = j.result
                r.set(_job_key(job_id), json.dumps(job))
            elif j.is_failed:
                job["status"] = "failed"
                # include traceback, which will show S3 bucket/key if upload failed
                job["error"] = j.exc_info
                r.set(_job_key(job_id), json.dumps(job))
            else:
                job["status"] = "started"
    except Exception as e:
        job["note"] = f"rq_lookup_failed: {e}"
    return job

# ============== Local download fallback (rarely used now) ==============
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
