# app.py — web API for EditDNA (with synchronous /process + proxy pipeline + RunPod autoscale)
from __future__ import annotations

import os
import json
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# Optional Redis / RQ: app still works without them for /process
try:
    from redis import Redis  # type: ignore
    from rq import Queue     # type: ignore
    from rq.job import Job   # type: ignore
    HAVE_RQ = True
except Exception:
    HAVE_RQ = False
    Redis = None            # type: ignore
    Queue = None            # type: ignore
    Job = None              # type: ignore

APP_VERSION = "1.3.2-openh264"

# ---------- Paths ----------
HOME = Path.home()
OUT_DIR = HOME / "outputs"
PROXY_DIR = HOME / "proxies"
TMP_DIR = HOME / "tmp"
for d in (OUT_DIR, PROXY_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Optional Redis / RQ setup ----------
if HAVE_RQ:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis = Redis.from_url(REDIS_URL)  # type: ignore
        queue = Queue("default", connection=redis)  # type: ignore
    except Exception:
        redis = None
        queue = None
else:
    redis = None
    queue = None

# ---- RunPod autoscale helpers (start on enqueue; stop handled elsewhere) ----
import requests  # add to requirements.txt if not present

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")           # set in Render env if you use autoscale
RUNPOD_TEMPLATE_ID = os.getenv("RUNPOD_TEMPLATE_ID")   # your RunPod template id
RP_API = "https://api.runpod.ai/v2"
RP_HEADERS = (
    {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    if RUNPOD_API_KEY else None
)

def _rp_enabled() -> bool:
    return bool(RUNPOD_API_KEY and RUNPOD_TEMPLATE_ID and RP_HEADERS)

def _rp_list_running() -> list[dict]:
    if not _rp_enabled():
        return []
    try:
        pods = requests.get(f"{RP_API}/pods", headers=RP_HEADERS, timeout=20).json().get("data", [])
        return [p for p in pods if p.get("templateId") == RUNPOD_TEMPLATE_ID and p.get("desiredStatus") == "RUNNING"]
    except Exception:
        return []

def _rp_start_one() -> None:
    if not _rp_enabled():
        return
    try:
        requests.post(
            f"{RP_API}/pods",
            headers=RP_HEADERS,
            data=json.dumps({"templateId": RUNPOD_TEMPLATE_ID}),
            timeout=30,
        )
    except Exception:
        pass  # silent fail – web API still returns job_id

# ---------- Models ----------
class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True
    mode: Optional[str] = "concat"  # "best"|"first"|"concat"
    max_duration: Optional[int] = None
    take_top_k: Optional[int] = None
    min_clip_seconds: Optional[float] = None
    max_clip_seconds: Optional[float] = None
    drop_silent: Optional[bool] = True
    drop_black: Optional[bool] = True
    with_captions: Optional[bool] = False

class ProcessRequest(BaseModel):
    session_id: Optional[str] = "session"
    mode: Optional[str] = "best"
    input_url: Optional[str] = None
    files: Optional[List[str | HttpUrl]] = None
    portrait: Optional[bool] = True
    max_duration: Optional[int] = 60
    output_prefix: Optional[str] = "editdna/outputs"

# ---------- Helpers ----------
def _safe_name(url_or_path: str) -> str:
    parsed = urlparse(url_or_path)
    base = os.path.basename(parsed.path) or "input"
    return base.replace(" ", "_")

def _run(cmd: str) -> None:
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stdout.decode(errors='ignore')}")

def _http_base() -> Optional[str]:
    host_env = os.getenv("PUBLIC_BASE")
    if host_env:
        return host_env.rstrip("/")
    return None

# ---------- Proxy pipeline (uses libopenh264) ----------
def build_proxy(input_url: str, portrait: bool = True, max_seconds: Optional[int] = None) -> Path:
    src_name = _safe_name(input_url)
    local_src = TMP_DIR / f"src_{uuid.uuid4().hex}_{src_name}"
    if str(input_url).startswith("http"):
        urlretrieve(input_url, local_src)
    else:
        _run(f'cp {shlex.quote(str(input_url))} {shlex.quote(str(local_src))}')

    if portrait:
        scale_filter = 'scale=1080:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2'
    else:
        scale_filter = 'scale=960:trunc(ow/a/2)*2:force_original_aspect_ratio=decrease,pad=960:540:(960-iw)/2:(540-ih)/2'

    limit = f"-t {int(max_seconds)}" if max_seconds and max_seconds > 0 else ""
    proxy_path = PROXY_DIR / f"proxy_{uuid.uuid4().hex}.mp4"

    # IMPORTANT: libopenh264, no -preset/-crf
    cmd = (
        f'ffmpeg -y -ss 0 -i {shlex.quote(str(local_src))} '
        f'-f lavfi -i anullsrc=channel_layout=mono:sample_rate=48000 '
        f'{limit} -map 0:v:0 -map 1:a:0 '
        f'-vf "{scale_filter},fps=24" '
        f'-c:v libopenh264 -b:v 2500k -maxrate 2500k -bufsize 5000k -pix_fmt yuv420p -g 48 '
        f'-c:a aac -ar 48000 -ac 1 -b:a 128k '
        f'-shortest -movflags +faststart {shlex.quote(str(proxy_path))}'
    )
    _run(cmd)
    return proxy_path

def pick_good_takes(proxies: List[Path], mode: str = "best", take_top_k: Optional[int] = 1) -> List[Path]:
    if not proxies:
        return []
    if mode == "concat":
        return proxies
    if mode in ("best", "first"):
        k = take_top_k or 1
        return proxies[:k]
    return proxies[:1]

def render_concat(takes: List[Path], output_prefix: str = "editdna/outputs") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{Path(output_prefix).name}_{uuid.uuid4().hex}.mp4"

    if len(takes) == 1:
        # remux only
        _run(f'ffmpeg -y -i {shlex.quote(str(takes[0]))} -c copy -movflags +faststart {shlex.quote(str(out_file))}')
        return out_file

    list_txt = TMP_DIR / f"concat_{uuid.uuid4().hex}.txt"
    with list_txt.open("w") as f:
        for p in takes:
            f.write(f"file '{p.as_posix()}'\n")
    # stream-copy concat (works if streams match). If it fails, the caller should re-encode externally.
    _run(f'ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(list_txt))} -c copy -movflags +faststart {shlex.quote(str(out_file))}')
    return out_file

# ---------- RQ helpers ----------
def _job_payload(job: "Job") -> Dict[str, Any]:  # type: ignore
    status = job.get_status(refresh=True)
    result = job.result if status == "finished" else None
    error = job.exc_info if status == "failed" else None
    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }

def _get_job_or_404(job_id: str) -> "Job":  # type: ignore
    try:
        return Job.fetch(job_id, connection=redis)  # type: ignore
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")

# ---------- Routes ----------
app = FastAPI(title="editdna", version=APP_VERSION)

@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "editdna", "version": APP_VERSION, "time": int(datetime.utcnow().timestamp())})

@app.get("/health")
def health() -> JSONResponse:
    redis_ok = False
    q_count: Optional[int] = None
    if HAVE_RQ and redis is not None and queue is not None:
        try:
            redis_ok = bool(redis.ping())  # type: ignore
        except Exception:
            redis_ok = False
        try:
            q_count = queue.count  # type: ignore
        except Exception:
            q_count = None
    return JSONResponse({
        "ok": True,
        "version": APP_VERSION,
        "queue": {"enabled": HAVE_RQ and queue is not None, "name": getattr(queue, "name", None), "pending": q_count},
    })

@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    if not (HAVE_RQ and redis is not None):
        raise HTTPException(status_code=503, detail="RQ/Redis not available")
    job = _get_job_or_404(job_id)
    return JSONResponse(_job_payload(job))

@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    if not (HAVE_RQ and queue is not None):
        raise HTTPException(status_code=503, detail="RQ/Redis not available")

    payload = req.dict()
    job = queue.enqueue(  # type: ignore
        "tasks.job_render",
        payload,
        job_timeout=60 * 60,
        result_ttl=86400,
        ttl=7200,
    )

    # autoscale: start RunPod GPU if queue has jobs and no pod running
    try:
        pending = queue.count  # type: ignore
        if pending and not _rp_list_running():
            _rp_start_one()
    except Exception:
        pass

    return JSONResponse({"job_id": job.id, "session_id": req.session_id or "session"})

@app.post("/process")
def process(req: ProcessRequest) -> JSONResponse:
    try:
        session_id = req.session_id or "session"
        files = req.files or ([req.input_url] if req.input_url else [])
        if not files:
            raise HTTPException(status_code=400, detail="Provide input_url or files[]")

        proxies = [build_proxy(str(f), portrait=bool(req.portrait), max_seconds=req.max_duration) for f in files]
        takes = pick_good_takes(proxies, mode=req.mode or "best", take_top_k=1)
        out_path = render_concat(takes, output_prefix=req.output_prefix or "editdna/outputs")
        base = _http_base()

        return JSONResponse({
            "ok": True,
            "session_id": session_id,
            "mode": req.mode or "best",
            "inputs": [str(f) for f in files],
            "output_path": out_path.as_posix(),
            "output_url_hint": (f"{base}/{out_path.relative_to(HOME).as_posix()}" if base else None),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

