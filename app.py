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

# New for analyze/transcribe
from bs4 import BeautifulSoup
from openai import OpenAI

# ===================== App setup =====================
VERSION = "1.4.0-full-scope"
app = FastAPI(title="EditDNA Web API", version=VERSION)

# CORS (configure CORS_ORIGINS in Render if needed)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Working directory (per-container ephemeral disk)
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# Redis & RQ (keep default: bytes, not decode_responses)
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not set")
_redis = redis.from_url(REDIS_URL)

def get_q() -> Queue:
    return Queue("default", connection=_redis)

# Optional: If you later upload outputs to S3/CloudFront, return a public URL
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")

# OpenAI client (used by /analyze and /transcribe_urls)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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
    """Run ffmpeg and raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def _download_to_tmp(url: str, dst: Path):
    # Stream download to file (keeps RAM low)
    with requests.get(url, stream=True) as res:
        res.raise_for_status()
        with dst.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _safe_name(name: str) -> str:
    name = (name or "").strip() or "final.mp4"
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", "."))

def _public_or_download(session_id: str, fname: str) -> Dict[str, str]:
    if S3_PUBLIC_BASE:
        return {"public_url": f"{S3_PUBLIC_BASE}/sessions/{session_id}/{fname}"}
    return {"download_path": f"/download/{session_id}/{fname}"}

def _scrape_product_text(url: str, max_chars: int = 6000) -> str:
    """Lightweight scrape of titles/headers/descriptions/reviews."""
    try:
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        bits = []
        for sel in ["title", "h1", "h2", "p", "[class*=description]", "[id*=description]", "[class*=review]"]:
            for node in soup.select(sel):
                t = node.get_text(separator=" ", strip=True)
                if t:
                    bits.append(t)
                if sum(len(x) for x in bits) > max_chars:
                    break
            if sum(len(x) for x in bits) > max_chars:
                break
        text = " ".join(bits)
        return " ".join(text.split())[:max_chars]
    except Exception as e:
        return f"(scrape_failed: {e})"


# ===================== Core stitch (two forms) =====================
def _stitch_do(urls: Dict[str, str], session_id: str, manifest: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Downloads/caches sources, trims each requested segment with uniform fps/scale,
    concatenates into a final mp4, and returns a local download path or public URL.
    """
    sd = sess_dir(session_id)
    work = sd / "work"
    work.mkdir(parents=True, exist_ok=True)

    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 1080))
    segments = manifest.get("segments", [])
    if not segments:
        raise RuntimeError("manifest has no segments")
    if not urls:
        raise RuntimeError("no URLs provided for this job")

    parts: List[Path] = []
    for idx, seg in enumerate(segments):
        fid = seg["file_id"]
        src_url = urls.get(fid)
        if not src_url:
            raise RuntimeError(f"missing url for file_id {fid}")

        # cache original once per file_id
        cache_vid = work / f"src_{fid}.cache"
        if not cache_vid.exists():
            _download_to_tmp(src_url, cache_vid)

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

def stitch_core_from_session(session: Dict[str, Any], manifest: Dict[str, Any], filename: str = "final.mp4") -> Dict[str, Any]:
    """Worker job: receives ENTIRE session (including urls), so it doesn't need local files."""
    session_id = session.get("session_id") or uuid.uuid4().hex
    urls: Dict[str, str] = session.get("urls") or {}
    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)

def stitch_core_from_disk(session_id: str, manifest: Dict[str, Any], filename: str = "final.mp4") -> Dict[str, Any]:
    """Sync path on web dyno (tiny tests): reads session.json from THIS container."""
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise RuntimeError("session not found")
    session = load_json(meta_path)
    urls: Dict[str, str] = session.get("urls") or {}
    return _stitch_do(urls=urls, session_id=session_id, manifest=manifest, filename=filename)


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

# Analyze / Slots / Transcribe
class AnalyzeIn(BaseModel):
    product_link: str = ""
    tone: str = "casual"
    style: str = "Talking Head"  # Talking Head | Skit | Testimonial | Voiceover
    features_csv: str = ""

class SlotsIn(BaseModel):
    script_text: str
    style: str = "Talking Head"

class TranscribeIn(BaseModel):
    urls: List[str]


# ===================== Routes =====================
@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-web", "version": VERSION, "redis": True}

@app.post("/process_urls")
def process_urls(body: ProcessURLsIn):
    """Register external/presigned video URLs for this session."""
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
    """Minimal draft: spread max_total_sec across files (2–8s per file)."""
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

@app.post("/stitch")
def stitch(body: StitchIn):
    """Synchronous stitch on web dyno (use for tiny proofs only)."""
    try:
        return stitch_core_from_disk(body.session_id, body.manifest, body.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stitch_async")
def stitch_async(body: StitchAsyncIn):
    """
    Enqueue background stitch with the WHOLE session payload so the worker
    doesn't need files from web dyno's /tmp.
    """
    sd = sess_dir(body.session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)

    q = get_q()
    job = q.enqueue(stitch_core_from_session, session, body.manifest, body.filename)
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

# ---- Analyze / Slots / Transcribe ----
@app.post("/analyze")
def analyze(body: AnalyzeIn):
    """Generate a funnel script + basic slot mapping from product page + features."""
    if not client:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")

    scraped = _scrape_product_text(body.product_link) if body.product_link else ""
    features = [x.strip() for x in (body.features_csv or "").split(",") if x.strip()]

    sys = (
        "You are a direct-response ad copywriter for short UGC videos. "
        "Write concise lines. Target 70–120 words total. "
        "Structure strictly as Hook → Feature → Proof → CTA. "
        "Return clean text only (no hashtags)."
    )
    user = (
        f"STYLE: {body.style}\n"
        f"TONE: {body.tone}\n"
        f"FEATURES: {features}\n"
        f"SCRAPED_TEXT: {scraped[:4000]}\n\n"
        "Write a conversion-first script. Then list 3 alternative Hooks and 2 CTAs."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
        )
        script = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai_error: {e}")

    # Quick slot mapping
    try:
        map_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Split the given script into JSON slots: {hook, feature, proof, cta}. Keep values short."},
                {"role": "user", "content": script},
            ],
        )
        slots_raw = map_resp.choices[0].message.content
        try:
            import json as _json
            slots = _json.loads(slots_raw)
        except Exception:
            slots = {"hook": "", "feature": "", "proof": "", "cta": "", "raw": slots_raw}
    except Exception as e:
        slots = {"hook": "", "feature": "", "proof": "", "cta": "", "error": f"slot_map_error: {e}"}

    return {
        "ok": True,
        "style": body.style,
        "tone": body.tone,
        "script": script,
        "slots": slots,
        "product_link": body.product_link,
        "features": features,
    }

@app.post("/slots/build")
def slots_build(body: SlotsIn):
    """Map any script text into storyboard slots."""
    if not body.script_text.strip():
        raise HTTPException(status_code=400, detail="script_text is empty")

    if not client:
        txt = body.script_text.strip()
        return {
            "ok": True,
            "style": body.style,
            "slots": {
                "hook": txt.split(".")[0] if "." in txt else txt[:120],
                "feature": "",
                "proof": "",
                "cta": "",
                "raw": txt,
                "note": "OPENAI_API_KEY missing; returned naive split"
            }
        }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Split the given script into JSON: {hook, feature, proof, cta}. Keep concise."},
                {"role": "user", "content": body.script_text},
            ],
        )
        text = resp.choices[0].message.content
        import json as _json
        slots = _json.loads(text)
    except Exception as e:
        slots = {"hook": "", "feature": "", "proof": "", "cta": "", "error": f"slot_map_error: {e}", "raw": body.script_text}

    return {"ok": True, "style": body.style, "slots": slots}

@app.post("/transcribe_urls")
def transcribe_urls(body: TranscribeIn):
    """Quick Whisper transcription for a few short clips by URL."""
    if not client:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    if not body.urls:
        raise HTTPException(status_code=400, detail="no urls provided")

    results = []
    tmp_root = Path("/tmp/transcribe")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for i, u in enumerate(body.urls[:5]):  # small cap for MVP
        dst = tmp_root / f"clip_{i:03d}"
        try:
            _download_to_tmp(u, dst)
            real = dst.with_suffix(".mp4")
            dst.rename(real)

            with real.open("rb") as fh:
                tr = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=fh,
                    response_format="json",
                    temperature=0.0,
                )
            text = tr.text if hasattr(tr, "text") else tr.get("text")
            results.append({"url": u, "text": text})
        except Exception as e:
            results.append({"url": u, "error": f"transcription_failed: {e}"})

    return {"ok": True, "items": results}

# ---- Local download (fallback if you didn't upload final.mp4 to S3) ----
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
