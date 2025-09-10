import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# ---------- App & Config ----------
app = FastAPI()
VERSION = "1.2.0-batch-process"
SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
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

def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def safe_filename(name: str, default: str) -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or default

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.post("/process")
async def process_video(
    videos: Optional[List[UploadFile]] = File(None, description="One or more files per request"),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form(""),
    # NEW: allow batching into the same session
    session_id: Optional[str] = Form(None)
):
    """
    Batch-friendly upload.
    - If session_id is omitted: a new session is created.
    - If session_id is provided and exists: files are APPENDED to that session.
    - You can call this endpoint multiple times with the same session_id to upload in chunks.
    """
    if session_id:
        # Reuse existing
        sd = sess_dir(session_id)
        meta_path = sd / "session.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="session not found for provided session_id")
        session = load_json(meta_path)
    else:
        # New session
        session_id = uuid.uuid4().hex
        sd = sess_dir(session_id)
        session = {
            "session_id": session_id,
            "files": [],
            "file_paths": {},
            "tone": tone,
            "features_csv": features_csv,
            "product_link": product_link,
        }

    files_meta: List[Dict[str, Any]] = session.get("files", [])
    file_paths: Dict[str, str] = session.get("file_paths", {})

    appended: List[Dict[str, Any]] = []
    if videos:
        uploads_dir = sd / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        for up in videos:
            fid = uuid.uuid4().hex[:8]
            orig_name = up.filename or f"{fid}.mp4"
            dst = uploads_dir / orig_name
            with dst.open("wb") as w:
                shutil.copyfileobj(up.file, w)
            file_rec = {"file_id": fid, "filename": orig_name}
            files_meta.append(file_rec)
            file_paths[fid] = str(dst)
            appended.append(file_rec)

    # persist
    session["files"] = files_meta
    session["file_paths"] = file_paths
    # keep tone/features/product_link only if originally blank (don't overwrite on later batches)
    session.setdefault("tone", tone)
    session.setdefault("features_csv", features_csv)
    session.setdefault("product_link", product_link)
    save_json(sd / "session.json", session)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "added_files": appended,
        "total_files_in_session": len(files_meta)
    })

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    start_seconds: float = Form(0.0),
    duration_seconds: float = Form(10.0),
    fps: int = Form(30),
    scale: int = Form(1080)
):
    """
    Quick trim/export of the FIRST uploaded file in the session.
    """
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    files = session.get("files") or []
    file_paths = session.get("file_paths") or {}
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")
    first = files[0]
    src_path = file_paths.get(first["file_id"])
    if not src_path or not Path(src_path).exists():
        raise HTTPException(status_code=404, detail="source video not found")

    out_name = safe_filename(filename, "draft.mp4")
    out_path = sd / out_name

    try:
        run([
            "ffmpeg","-y",
            "-ss", f"{float(start_seconds):.3f}",
            "-t",  f"{max(0.1, float(duration_seconds)):.3f}",
            "-i",  src_path,
            "-vf", f"scale={scale}:-2:flags=lanczos",
            "-r",  str(int(fps)),
            "-c:v","libx264","-preset","veryfast","-crf","22",
            "-c:a","aac","-b:a","128k",
            str(out_path)
        ])
    except Exception:
        shutil.copy(src_path, out_path)

    return {
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": out_name,
        "download": f"/download/{session_id}/{out_name}"
    }

# ---- Very simple "analyze" placeholder (splits each file into short segments) ----
def impl_analyze(session_id: str, max_segments_per_file: int = 8) -> dict:
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    file_paths = session.get("file_paths", {})
    analysis = {"session_id": session_id, "files": []}
    # naive: make up to N 1.5s segments starting at 0,1.7,3.4,... (just to demo pipeline)
    for rec in session.get("files", []):
        fid = rec["file_id"]
        if fid not in file_paths:
            continue
        segs = []
        start = 0.0
        for i in range(max_segments_per_file):
            segs.append({"start": round(start, 3), "end": round(start + 1.5, 3)})
            start += 1.7
        analysis["files"].append({"file_id": fid, "segments": segs})
    save_json(sd / "analysis.json", analysis)
    return analysis

@app.post("/analyze")
def analyze_endpoint(
    session_id: str = Form(...),
    max_segments_per_file: int = Form(8)
):
    analysis = impl_analyze(session_id, max_segments_per_file=max_segments_per_file)
    return {"ok": True, "analysis": analysis}

# ---- Simple classifier placeholder (assigns segments round-robin to H/F/P/CTA) ----
SLOTS = ["Hook","Feature","Proof","CTA"]

def impl_classify(session_id: str, tone: str = "casual") -> dict:
    sd = sess_dir(session_id)
    try:
        analysis = load_json(sd / "analysis.json")
    except FileNotFoundError:
        analysis = impl_analyze(session_id)
    buckets: Dict[str, List[dict]] = {k: [] for k in SLOTS}
    idx = 0
    for f in analysis.get("files", []):
        fid = f["file_id"]
        for seg in f.get("segments", []):
            buckets[SLOTS[idx % len(SLOTS)]].append({"file_id": fid, **seg})
            idx += 1
    result = {"session_id": session_id, "tone": tone, "buckets": buckets}
    save_json(sd / "classify.json", result)
    return result

@app.post("/classify")
def classify_endpoint(
    session_id: str = Form(...),
    tone: str = Form("casual")
):
    result = impl_classify(session_id, tone=tone)
    return {"ok": True, **result}

# ---- Stitch using a manifest JSON string ----
def impl_stitch(session_id: str, manifest: dict, filename: str) -> dict:
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    file_paths = session.get("file_paths", {})
    segs = manifest.get("segments") or []
    if not segs:
        raise HTTPException(status_code=400, detail="no valid segments to stitch")

    out_name = safe_filename(filename, "final.mp4")
    out_path = sd / out_name

    # Build ffmpeg filter complex
    inputs: List[str] = []
    filter_parts: List[str] = []
    concat_labels: List[str] = []
    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 720))

    for i, s in enumerate(segs):
        fid = s.get("file_id")
        if not fid or fid not in file_paths:
            print(f"stitch: skipping unknown file_id {fid}")
            continue
        src = file_paths[fid]
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start + 1.0))
        dur = max(0.05, end - start)
        inputs += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src]
        # label chain for each clip
        filter_parts.append(
            f"[{i}:v]scale={scale}:-2:flags=lanczos,fps={fps}[v{i}];"
            f"[{i}:a]anull[a{i}]"
        )
        concat_labels.append(f"[v{i}][a{i}]")

    if not concat_labels:
        raise HTTPException(status_code=400, detail="no valid segments to stitch")

    n = len(concat_labels)
    filter_str = "".join(filter_parts) + "".join(concat_labels) + f"concat=n={n}:v=1:a=1[v][a]"

    cmd = ["ffmpeg","-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map","[v]","-map","[a]",
        "-c:v","libx264","-preset","veryfast","-crf","22",
        "-c:a","aac","-b:a","128k",
        str(out_path)
    ]
    run(cmd)

    return {
        "ok": True,
        "message": "stitch complete",
        "session_id": session_id,
        "filename": out_name,
        "download": f"/download/{session_id}/{out_name}",
        "segments_used": segs
    }

@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest: str = Form(...),
    filename: str = Form("final.mp4")
):
    try:
        mani = json.loads(manifest)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid manifest JSON: {e}")
    out = impl_stitch(session_id, mani, filename)
    return out

# ---- Auto-manifest: pick first few short segments up to max_total_sec ----
@app.post("/automanifest")
def automanifest(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(12.0),
    max_segments_per_file: int = Form(8),
):
    sd = sess_dir(session_id)
    try:
        analysis = load_json(sd / "analysis.json")
    except FileNotFoundError:
        analysis = impl_analyze(session_id, max_segments_per_file=max_segments_per_file)

    total = 0.0
    segs: List[dict] = []
    for f in analysis.get("files", []):
        for s in f.get("segments", []):
            if total >= max_total_sec:
                break
            start = float(s["start"])
            end = float(s["end"])
            dur = max(0.05, end - start)
            if total + dur > max_total_sec:
                end = start + (max_total_sec - total)
                dur = end - start
            segs.append({"file_id": f["file_id"], "start": round(start, 3), "end": round(end, 3)})
            total += dur
        if total >= max_total_sec:
            break

    mani = {"segments": segs, "fps": fps, "scale": scale}
    save_json(sd / "automanifest.json", mani)
    return {"ok": True, "session_id": session_id, "filename": filename, "manifest": mani}

@app.post("/autoassemble")
def autoassemble(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    tone: str = Form("casual"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(12.0),
    max_segments_per_file: int = Form(8)
):
    # analyze -> classify (placeholder) -> automanifest -> stitch
    impl_analyze(session_id, max_segments_per_file=max_segments_per_file)
    impl_classify(session_id, tone=tone)
    mani_resp = automanifest(session_id, filename, fps, scale, max_total_sec, max_segments_per_file)
    mani = mani_resp["manifest"]
    out = impl_stitch(session_id, mani, filename)
    return out

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
