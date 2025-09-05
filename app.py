import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI  # v1 SDK

app = FastAPI()

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "1.1.0-stitch"

# ---- Helpers ----
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
    """Run a shell command; raise on non-zero exit and return stdout."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

# ---- Routes ----
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.post("/process")
async def process_video(
    videos: List[UploadFile] = File(...),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form("")
):
    # Save uploaded files into a unique session folder and record metadata
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    file_paths = {}

    for up in videos:
        fid = uuid.uuid4().hex[:8]
        orig_name = up.filename or f"{fid}.mp4"
        dst = sd / orig_name
        with dst.open("wb") as w:
            shutil.copyfileobj(up.file, w)
        files_meta.append({"file_id": fid, "filename": orig_name})
        file_paths[fid] = str(dst)

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "file_paths": file_paths,
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
    }
    save_json(sd / "session.json", session_json)

    return JSONResponse({"ok": True, "session_id": session_id, "files": files_meta})

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    start_seconds: float = Form(0.0),
    duration_seconds: float = Form(10.0)
):
    # Trim the first uploaded video to a playable MP4 using a low-RAM strategy
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    session = load_json(meta_path)
    files = session.get("files") or []
    file_paths = session.get("file_paths") or {}
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")

    first = files[0]
    fid = first["file_id"]
    src_path = file_paths.get(fid)
    if not src_path or not Path(src_path).exists():
        raise HTTPException(status_code=404, detail="source video not found")

    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "draft.mp4"
    out_path = sd / safe_name
    start_s = float(start_seconds)
    dur_s = max(0.1, float(duration_seconds))

    # 1) Stream-copy (no re-encode) → minimal memory
    try:
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-t", f"{dur_s:.3f}",
            "-i", src_path,
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ])
    except Exception as e1:
        print("FFMPEG stream-copy failed:", e1)
        # 2) Very light re-encode (720p, 1 thread)
        try:
            run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_s:.3f}",
                "-t", f"{dur_s:.3f}",
                "-i", src_path,
                "-vf", "scale=720:-2:flags=lanczos",
                "-r", "30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-threads", "1",
                "-movflags", "+faststart",
                str(out_path)
            ])
        except Exception as e2:
            print("FFMPEG re-encode failed:", e2)
            # 3) Last resort: copy full source file so there is always a download
            shutil.copy(src_path, out_path)

    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}"
    })

@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest: str = Form(...),  # JSON string with {"segments":[...], "fps":30, "scale":1080}
    filename: str = Form("final.mp4")
):
    # Build a single video by trimming multiple segments and concatenating
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    session = load_json(meta_path)
    file_paths = session.get("file_paths") or {}

    try:
        mani = json.loads(manifest)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid manifest JSON: {e}")

    segs = mani.get("segments") or []
    if not segs:
        raise HTTPException(status_code=400, detail="manifest has no segments")

    fps = int(mani.get("fps", 30))
    scale = int(mani.get("scale", 720))  # keep 720 by default to avoid OOM

    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".",)).strip() or "final.mp4"
    out_path = sd / safe_name

    work = sd / f"stitch_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)

    list_file = work / "list.txt"
    used_segments = []
    list_lines = []

    # Generate uniformly-encoded intermediate clips (low RAM, single thread)
    for idx, s in enumerate(segs):
        fid = s.get("file_id")
        if not fid or fid not in file_paths:
            print("stitch: skipping unknown file_id", fid)
            continue

        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        duration = max(0.05, end - start)
        src = file_paths[fid]
        seg_out = work / f"seg_{idx:03d}.mp4"

        try:
            run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", src,
                "-vf", f"scale={scale}:-2:flags=lanczos",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-threads", "1",
                "-movflags", "+faststart",
                str(seg_out)
            ])
            list_lines.append(f"file '{seg_out.as_posix()}'\n")
            used_segments.append({
                "file_id": fid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            })
        except Exception as e:
            print("stitch: segment failed, skipping:", e)

    if not list_lines:
        raise HTTPException(status_code=400, detail="no valid segments to stitch")

    list_file.write_text("".join(list_lines))

    # Try concat with stream copy first
    try:
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ])
    except Exception as e1:
        print("stitch: concat copy failed, re-encoding:", e1)
        # Fallback: re-encode the concatenation
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", f"scale={scale}:-2:flags=lanczos",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k",
            "-threads", "1",
            "-movflags", "+faststart",
            str(out_path)
        ])

    return JSONResponse({
        "ok": True,
        "message": "stitch complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}",
        "segments_used": used_segments
    })

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
