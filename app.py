import os
import uuid
import json
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from openai import OpenAI  # v1 SDK
from bs4 import BeautifulSoup
import requests

app = FastAPI()

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)  # no proxies arg

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "0.9.6-export-stable"

# ---------- Helpers ----------
def sess_path(session_id: str) -> Path:
    p = SESS_ROOT / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())

def scrape_title(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        if soup.title and soup.title.text:
            return soup.title.text.strip()
    except Exception:
        pass
    return None

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form("")
):
    """
    Accepts 1+ video files and returns:
      - session_id
      - dumb-but-useful 'segments' (we don't run heavy ML here)
      - buckets {hooks/features/proof/cta} derived from transcript text
    Everything is saved under /tmp so /export works later.
    """
    session_id = uuid.uuid4().hex
    sp = sess_path(session_id)

    # store incoming files
    file_meta = []
    for vf in videos:
        fid = uuid.uuid4().hex[:8]
        fname = vf.filename or f"{fid}.mp4"
        out = sp / fname
        with out.open("wb") as w:
            shutil.copyfileobj(vf.file, w)
        file_meta.append({"file_id": fid, "filename": fname})

    # naive "transcript": just filenames + stub text to keep your current flow working
    # (You were already verifying that export works—this keeps behavior predictable.)
    transcript_texts = []
    for m in file_meta:
        # lightweight placeholder per file
        transcript_texts.append(
            f"{m['filename']} uploaded. (stub transcript for demo; real ASR can be added later)"
        )

    # toy segmentation: first 10s "hook", rest "feature/cta/proof" placeholders
    # (keeps your Postman expectations consistent)
    segments = []
    t = 0.0
    for m in file_meta:
        segments.append({
            "start": 0.0,
            "end": 10.0,
            "text": f"Hook from {m['filename']} (example).",
            "file_id": m["file_id"],
            "filename": m["filename"],
        })
        segments.append({
            "start": 10.0,
            "end": 60.0,
            "text": f"Feature/Proof/CTA from {m['filename']} (example).",
            "file_id": m["file_id"],
            "filename": m["filename"],
        })

    buckets = {
        "hooks": {
            "intro_hooks": [
                {
                    "text": s["text"],
                    "start": s["start"],
                    "end": s["end"],
                    "score": 0.8,
                    "is_early": True,
                }
                for s in segments if "Hook" in s["text"]
            ],
            "in_body_hooks": []
        },
        "features": [
            {
                "text": s["text"],
                "start": s["start"],
                "end": s["end"]
            }
            for s in segments if "Feature" in s["text"]
        ],
        "proof": [
            {
                "text": s["text"],
                "start": s["start"],
                "end": s["end"]
            }
            for s in segments if "Proof" in s["text"]
        ],
        "cta": [
            {
                "text": s["text"],
                "start": s["start"],
                "end": s["end"]
            }
            for s in segments if "CTA" in s["text"]
        ]
    }

    # Optional: enrich with product page title
    product_title = scrape_title(product_link) if product_link else None

    # persist session JSON
    session_json = {
        "session_id": session_id,
        "files": file_meta,
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "product_title": product_title,
        "segments": segments,
        "buckets": buckets,
        "transcript_chars": sum(len(t) for t in transcript_texts),
    }
    save_json(sp / "session.json", session_json)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "files": file_meta,
        "transcript_chars": session_json["transcript_chars"],
        "segments": segments,
        "buckets": buckets
    })

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4")
):
    """
    For now, this just verifies the session exists and returns a download URL.
    (You can later stitch clips with ffmpeg here.)
    """
    sp = sess_path(session_id)
    meta_path = sp / "session.json"
    if not meta_path.exists():
        return JSONResponse({"ok": False, "reason": "session not found"})

    # Touch an empty file to simulate a rendered export
    out_file = sp / filename
    out_file.write_bytes(b"")  # placeholder artifact

    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": filename,
        "download": f"/download/{session_id}/{filename}",
        "segments_used": (load_json(meta_path).get("segments") or [])[:1]
    })

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sp = sess_path(session_id)
    f = sp / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    # Return the (placeholder) file
    return FileResponse(path=f, filename=filename, media_type="video/mp4")
