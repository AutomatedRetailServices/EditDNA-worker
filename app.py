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

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "0.9.6-export-stable"

# -------- Helpers --------
def save_session_file(session_id: str, filename: str, data: dict):
    session_path = SESS_ROOT / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    file_path = session_path / filename
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path)

def load_session_file(session_id: str, filename: str):
    file_path = SESS_ROOT / session_id / filename
    if not file_path.exists():
        raise FileNotFoundError
    with open(file_path, "r") as f:
        return json.load(f)

# -------- Endpoints --------
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.post("/process")
async def process_video(
    videos: List[UploadFile] = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    session_id = uuid.uuid4().hex
    transcript_data = {"session_id": session_id, "files": [], "segments": []}

    for video in videos:
        file_id = uuid.uuid4().hex[:8]
        filename = video.filename
        transcript_data["files"].append({"file_id": file_id, "filename": filename})

        # placeholder transcript simulation
        transcript_data["segments"].append({
            "file_id": file_id,
            "filename": filename,
            "start": 0,
            "end": 5,
            "text": f"Simulated transcript for {filename}"
        })

    save_session_file(session_id, "transcript.json", transcript_data)

    return JSONResponse({"ok": True, "session_id": session_id, "files": transcript_data["files"]})

@app.post("/export")
def export_video(session_id: str = Form(...), filename: str = Form(...)):
    try:
        data = load_session_file(session_id, "transcript.json")
    except FileNotFoundError:
        return JSONResponse({"ok": False, "reason": "session not found"})

    output_path = SESS_ROOT / session_id / filename
    with open(output_path, "w") as f:
        f.write("Simulated video export")
    
    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": filename,
        "download": f"/download/{session_id}/{filename}",
        "segments_used": data.get("segments", [])
    })

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    file_path = SESS_ROOT / session_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

