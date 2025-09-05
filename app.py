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

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "1.0.2-playable-export-lowram"

# -------- Helpers --------
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

def run(cmd: list):
    """Run a shell command, raise on error (captures stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

# -------- Routes --------
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
    """
    Saves uploaded videos to /tmp and stores session.json with file paths.
    Returns session_id + file list.
    """
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

    return JSONResponse({
        "ok": True,
        "
