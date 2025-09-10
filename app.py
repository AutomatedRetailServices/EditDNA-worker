import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# -------- Config --------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "editdna-uploads")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("WARNING: Missing AWS credentials")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

app = FastAPI()

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "2.0.0-s3-upload"

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": VERSION}

@app.post("/process_urls")
def process_urls(
    urls: List[str] = Form(...),
    tone: str = Form("casual"),
    product_link: str = Form(""),
):
    """
    Register S3 file URLs (client uploads directly to S3).
    Creates a session.json with file list and metadata.
    """
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    for u in urls:
        fid = uuid.uuid4().hex[:8]
        files_meta.append({"file_id": fid, "url": u})

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "tone": tone,
        "product_link": product_link,
    }
    save_json(sd / "session.json", session_json)

    return JSONResponse({"ok": True, "session_id": session_id, "files": files_meta})

@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest_json: str = Form(...),
    filename: str = Form("final.mp4"),
):
    """
    Downloads files from S3, stitches them with ffmpeg, re-uploads to S3.
    """
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    session = load_json(meta_path)
    files = session.get("files") or []
    manifest = json.loads(manifest_json)

    urls_by_id = {f["file_id"]: f["url"] for f in files}

    # Download locally
    local_inputs = []
    for slot in manifest:
        fid = slot["file_id"]
        if fid not in urls_by_id:
            print(f"stitch: skipping unknown file_id {fid}")
            continue
        url = urls_by_id[fid]
        local_path = sd / f"{fid}.mp4"
        s3.download_file(S3_BUCKET, url.split("/")[-1], str(local_path))
        local_inputs.append(str(local_path))

    if not local_inputs:
        raise HTTPException(status_code=400, detail="no valid inputs to stitch")

    out_path = sd / filename

    # ffmpeg concat
    with open(sd / "inputs.txt", "w") as f:
        for lp in local_inputs:
            f.write(f"file '{lp}'\n")

    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(sd / "inputs.txt"),
        "-vf", "scale=1080:-2:flags=lanczos",
        "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path)
    ])

    # Upload stitched file to S3
    key = f"{session_id}/{filename}"
    s3.upload_file(str(out_path), S3_BUCKET, key)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "filename": filename,
        "s3_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    })
