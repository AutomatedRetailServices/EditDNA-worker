import os
import uuid
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI

# init
app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# store sessions on disk (/tmp so Render survives restarts)
SESSIONS_DIR = "/tmp/sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

@app.get("/health")
async def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "1.0.0"}

@app.post("/process")
async def process_video(
    videos: list[UploadFile],
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    session_id = uuid.uuid4().hex
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    # Dummy segmentation logic (replace later with Whisper/LLM pipeline)
    segments = []
    for v in videos:
        text = f"Dummy transcript for {v.filename} with tone={tone} and product={product_link}"
        segments.append({
            "file_id": uuid.uuid4().hex,
            "filename": v.filename,
            "text": text,
            "start": 0,
            "end": 10
        })

    # Save session
    data = {"session_id": session_id, "segments": segments}
    with open(session_path, "w") as f:
        json.dump(data, f)

    return {"ok": True, "session_id": session_id, "segments": segments}

@app.post("/export")
async def export_video(session_id: str = Form(...), filename: str = Form("draft.mp4")):
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        return JSONResponse(status_code=404, content={"ok": False, "reason": "session not found"})

    with open(session_path, "r") as f:
        session = json.load(f)

    # Dummy export — we just save transcript into a .txt
    export_path = os.path.join(SESSIONS_DIR, filename.replace(".mp4", ".txt"))
    with open(export_path, "w") as f:
        for s in session["segments"]:
            f.write(s["text"] + "\n")

    return {
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": filename,
        "download": f"/download/{session_id}/{filename}"
    }

@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    path = os.path.join(SESSIONS_DIR, filename.replace(".mp4", ".txt"))
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    return JSONResponse(status_code=404, content={"ok": False, "reason": "file not found"})
