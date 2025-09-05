import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import shutil
import uuid

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Setup OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---- Health Check ----
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "script2clipshop-worker",
        "version": "1.0.0"
    }

# ---- Process Endpoint ----
@app.post("/process")
async def process_video(
    videos: list[UploadFile],
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    session_id = uuid.uuid4().hex
    transcript_segments = []

    for video in videos:
        temp_filename = f"/tmp/{uuid.uuid4().hex}_{video.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Example: call OpenAI Whisper for transcription
        with open(temp_filename, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        transcript_segments.append(transcript.text)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "transcript": transcript_segments
    })

# ---- Export Endpoint ----
@app.post("/export")
async def export(session_id: str = Form(...), filename: str = Form("draft.mp4")):
    # Dummy export (replace with your actual merge/edit logic)
    return {
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": filename,
        "download": f"/download/{session_id}/{filename}",
        "segments_used": [
            {"text": "Example segment from transcript"}
        ]
    }
