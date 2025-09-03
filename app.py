from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # 1) read bytes
    data = await video.read()
    await video.seek(0)

    # 2) save to a temp file so Whisper can read it
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temp file: {e}")

    # 3) transcribe with Whisper (OpenAI)
    try:
        with open(tmp_path, "rb") as f:
            # whisper-1 is the speech-to-text model
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json"
            )
        transcript = resp.text if hasattr(resp, "text") else resp.get("text", "")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription error: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 4) return minimal success payload
    return JSONResponse({
        "ok": True,
        "bytes": len(data),
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "transcript": transcript[:500]  # first 500 chars so response is small
    })
