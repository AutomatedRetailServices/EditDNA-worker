from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os, tempfile, subprocess, json
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- helper functions ---

def run(cmd):
    """Run a shell command, raise if it fails."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr.decode()}")

def transcribe_in_chunks(client, video_bytes: bytes, filename: str) -> str:
    """Split long audio into ~60s chunks and transcribe with Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name

    try:
        with tempfile.TemporaryDirectory() as td:
            # Convert to 16k mono wav for stable STT
            wav_path = os.path.join(td, "audio.wav")
            run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", wav_path])

            # Get total duration
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                total = float(probe.stdout.decode().strip())
            except:
                total = 0.0
            if total <= 0.0:
                raise HTTPException(status_code=400, detail="Could not read audio duration")

            # Cut into ~60s chunks, transcribe each, join
            CHUNK = 60.0
            cur = 0.0
            pieces = []
            while cur < total:
                dur = min(CHUNK, total - cur)
                cut_wav = os.path.join(td, f"chunk_{int(cur)}.wav")
                run([
                    "ffmpeg", "-y",
                    "-ss", f"{cur:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", wav_path,
                    "-ac", "1", "-ar", "16000",
                    cut_wav
                ])
                with open(cut_wav, "rb") as f:
                    tr = client.audio.transcriptions.create(model="whisper-1", file=f)
                pieces.append(tr.text.strip() if hasattr(tr, "text") else "")
                cur += CHUNK

            return " ".join(p for p in pieces if p).strip()
    finally:
        try: os.remove(in_path)
        except: pass

def classify_transcript(client, transcript: str, features_csv: str, tone: str):
    """
    Returns a JSON object with hook, feature_lines, proof_lines, cta_lines,
    all verbatim (no rewriting), 1–3 items per bucket.
    """
    system = (
        "You are an ad pre-editor. From the transcript, extract ONLY lines that already exist "
        "in the transcript (no rewriting). Return a JSON object with keys: "
        "hook (string), feature_lines (array of strings), proof_lines (array of strings), "
        "cta_lines (array of strings). Keep 1–3 short lines per list. Strict JSON."
    )
    user = (
        f"Tone: {tone}\n"
        f"Key features to prioritize: {features_csv}\n\n"
        f"Transcript:\n{transcript}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
        max_tokens=500,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"hook": "", "feature_lines": [], "proof_lines": [], "cta_lines": []}
    return {
        "hook": data.get("hook", ""),
        "feature_lines": data.get("feature_lines", [])[:3],
        "proof_lines": data.get("proof_lines", [])[:3],
        "cta_lines": data.get("cta_lines", [])[:3],
    }

# --- API endpoint ---

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # 1) Read uploaded file
    data = await video.read()
    await video.seek(0)

    # 2) Transcribe in chunks
    try:
        transcript = transcribe_in_chunks(client, data, video.filename or "upload.mp4")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription error: {e}")

    # 3) Classify transcript into buckets
    buckets = classify_transcript(client, transcript, features_csv, tone)

    # 4) Return everything
    return JSONResponse({
        "ok": True,
        "bytes": len(data),
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "transcript": transcript,
        "transcript_chars": len(transcript),
        "buckets": buckets
    })
