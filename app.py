from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os, tempfile, subprocess, json, re
from openai import OpenAI

app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY env var)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -----------------------
# Utility helpers
# -----------------------

def run(cmd):
    """
    Run a shell command and raise with stderr on failure.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")

def extract_duration_seconds(audio_path: str) -> float:
    """
    Read total duration (seconds) using ffprobe.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0

# -----------------------
# Speech segmentation + transcription
# -----------------------

def segment_by_silence_and_transcribe(video_bytes: bytes, filename: str):
    """
    1) Save upload to temp
    2) Extract mono 16k WAV
    3) Detect silences with ffmpeg silencedetect
    4) Cut spoken segments, transcribe each with Whisper
    5) Return [{"start","end","text"}, ...] plus full transcript
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name

    segments = []
    try:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "audio.wav")
            # Convert to 16k mono WAV for stable VAD/STT
            run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", wav])

            # Detect silences (tweak noise/d for your content)
            # Logs lines like: silence_start: 3.210 / silence_end: 5.870
            proc = subprocess.run(
                ["ffmpeg", "-i", wav, "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            log = proc.stderr or ""

            # Build start/end lists from silencedetect output
            starts = [0.0]
            ends = []
            for m in re.finditer(r"silence_start:\s*([0-9.]+)", log):
                try:
                    ends.append(float(m.group(1)))
                except:
                    pass
            for m in re.finditer(r"silence_end:\s*([0-9.]+)", log):
                try:
                    starts.append(float(m.group(1)))
                except:
                    pass

            # Ensure last end == total duration
            total = extract_duration_seconds(wav)
            if not ends or (ends and ends[-1] < total):
                ends.append(total)

            # Cut & transcribe each segment between silence regions
            for s, e in zip(starts, ends):
                # skip ultra-short blips
                if (e - s) < 0.25:
                    continue
                chunk = os.path.join(td, f"seg_{int(s*1000)}.wav")
                run([
                    "ffmpeg", "-y",
                    "-ss", f"{s:.3f}",
                    "-t", f"{(e - s):.3f}",
                    "-i", wav,
                    "-ac", "1", "-ar", "16000",
                    chunk
                ])
                with open(chunk, "rb") as f:
                    tr = client.audio.transcriptions.create(model="whisper-1", file=f)
                text = (getattr(tr, "text", "") or "").strip()
                if text:
                    segments.append({
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "text": text
                    })

            # Join transcript
            transcript = " ".join(seg["text"] for seg in segments)
            return segments, transcript
    finally:
        try:
            os.remove(in_path)
        except:
            pass

# -----------------------
# Classification (timestamp-aware)
# -----------------------

def classify_buckets(client, transcript: str, segments: list, features_csv: str, tone: str):
    """
    Ask GPT to extract ONLY verbatim lines from transcript, sorted into:
    - hook (string)
    - feature_lines (list)
    - proof_lines (list)
    - cta_lines (list)

    We provide segments with timestamps and instruct it to PREFER hook lines that
    occur in the first ~0–15 seconds if available.
    """
    # Build a compact segment listing for the prompt
    # (Keep it short to avoid token bloat)
    seg_listing = []
    for seg in segments[:80]:  # cap to first 80 segments for safety
        seg_listing.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
    seg_text = "\n".join(seg_listing)

    system = (
        "You are an ad pre-editor for TikTok Shop/UGC. "
        "From the given transcript and timestamped segments, extract ONLY lines that already exist "
        "(no rewriting). Return STRICT JSON with keys: "
        "hook (string), feature_lines (array of strings), proof_lines (array of strings), "
        "cta_lines (array of strings). "
        "Prefer hooks that appear in the first 0–15 seconds when possible. "
        "Keep each list to at most 3 short items."
    )
    user = (
        f"Tone: {tone}\n"
        f"Key features to prioritize: {features_csv}\n\n"
        f"Transcript (full):\n{transcript}\n\n"
        f"Segments (timestamped, earliest first):\n{seg_text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
        max_tokens=650,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except:
        data = {"hook": "", "feature_lines": [], "proof_lines": [], "cta_lines": []}

    # Normalize keys & cap lengths
    return {
        "hook": data.get("hook", "") or "",
        "feature_lines": (data.get("feature_lines", []) or [])[:3],
        "proof_lines": (data.get("proof_lines", []) or [])[:3],
        "cta_lines": (data.get("cta_lines", []) or [])[:3],
    }

# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.2.0"}

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # 1) Read uploaded bytes
    data = await video.read()
    await video.seek(0)

    # 2) Segment + transcribe (with timestamps)
    try:
        segments, transcript = segment_by_silence_and_transcribe(
            data, video.filename or "upload.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Segmentation/Transcription error: {e}")

    # 3) Classify into buckets (timestamp-aware prompt prefers early hooks)
    buckets = classify_buckets(client, transcript, segments, features_csv, tone)

    # 4) Return everything
    return JSONResponse({
        "ok": True,
        "bytes": len(data),
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "transcript": transcript,                # full text
        "transcript_chars": len(transcript),
        "segments": segments,                    # [{start,end,text}, ...]
        "buckets": buckets                       # {hook, feature_lines[], proof_lines[], cta_lines[]}
    })
