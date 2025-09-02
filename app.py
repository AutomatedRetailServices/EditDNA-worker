import os
import io
import json
import uuid
import math
import shutil
import tempfile
import subprocess
from typing import List, Optional, Dict, Any

import boto3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment, silence

# OpenAI (>=1.0) SDK style
from openai import OpenAI

# -------------------------
# Config / Clients
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET env var is required")

client = OpenAI(api_key=OPENAI_API_KEY)
s3 = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI(title="Script2ClipShop Worker", version="0.1.0")


# -------------------------
# Models
# -------------------------
class ProcessResponse(BaseModel):
    job_id: str
    selected: Dict[str, List[Dict[str, Any]]]  # top picks per slot
    draft_order: List[Dict[str, Any]]          # default H->F->P->CTA picks
    final_video_s3_url: str
    metadata: Dict[str, Any]


# -------------------------
# Utils
# -------------------------
def run(cmd: List[str]) -> None:
    """Run a shell command and raise if it fails."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout.decode()}\nSTDERR:\n{proc.stderr.decode()}"
        )


def detect_segments_by_silence(video_path: str) -> List[Dict[str, float]]:
    """
    Convert to wav, then use pydub to find non-silent chunks.
    Returns list of {start, end} in seconds.
    """
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        # Extract mono 16k wav for stable silence detection
        run([
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000",
            "-vn", wav_path
        ])
        audio = AudioSegment.from_wav(wav_path)
        # Parameters tuned for talking-head video; adjust if needed
        nonsilent = silence.detect_nonsilent(
            audio,
            min_silence_len=400,    # ms
            silence_thresh=-38,     # dBFS
            seek_step=10            # ms
        )
        segments = []
        for start_ms, end_ms in nonsilent:
            # merge micro-segments shorter than ~0.6s
            if end_ms - start_ms < 600:
                continue
            segments.append({"start": start_ms / 1000.0, "end": end_ms / 1000.0})
        return segments


def transcribe_chunk(video_path: str, start: float, end: float) -> str:
    """
    Cut chunk, transcribe with Whisper.
    """
    with tempfile.TemporaryDirectory() as td:
        cut_path = os.path.join(td, "cut.mp4")
        duration = max(0.01, end - start)
        run([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-c", "copy",
            cut_path
        ])

        # Send audio stream for transcription (OpenAI supports video; we’ll extract audio)
        wav_path = os.path.join(td, "cut.wav")
        run(["ffmpeg", "-y", "-i", cut_path, "-ac", "1", "-ar", "16000", wav_path])

        with open(wav_path, "rb") as f:
            # Use whisper-1 for stable STT
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return tr.text.strip()


def classify_and_score(transcript: str, tone: str, product_link: Optional[str], features: List[str]) -> Dict[str, Any]:
    """
    Ask GPT to score likelihood that this chunk is a good Hook / Feature / Proof / CTA.
    Return JSON with fields: {scores: {Hook, Feature, Proof, CTA}, rationale, snippet}
    """
    sys = (
        "You are an editor for short-form ads. "
        "Given a transcript chunk, score how well it serves each role: Hook, Feature, Proof, CTA. "
        "Score 0-100 (higher = stronger fit). Return strict JSON."
    )
    feats = ", ".join(features) if features else "n/a"
    user_msg = f"""
Tone: {tone}
Product link (optional): {product_link or 'n/a'}
Key features to highlight: {feats}

Transcript chunk:
\"\"\"{transcript}\"\"\"

Return JSON with:
- scores: object with keys Hook, Feature, Proof, CTA (0-100)
- rationale: short reason
- snippet: <= 18 words, the best 1-line pull quote
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_msg}],
        temperature=0.2
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback
        data = {
            "scores": {"Hook": 0, "Feature": 0, "Proof": 0, "CTA": 0},
            "rationale": "Failed to parse; defaulting to zeros.",
            "snippet": transcript[:100]
        }
    # Guard rails
    for k in ["Hook", "Feature", "Proof", "CTA"]:
        v = data.get("scores", {}).get(k, 0)
        if not isinstance(v, (int, float)):
            data["scores"][k] = 0
        else:
            data["scores"][k] = max(0, min(100, int(v)))
    return data


def cut_and_concat(video_path: str, picks: List[Dict[str, float]], out_path: str) -> None:
    """
    Cut selected segments and concat into a single vertical 1080x1920@30 mp4.
    We re-encode for safe concat and normalization.
    """
    with tempfile.TemporaryDirectory() as td:
        cut_list = []
        for i, seg in enumerate(picks):
            cut_i = os.path.join(td, f"seg_{i:02d}.mp4")
            dur = max(0.01, seg["end"] - seg["start"])
            run([
                "ffmpeg", "-y",
                "-ss", f"{seg['start']:.3f}",
                "-t", f"{dur:.3f}",
                "-i", video_path,
                # Re-encode now so concat is smooth
                "-vf", "scale=-2:1920:flags=bicubic,fps=30,format=yuv420p"
                       ",crop=1080:1920:exact=1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                cut_i
            ])
            cut_list.append(cut_i)

        # Create a concat list file
        list_path = os.path.join(td, "list.txt")
        with open(list_path, "w") as f:
            for p in cut_list:
                f.write(f"file '{p}'\n")

        # Concat with re-encode to avoid stream issues
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ])


def upload_to_s3(local_path: str, key: str) -> str:
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.1.0"}


@app.post("/process", response_model=ProcessResponse)
async def process_video(
    video: UploadFile = File(..., description="Raw vertical video (mp4/mov)"),
    product_link: Optional[str] = Form(None),
    features_csv: Optional[str] = Form(None, description="Comma-separated 3 key features"),
    tone: Optional[str] = Form("Casual")
):
    """
    Main entrypoint:
    1) split by silence
    2) transcribe chunks
    3) classify+score as Hook/Feature/Proof/CTA
    4) pick top 1-3 per slot
    5) build default draft: best Hook -> best Feature -> best Proof -> best CTA (20–45s target)
    6) render & upload to S3
    """
    job_id = str(uuid.uuid4())

    # Save upload to disk
    with tempfile.TemporaryDirectory() as td:
        raw_ext = os.path.splitext(video.filename or "")[1].lower() or ".mp4"
        raw_path = os.path.join(td, f"input{raw_ext}")

        with open(raw_path, "wb") as f:
            f.write(await video.read())

        # 1) detect segments
        segments = detect_segments_by_silence(raw_path)
        if not segments:
            # fallback: single segment full video max 60s
            # (keeps things moving if silence detection fails on some files)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", raw_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                total = float(probe.stdout.decode().strip())
            except Exception:
                total = 60.0
            segments = [{"start": 0.0, "end": min(total, 60.0)}]

        # 2) transcribe + 3) classify
        feats = [x.strip() for x in (features_csv or "").split(",") if x.strip()][:3]
        scored = []
        for seg in segments:
            text = transcribe_chunk(raw_path, seg["start"], seg["end"])
            info = classify_and_score(text, tone=tone, product_link=product_link, features=feats)
            scored.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "scores": info.get("scores", {}),
                "snippet": info.get("snippet", ""),
                "rationale": info.get("rationale", "")
            })

        # 4) select top 1–3 per slot
        def top_for(slot: str, k: int = 3):
            pool = sorted(scored, key=lambda x: x["scores"].get(slot, 0), reverse=True)
            return pool[:k]

        selections = {
            "Hook": top_for("Hook", 3),
            "Feature": top_for("Feature", 3),
            "Proof": top_for("Proof", 3),
            "CTA": top_for("CTA", 3),
        }

        # 5) build default draft (best H/F/P/CTA), keep total target 20–45s
        draft = []
        for slot in ["Hook", "Feature", "Proof", "CTA"]:
            pick = selections[slot][0] if selections[slot] else None
            if pick:
                draft.append({"slot": slot, **pick})

        # duration control: trim if >45s or pad if <20s (we’ll just trust the selection for V1)
        # Optional: you could truncate long segments, but we’ll keep it simple in V1.

        # Extract segments in order for render
        picks_for_render = [{"start": d["start"], "end": d["end"]} for d in draft]

        # 6) render & upload
        out_name = f"{job_id}.mp4"
        out_local = os.path.join(td, out_name)
        cut_and_concat(raw_path, picks_for_render, out_local)

        s3_key = f"script2clipshop/{out_name}"
        final_url = upload_to_s3(out_local, s3_key)

        resp = {
            "job_id": job_id,
            "selected": selections,
            "draft_order": draft,
            "final_video_s3_url": final_url,
            "metadata": {
                "tone": tone,
                "product_link": product_link,
                "features": feats,
                "segments_detected": len(segments),
                "version": "v1-preeditor"
            }
        }
        return JSONResponse(resp)

