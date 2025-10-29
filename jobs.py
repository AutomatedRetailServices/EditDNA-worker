import os
import json
import tempfile
import subprocess
import uuid
import boto3
import whisper
import torch
from moviepy.editor import VideoFileClip

# ---------- ENV CONFIG ----------
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_DOWNLOAD_ROOT = os.getenv("ASR_DOWNLOAD_ROOT", "/workspace/.cache/whisper")

# ---------- UTILS ----------
def _run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def _upload_to_s3(local_path, s3_key):
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    url = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"[s3] uploaded: {url}")
    return url

# ---------- ASR (Whisper) ----------
def _do_whisper_asr(local_path: str):
    """
    Run Whisper ASR and return segment dicts:
    [{start, end, text}, ...]
    Uses local cached model to avoid network timeouts.
    """
    import os, json, whisper

    ASR_MODEL   = os.getenv("ASR_MODEL_SIZE", "tiny")
    ASR_DEVICE  = os.getenv("ASR_DEVICE", "cuda")  # "cuda" or "cpu"
    ASR_ROOT    = os.getenv("ASR_DOWNLOAD_ROOT", "/workspace/.cache/whisper")

    os.makedirs(ASR_ROOT, exist_ok=True)
    print(f"[asr] loading whisper model='{ASR_MODEL}' device='{ASR_DEVICE}' cache='{ASR_ROOT}'")

    # Always use pre-downloaded cache
    model = whisper.load_model(ASR_MODEL, device=ASR_DEVICE, download_root=ASR_ROOT)

    result = model.transcribe(local_path, fp16=(ASR_DEVICE == "cuda"))
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip()
        })

    print(f"[asr] segments: {len(segments)}")
    return segments

# ---------- SEGMENT BUILDER ----------
def _build_takes(local_video):
    print(f"[build_takes] using local video: {local_video}")
    segs = _do_whisper_asr(local_video)
    takes = []
    for idx, s in enumerate(segs):
        takes.append({
            "id": f"T{idx+1:04d}",
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "slot": _guess_slot(s["text"]),
        })
    print(f"[build_takes] total takes: {len(takes)}")
    return takes

def _guess_slot(text: str):
    text_l = text.lower()
    if "why" in text_l or "what if" in text_l:
        return "HOOK"
    if "problem" in text_l or "issue" in text_l:
        return "PROBLEM"
    if "feature" in text_l or "it has" in text_l:
        return "FEATURE"
    if "result" in text_l or "proof" in text_l:
        return "PROOF"
    if "buy" in text_l or "link" in text_l or "check them out" in text_l:
        return "CTA"
    return "HOOK"

# ---------- MAIN PIPELINE ----------
def render_funnel(local_video):
    print(f"[pipeline] starting funnel render for: {local_video}")
    takes = _build_takes(local_video)

    # Split video and export clips
    video = VideoFileClip(local_video)
    clips = []
    for take in takes:
        out_name = f"/tmp/{take['id']}.mp4"
        sub = video.subclip(take["start"], take["end"])
        sub.write_videofile(out_name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        clips.append({
            "id": take["id"],
            "slot": take["slot"],
            "start": take["start"],
            "end": take["end"],
        })
    video.close()
    print(f"[pipeline] exported {len(clips)} clips")

    # Combine slots
    slots = {}
    for c in clips:
        slots.setdefault(c["slot"], []).append(c)
    return local_video, clips, slots

# ---------- PIPELINE WRAPPER ----------
def run_pipeline(local_path, payload):
    print("[run_pipeline] starting full pipeline")
    out_path, clips, slots = render_funnel(local_path)

    s3_key = f"editdna/outputs/{uuid.uuid4().hex}_{int(uuid.uuid1().time)}.mp4"
    s3_url = _upload_to_s3(out_path, s3_key)
    https_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"

    duration = VideoFileClip(local_path).duration
    return {
        "ok": True,
        "input_local": local_path,
        "duration_sec": duration,
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "semantic": True,
        "vision": False,
        "asr": True
    }

# ---------- JOB HANDLER ----------
def job_render(payload):
    """
    Entry point for RQ worker. 
    Handles downloading input file, running pipeline, and returning JSON.
    """
    import urllib.request

    print(f"[job_render] received payload: {payload}")
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mov").name
    input_url = payload["files"][0]
    print(f"[download] {input_url} -> {tmp_path}")
    urllib.request.urlretrieve(input_url, tmp_path)

    result = run_pipeline(tmp_path, payload)
    print(f"[job_render] done, result: {json.dumps(result, indent=2)[:500]}...")
    return result
