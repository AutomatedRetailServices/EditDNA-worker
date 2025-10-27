import os, tempfile, subprocess, json, uuid, boto3
from datetime import datetime
from typing import Dict, Any, List

from .tasks import *
from .s3_utils import upload_file_to_s3

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _run(cmd, check=True):
    """Run subprocess and log output"""
    print(f"[ff] $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stderr.decode(), flush=True)
        if check:
            raise subprocess.CalledProcessError(res.returncode, cmd)
    return res

def ffmpeg_subclip(src, dst, ss, ee):
    """Trim clip from ss to ee"""
    dur = float(ee) - float(ss)
    cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-ss", f"{ss:.3f}",
        "-i", src,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        dst
    ]
    _run(cmd)

# --------------------------------------------------
# Main Job
# --------------------------------------------------

def job_render(payload: Dict[str, Any]):
    """Main entry point for worker render job"""
    print(f"[jobs] payload keys: {list(payload.keys())}", flush=True)

    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:6]}")
    files = payload.get("files", [])
    mode = payload.get("mode", "funnel")

    # Download first file to temp
    video_url = files[0]
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    print(f"[jobs] downloading {video_url} → {tmp_video}", flush=True)
    _run(["curl", "-L", "-o", tmp_video, video_url])

    # Run pipeline
    out_path, meta = render_pipeline(tmp_video, mode)

    # Upload to S3
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")
    key_prefix = os.getenv("S3_PREFIX", "editdna/outputs")

    s3_key = f"{key_prefix}/{uuid.uuid4().hex}_{int(datetime.now().timestamp())}.mp4"
    https_url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"

    upload_file_to_s3(out_path, bucket, s3_key)
    print(f"[jobs] uploaded final → {https_url}", flush=True)

    # Compute duration
    cmd = [
        "/usr/bin/ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        out_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    dur = float(res.stdout.decode().strip())

    return {
        "ok": True,
        "input_local": tmp_video,
        "duration_sec": round(dur, 3),
        "s3_key": s3_key,
        "s3_url": f"s3://{bucket}/{s3_key}",
        "https_url": https_url,
        "clips": meta.get("clips", []),
        "slots": meta.get("slots", {}),
        "semantic": meta.get("semantic", False),
        "vision": meta.get("vision", False),
        "asr": meta.get("asr", False),
    }

# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def render_pipeline(video_path, mode="funnel"):
    """Main logic for segmenting and rendering"""
    from .captions import transcribe_audio
    from .semantic_visual_pass import tag_slot, Take, dedup_takes

    # --- Step 1: Transcribe / Segment
    print("[asr] running transcription...", flush=True)
    asr_segments = transcribe_audio(video_path)
    takes = [
        Take(
            id=f"T{str(i+1).zfill(4)}",
            start=s["start"],
            end=s["end"],
            text=s["text"],
            face_q=1.0,
            scene_q=1.0
        )
        for i, s in enumerate(asr_segments)
    ]
    print(f"[seg] takes: {len(takes)}", flush=True)

    # --- Step 2: Semantic Tagging
    takes = dedup_takes(takes)
    for t in takes:
        t.slot_hint = tag_slot(t)

    # --- Step 3: Group by slot
    by_slot: Dict[str, List[Dict[str, Any]]] = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    for t in takes:
        s = t.slot_hint or "FEATURE"
        if s not in by_slot:
            s = "FEATURE"
        by_slot[s].append({
            "id": t.id,
            "slot": s,
            "start": t.start,
            "end": t.end,
            "text": t.text,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "has_product": t.has_product,
            "ocr_hit": t.ocr_hit
        })

    # --- Step 4: Funnel render
    if mode == "funnel":
        final_out, clips_meta = render_funnel(video_path, by_slot)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    meta = {
        "clips": clips_meta,
        "slots": by_slot,
        "semantic": True,
        "vision": False,
        "asr": True
    }
    return final_out, meta

# --------------------------------------------------
# Funnel Builder (Updated)
# --------------------------------------------------

def render_funnel(video_path, by_slot):
    """
    Build a final funnel (HOOK → PROBLEM/BENEFITS → FEATURE → PROOF → CTA)
    Uses FUNNEL_COUNTS, but will not truncate valid flow.
    """
    import os, tempfile

    out_parts = []
    meta = []
    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name

    # Parse funnel counts
    counts_str = os.getenv("FUNNEL_COUNTS", "1,5,5,5,1")
    try:
        funnel_counts = [int(x) for x in counts_str.split(",")]
    except Exception:
        funnel_counts = [1, 5, 5, 5, 1]
    slot_order = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

    # Flatten in logical order
    funnel_takes = []
    for i, slot in enumerate(slot_order):
        takes = by_slot.get(slot, [])
        maxn = funnel_counts[i] if i < len(funnel_counts) else 0
        if maxn == 0:
            funnel_takes.extend(takes)
        else:
            funnel_takes.extend(takes[:maxn])

    # Sort by timeline
    funnel_takes = sorted(funnel_takes, key=lambda t: t["start"])

    # Safety fallback
    if not funnel_takes:
        raise RuntimeError("No valid takes found for funnel render.")

    # Render parts
    part_files = []
    for idx, t in enumerate(funnel_takes):
        part = f"/tmp/ed_{uuid.uuid4().hex}.part{idx+1:02d}.mp4"
        ffmpeg_subclip(video_path, part, t["start"], t["end"])
        part_files.append(part)
        meta.append({
            "slot": t.get("slot"),
            "start": t["start"],
            "end": t["end"],
            "text": t.get("text", ""),
            "score": t.get("score")
        })

    # Write concat list
    with open(tmp_txt, "w") as f:
        for p in part_files:
            f.write(f"file '{p}'\n")

    # Concatenate
    final_out = f"/tmp/ed_{uuid.uuid4().hex()}.mp4"
    cmd = [
        "/usr/bin/ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp_txt,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        final_out
    ]
    _run(cmd)

    print(f"[funnel] rendered {len(part_files)} parts → {final_out}", flush=True)
    return final_out, meta
