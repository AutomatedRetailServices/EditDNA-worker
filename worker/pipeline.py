iimport os
import io
import json
import uuid
import time
import logging
import tempfile
import subprocess
from typing import List, Dict, Any

import requests
import boto3
from faster_whisper import WhisperModel

from .llm import score_clause_multimodal

logger = logging.getLogger("worker.pipeline")
logger.setLevel(logging.INFO)

# ---------- ENV CONFIG ----------

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")   # "cuda" or "cpu"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "9999"))  # effectively no cap

# simple funnel target counts
TARGET_HOOKS = int(os.getenv("TARGET_HOOKS", "1"))
TARGET_PROBLEMS = int(os.getenv("TARGET_PROBLEMS", "1"))
TARGET_FEATURES = int(os.getenv("TARGET_FEATURES", "4"))
TARGET_PROOFS = int(os.getenv("TARGET_PROOFS", "2"))
TARGET_CTAS = int(os.getenv("TARGET_CTAS", "1"))

_whisper_model = None
_s3_client = None


def get_s3():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    logger.info("Loading WhisperModel(%s, device=%s)", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
    _whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    return _whisper_model


def download_to_temp(url: str) -> str:
    logger.info("Downloading %s", url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    suffix = ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    logger.info("Downloaded to %s", path)
    return path


def transcribe_video(path: str) -> Dict[str, Any]:
    """
    Run faster-whisper and return segments with start/end/text.
    """
    model = get_whisper_model()
    logger.info("Starting Whisper transcription...")
    segments, info = model.transcribe(
        path,
        vad_filter=True,
        word_timestamps=False,
    )

    out_segments = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        out_segments.append({
            "id": f"ASR{i:04d}",
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
        })

    duration = float(info.duration) if getattr(info, "duration", None) else 0.0
    logger.info("Transcription complete: %d segments, duration=%.2fs", len(out_segments), duration)
    return {
        "duration": duration,
        "segments": out_segments,
    }


def cut_and_concat_clips(input_path: str,
                         clips: List[Dict[str, Any]],
                         session_id: str) -> str:
    """
    Use ffmpeg to cut each clip and concatenate them.
    Returns path to final mp4.
    """
    tmpdir = tempfile.mkdtemp(prefix="editdna_")
    clip_paths = []

    # 1) cut each clip
    for idx, c in enumerate(clips):
        out_path = os.path.join(tmpdir, f"clip_{idx:03d}.mp4")
        start = max(0.0, c["start"])
        end = max(start, c["end"])
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", input_path,
            "-c", "copy",
            out_path,
        ]
        logger.info("Running ffmpeg cut: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(out_path)

    if not clip_paths:
        raise RuntimeError("No clips to concatenate")

    # 2) write concat list file
    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    # 3) run concat
    final_path = os.path.join(tmpdir, f"{session_id}_out.mp4")
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        final_path,
    ]
    logger.info("Running ffmpeg concat: %s", " ".join(cmd_concat))
    subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return final_path


def upload_to_s3(path: str, s3_prefix: str) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in environment")
    s3 = get_s3()
    key = f"{s3_prefix.rstrip('/')}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    logger.info("Uploading %s to s3://%s/%s", path, S3_BUCKET, key)
    with open(path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }


def select_funnel_clips(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask GPT to label each segment, then pick a simple funnel:
    HOOK -> PROBLEM -> FEATURE(s) -> PROOF(s) -> CTA
    """
    judged = []
    for seg in segments:
        slot, score, reason = score_clause_multimodal(seg["text"], frame_b64=None, slot_hint="STORY")
        seg_j = {**seg, "slot": slot, "score": score, "reason": reason}
        judged.append(seg_j)

    # group by slot
    by_slot: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": [], "STORY": []
    }
    for j in judged:
        by_slot.setdefault(j["slot"], []).append(j)

    # sort each slot by score desc, then by start time
    for k in by_slot:
        by_slot[k].sort(key=lambda x: (-x["score"], x["start"]))

    def take(slot: str, n: int) -> List[Dict[str, Any]]:
        return by_slot.get(slot, [])[:n]

    chosen = []
    chosen.extend(take("HOOK", TARGET_HOOKS))
    chosen.extend(take("PROBLEM", TARGET_PROBLEMS))
    chosen.extend(take("FEATURE", TARGET_FEATURES))
    chosen.extend(take("PROOF", TARGET_PROOFS))
    chosen.extend(take("CTA", TARGET_CTAS))

    # sort global by start time
    chosen.sort(key=lambda x: x["start"])

    # enforce global max duration if set
    total = 0.0
    final = []
    for c in chosen:
        dur = max(0.0, c["end"] - c["start"])
        if MAX_DURATION_SEC and total + dur > MAX_DURATION_SEC:
            # trim last clip if partial room remains
            remaining = max(0.0, MAX_DURATION_SEC - total)
            if remaining > 0.2:
                c = dict(c)
                c["end"] = c["start"] + remaining
                final.append(c)
                total += remaining
            break
        final.append(c)
        total += dur

    # build slots index for response
    slots_index = {k: [] for k in ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]}
    for c in final:
        s = c["slot"]
        if s in slots_index:
            slots_index[s].append({
                "id": c["id"],
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
                "meta": {
                    "slot": s,
                    "score": c["score"],
                    "reason": c["reason"],
                },
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            })

    return {
        "clips": final,
        "slots": slots_index,
    }


def run_pipeline(*,
                 session_id: str,
                 file_urls: List[str],
                 s3_prefix: str) -> Dict[str, Any]:
    """
    Main entrypoint used by tasks.job_render.
    """
    logger.info("[pipeline] run_pipeline() start")
    logger.info("  session_id=%s", session_id)
    logger.info("  file_urls=%s", file_urls)
    input_url = file_urls[0]

    local_in = download_to_temp(input_url)
    asr = transcribe_video(local_in)

    funnel = select_funnel_clips(asr["segments"])

    final_video = cut_and_concat_clips(local_in, funnel["clips"], session_id)
    s3_info = upload_to_s3(final_video, s3_prefix)

    # Build response similar to what your API expects
    clips_for_response = []
    for c in funnel["clips"]:
        clips_for_response.append({
            "id": c["id"],
            "slot": c["slot"],
            "start": c["start"],
            "end": c["end"],
            "score": c["score"],
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [c["id"]],
            "text": c["text"],
        })

    result = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_in,
        "duration_sec": asr["duration"],
        "s3_key": s3_info["s3_key"],
        "s3_url": s3_info["s3_url"],
        "https_url": s3_info["https_url"],
        "clips": clips_for_response,
        "slots": funnel["slots"],
        "asr": True,
        "semantic": True,
        "vision": False,
    }

    logger.info("[pipeline] run_pipeline() done")
    return result
