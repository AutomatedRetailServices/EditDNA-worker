import os
import json
import math
import uuid
import logging
import subprocess
from typing import List, Dict, Any, Optional

import requests
import boto3

# Whisper + torch
import torch
import whisper

# OpenAI new SDK
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# Global clients / models
# -----------------------------

_openai_client: Optional[OpenAI] = None
_whisper_model: Optional[Any] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is required")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_whisper_model() -> Any:
    global _whisper_model
    if _whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "large-v3")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on device '{device}'")
        _whisper_model = whisper.load_model(model_name, device=device)
    return _whisper_model


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


# -----------------------------
# Small utilities
# -----------------------------

def run_cmd(cmd: List[str]) -> None:
    """Run a shell command and raise if non-zero."""
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}\nOutput:\n{result.stdout}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        logger.info(result.stdout)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_to_local(url: str, dest_path: str) -> None:
    ensure_dir(os.path.dirname(dest_path))
    logger.info(f"Downloading input video: {url} -> {dest_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def probe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        logger.warning(f"ffprobe failed, assuming duration=0\n{result.stdout}")
        return 0.0
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# -----------------------------
# 1) ASR with Whisper
# -----------------------------

def transcribe_with_whisper(video_path: str) -> List[Dict[str, Any]]:
    """
    Run Whisper and return a list of segments:
    [
      { "start": float, "end": float, "text": str }
    ]
    """
    model = get_whisper_model()
    logger.info(f"Running Whisper on: {video_path}")
    result = model.transcribe(video_path, verbose=False)
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
        )
    return segments


# -----------------------------
# 2) Micro-segmentation
# -----------------------------

def micro_segment(
    segments: List[Dict[str, Any]],
    max_len: float = 4.0,
    max_gap: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Take Whisper segments and cut them into smaller micro clips.
    Each micro clip preserves the original text span, but duration is <= max_len.
    """
    micro = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        if not text:
            continue

        dur = end - start
        if dur <= max_len:
            micro.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
            continue

        # Split long segments into ~max_len chunks
        num_chunks = max(1, math.ceil(dur / max_len))
        chunk_len = dur / num_chunks
        for i in range(num_chunks):
            c_start = start + i * chunk_len
            c_end = min(end, c_start + chunk_len)
            micro.append(
                {
                    "start": c_start,
                    "end": c_end,
                    "text": text,
                }
            )

    # Optionally merge tiny gaps if needed (for now just return as is)
    return micro


# -----------------------------
# 3) LLM scoring for funnel slots
# -----------------------------

FUNNEL_SLOTS = ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]


def llm_score_segment(text: str) -> Dict[str, Any]:
    """
    Ask LLM to:
    - assign a funnel slot
        HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA
    - semantic_score: 0..1
    - keep: boolean (drop filler / retries / broken lines / obvious flubs)
    - reason: explanation (for debugging)

    Returns dict with these fields.
    """
    cleaned = text.strip()
    if not cleaned:
        return {
            "slot": "FEATURES",
            "semantic_score": 0.0,
            "keep": False,
            "reason": "Empty text",
        }

    # Super short / obviously filler → don't keep.
    if len(cleaned.split()) < 3:
        return {
            "slot": "FEATURES",
            "semantic_score": 0.1,
            "keep": False,
            "reason": "Too short / likely filler",
        }

    client = get_openai_client()
    model_name = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")

    system_prompt = """
You are scoring short transcript segments from TikTok-style UGC ads.

Your job:
1. Decide which funnel slot the line fits best:
   - HOOK: grabs attention, bold, spicy, problem intro
   - STORY: personal experience, relatable narrative
   - PROBLEM: clearly states pain, frustration, symptoms
   - BENEFITS: outcome-focused, how life feels better
   - FEATURES: ingredients, specs, how it works, instructions
   - PROOF: social proof, "I tried it", testimonials, evidence
   - CTA: tells viewer what to do (click, buy, try, etc.)

2. Decide if we should KEEP this line in the final edit:
   - keep = false for:
     - obviously wrong takes ("wait, not that", "am I saying it right?")
     - repeated attempts, restarts, corrections
     - incomplete phrases that stop mid-sentence
     - meaningless filler ("yeah", "is that good?", "that one good?")
   - keep = true only if it's coherent and useful in a funnel.

3. Give semantic_score from 0 to 1:
   - Higher = clearer, stronger, more on-message for selling.
   - Lower = confusing, weak, filler.

Return STRICT JSON:
{
  "slot": "HOOK" | "STORY" | "PROBLEM" | "BENEFITS" | "FEATURES" | "PROOF" | "CTA",
  "semantic_score": float (0-1),
  "keep": true/false,
  "reason": "short explanation"
}
    """.strip()

    user_prompt = f"Transcript segment:\n\"{cleaned}\""

    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.output[0].content[0].text
        data = json.loads(raw)

        slot = str(data.get("slot", "FEATURES")).upper().strip()
        if slot not in FUNNEL_SLOTS:
            slot = "FEATURES"

        semantic_score = float(data.get("semantic_score", 0.5))
        keep = bool(data.get("keep", True))
        reason = str(data.get("reason", "")).strip()

        return {
            "slot": slot,
            "semantic_score": max(0.0, min(1.0, semantic_score)),
            "keep": keep,
            "reason": reason or "No reason provided",
        }

    except Exception as e:
        logger.exception(f"llm_score_segment failed, falling back: {e}")
        # Fallback: treat as generic FEATURES with medium score.
        return {
            "slot": "FEATURES",
            "semantic_score": 0.5,
            "keep": True,
            "reason": "LLM error, fallback classification",
        }


def score_clip_for_funnel(seg: Dict[str, Any], clip_idx: int) -> Dict[str, Any]:
    """
    Wraps a micro-segment into the clip structure that Bubble expects.
    Adds LLM semantic info and dummy visual scores.
    """
    start = float(seg["start"])
    end = float(seg["end"])
    text = seg["text"].strip()
    clip_id = f"ASR{clip_idx:04d}_c1"

    llm_info = llm_score_segment(text)
    slot = llm_info["slot"]
    semantic_score = llm_info["semantic_score"]
    keep = llm_info["keep"]
    reason = llm_info["reason"]

    # For now visual is stubbed as 1.0 (we can wire vision later).
    visual_score = 1.0
    vtx_sim = semantic_score  # simple proxy

    score = min(semantic_score, visual_score)

    visual_flags = {
        "scene_jump": False,
        "motion_jump": False,
    }

    clip = {
        "id": clip_id,
        "slot": slot,
        "start": start,
        "end": end,
        "score": score,
        "semantic_score": semantic_score,
        "visual_score": visual_score,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": vtx_sim,
        "chain_ids": [clip_id],
        "text": text,
        "llm_reason": reason,
        "visual_flags": visual_flags,
        "meta": {
            "slot": slot,
            "semantic_score": semantic_score,
            "visual_score": visual_score,
            "score": score,
            "chain_ids": [clip_id],
            "keep": keep,
        },
    }

    return clip


# -----------------------------
# 4) Composer – funnel builder
# -----------------------------

def _select_best(clips: List[Dict[str, Any]], slot: str, min_score: float) -> List[Dict[str, Any]]:
    """
    Return all clips for a slot with:
      - meta.keep == True
      - semantic_score >= min_score
    ordered by start time.
    """
    slot_clips = [
        c
        for c in clips
        if c["slot"] == slot
        and c.get("meta", {}).get("keep", True)
        and c.get("semantic_score", 0.0) >= min_score
    ]
    slot_clips.sort(key=lambda c: c["start"])
    return slot_clips


def compose_funnel(scored_clips: List[Dict[str, Any]]) -> (Dict[str, Any], str, List[Dict[str, Any]]):
    """
    Build a TRUE funnel with explicit buckets:

    HOOK → STORY → PROBLEM → BENEFITS → FEATURES → PROOF → CTA

    - No hard limits on how many clips per bucket.
    - We use semantic_score + keep flag to filter.

    Returns:
      composer (dict),
      composer_human (str),
      ordered_clips (list)
    """
    if not scored_clips:
        return {
            "hook_id": None,
            "story_ids": [],
            "problem_ids": [],
            "benefit_ids": [],
            "feature_ids": [],
            "proof_ids": [],
            "cta_id": None,
            "used_clip_ids": [],
            "min_score": 0.0,
        }, "No clips available", []

    # Global min threshold
    min_semantic = 0.6

    hooks = _select_best(scored_clips, "HOOK", min_semantic)
    stories = _select_best(scored_clips, "STORY", min_semantic)
    problems = _select_best(scored_clips, "PROBLEM", min_semantic)
    benefits = _select_best(scored_clips, "BENEFITS", min_semantic)
    features = _select_best(scored_clips, "FEATURES", min_semantic)
    proofs = _select_best(scored_clips, "PROOF", min_semantic)
    ctas = _select_best(scored_clips, "CTA", min_semantic)

    # Choose primary HOOK + CTA (highest semantic score)
    hook_clip = max(hooks, key=lambda c: c["semantic_score"], default=None)
    cta_clip = max(ctas, key=lambda c: c["semantic_score"], default=None)

    # Build ordered list in funnel order.
    ordered: List[Dict[str, Any]] = []

    if hook_clip:
        ordered.append(hook_clip)

    ordered.extend(stories)
    ordered.extend(problems)
    ordered.extend(benefits)
    ordered.extend(features)
    ordered.extend(proofs)

    if cta_clip:
        ordered.append(cta_clip)

    # De-dupe while preserving order
    seen_ids = set()
    unique: List[Dict[str, Any]] = []
    for c in ordered:
        cid = c["id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(c)

    ordered = unique

    composer = {
        "hook_id": hook_clip["id"] if hook_clip else None,
        "story_ids": [c["id"] for c in stories],
        "problem_ids": [c["id"] for c in problems],
        "benefit_ids": [c["id"] for c in benefits],
        "feature_ids": [c["id"] for c in features],
        "proof_ids": [c["id"] for c in proofs],
        "cta_id": cta_clip["id"] if cta_clip else None,
        "used_clip_ids": [c["id"] for c in ordered],
        "min_score": min_semantic,
    }

    # Human-readable view
    lines = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")

    def _add_block(title: str, clips_block: List[Dict[str, Any]]):
        lines.append(f"{title}:\n")
        if not clips_block:
            lines.append("  (none)\n")
            return
        for c in clips_block:
            lines.append(f"  - [{c['id']}] score={c['semantic_score']:.2f} → \"{c['text']}\"\n")
        lines.append("\n")

    # Hook
    lines.append("HOOK:\n")
    if hook_clip:
        lines.append(f"  [{hook_clip['id']}] score={hook_clip['semantic_score']:.2f} → \"{hook_clip['text']}\"\n\n")
    else:
        lines.append("  (none)\n\n")

    _add_block("STORY", stories)
    _add_block("PROBLEM", problems)
    _add_block("BENEFITS", benefits)
    _add_block("FEATURES", features)
    _add_block("PROOF", proofs)

    lines.append("CTA:\n")
    if cta_clip:
        lines.append(f"  [{cta_clip['id']}] score={cta_clip['semantic_score']:.2f} → \"{cta_clip['text']}\"\n\n")
    else:
        lines.append("  (none)\n\n")

    # Final order
    lines.append("FINAL ORDER TIMELINE:\n")
    for idx, c in enumerate(ordered, start=1):
        lines.append(f"{idx}) {c['id']} → \"{c['text']}\"\n")

    lines.append("\n=====================================")

    composer_human = "".join(lines)
    return composer, composer_human, ordered


# -----------------------------
# 5) Video render (ffmpeg)
# -----------------------------

def render_funnel_video(
    input_video: str,
    ordered_clips: List[Dict[str, Any]],
    session_dir: str,
) -> Optional[str]:
    """
    Use ffmpeg to cut each selected clip and then concat.
    Returns path to final video or None if no clips.
    """
    if not ordered_clips:
        logger.warning("render_funnel_video: no clips to render")
        return None

    parts_dir = os.path.join(session_dir, "parts")
    ensure_dir(parts_dir)

    concat_list_path = os.path.join(session_dir, "concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for idx, clip in enumerate(ordered_clips):
            part_path = os.path.join(parts_dir, f"part_{idx:03d}.mp4")
            start = clip["start"]
            end = clip["end"]
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                input_video,
                "-c",
                "copy",
                part_path,
            ]
            run_cmd(cmd)
            f.write(f"file '{part_path}'\n")

    output_path = os.path.join(session_dir, "final.mp4")
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path,
        "-c",
        "copy",
        output_path,
    ]
    run_cmd(cmd_concat)

    if os.path.exists(output_path):
        return output_path
    return None


# -----------------------------
# 6) S3 upload
# -----------------------------

def upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    if not os.path.exists(local_path):
        logger.warning(f"upload_to_s3: file does not exist: {local_path}")
        return None

    bucket = os.getenv("S3_BUCKET") or "script2clipshop-video-automatedretailservices"
    prefix = os.getenv("S3_OUTPUT_PREFIX") or "editdna/outputs"
    key = f"{prefix.rstrip('/')}/{session_id}/final.mp4"

    logger.info(f"Uploading to s3://{bucket}/{key}")
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, key)

    public_base = os.getenv("PUBLIC_BASE_URL")
    if public_base:
        url = f"{public_base.rstrip('/')}/{key}"
    else:
        url = f"https://{bucket}.s3.amazonaws.com/{key}"

    logger.info(f"S3 public URL: {url}")
    return url


# -----------------------------
# 7) Main pipeline entry
# -----------------------------

def _run_pipeline_impl(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal pipeline. `job` MUST have:
      - session_id: str
      - file_urls: [str] (we take the first one for now)
    """
    session_id = job.get("session_id")
    file_urls = job.get("file_urls") or job.get("files")

    if not session_id:
        raise ValueError("job['session_id'] is required")
    if not file_urls or not isinstance(file_urls, list):
        raise ValueError("job['file_urls'] must be a non-empty list")

    input_url = file_urls[0]

    base_tmp = "/tmp/editdna"
    session_dir = os.path.join(base_tmp, session_id)
    ensure_dir(session_dir)

    input_local = os.path.join(session_dir, "input.mp4")
    download_to_local(input_url, input_local)

    duration_sec = probe_duration(input_local)

    # 1) ASR
    whisper_segments = transcribe_with_whisper(input_local)

    # 2) Micro segments
    micro_segments = micro_segment(whisper_segments)

    # 3) Score each micro segment
    clips: List[Dict[str, Any]] = []
    for idx, seg in enumerate(micro_segments):
        clip = score_clip_for_funnel(seg, idx)
        clips.append(clip)

    # 4) Build slots map for UI
    slots_map: Dict[str, List[Dict[str, Any]]] = {slot: [] for slot in FUNNEL_SLOTS}
    for c in clips:
        slots_map.setdefault(c["slot"], []).append(c)

    # 5) Compose funnel
    composer, composer_human, ordered_clips = compose_funnel(clips)

    # 6) Render video
    output_local = render_funnel_video(input_local, ordered_clips, session_dir)
    output_url = upload_to_s3(output_local, session_id) if output_local else None

    return {
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots_map,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_local,
        "output_video_url": output_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }


def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Public entry called from tasks.py:

        out = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )
    """
    job = {
        "session_id": session_id,
        "file_urls": file_urls,
    }
    return _run_pipeline_impl(job)
