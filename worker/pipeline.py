import os
import io
import json
import math
import uuid
import time
import shutil
import logging
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
import torch
import whisper
from openai import OpenAI

# ------------------------------------------------------------------------------------
# Global setup
# ------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

EDITDNA_TMP_ROOT = os.environ.get("DOWNLOAD_ROOT", "TMP")
EDITDNA_TMP_ROOT = EDITDNA_TMP_ROOT if os.path.isabs(EDITDNA_TMP_ROOT) else f"/tmp/{EDITDNA_TMP_ROOT}"

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

ASR_ENABLED = os.environ.get("ASR_ENABLED", "1") == "1"
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

VISION_ENABLED = os.environ.get("VISION_ENABLED", "1") == "1"
W_VISION = float(os.environ.get("W_VISION", "0.7"))

EDITDNA_MIN_CLIP_SCORE = float(os.environ.get("EDITDNA_MIN_CLIP_SCORE", "0.7"))
EDITDNA_HOOK_MIN_SCORE = float(os.environ.get("EDITDNA_HOOK_MIN_SCORE", "0.7"))
EDITDNA_CTA_MIN_SCORE = float(os.environ.get("EDITDNA_CTA_MIN_SCORE", "0.6"))

VETO_MIN_SCORE = float(os.environ.get("VETO_MIN_SCORE", "0.4"))

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs").strip("/")
S3_ACL = os.environ.get("S3_ACL", "public-read")
PRESIGN_EXPIRES = int(os.environ.get("PRESIGN_EXPIRES", "604800"))  # default 7 days

TARGET_DURATION_SEC = float(os.environ.get("TARGET_DURATION_SEC", "0"))  # 0 = no target

client = OpenAI()  # uses OPENAI_API_KEY from env

# Lazy-loaded Whisper model
_WHISPER_MODEL = None


# ------------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------------

@dataclass
class Clip:
    id: str
    slot: str
    start: float
    end: float
    score: float
    semantic_score: float
    visual_score: float
    face_q: float
    scene_q: float
    vtx_sim: float
    chain_ids: List[str]
    text: str
    llm_reason: str
    visual_flags: Dict[str, bool]
    meta: Dict[str, Any]


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download_to_local(session_dir: str, url: str) -> str:
    """
    Download a remote video URL to session_dir/input.mp4
    (simple implementation using urllib).
    """
    _ensure_dir(session_dir)
    local_path = os.path.join(session_dir, "input.mp4")

    logger.info("⬇️ Downloading input video", extra={"url": url, "local": local_path})
    with urlopen(url) as r, open(local_path, "wb") as f:
        shutil.copyfileobj(r, f)

    return local_path


def _load_whisper_model() -> whisper.Whisper:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    device = "cuda" if (ASR_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("🧠 Loading Whisper model", extra={"model": WHISPER_MODEL_NAME, "device": device})
    _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    return _WHISPER_MODEL


def _run_asr(input_path: str) -> List[Dict[str, Any]]:
    """
    Run Whisper ASR and return segments with start, end, text.
    """
    if not ASR_ENABLED:
        logger.warning("ASR is disabled via ASR_ENABLED=0")
        return []

    model = _load_whisper_model()
    logger.info("🎙  Running ASR", extra={"input": input_path})
    result = model.transcribe(input_path, language=ASR_LANGUAGE, verbose=False)
    segments = result.get("segments", [])
    logger.info("🎙  ASR done", extra={"segments": len(segments)})
    return segments


# ------------------------------------------------------------------------------------
# LLM scoring
# ------------------------------------------------------------------------------------

FUNNEL_SLOTS = ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]


def llm_score_segment(text: str) -> Dict[str, Any]:
    """
    Ask the LLM to:
    - classify the segment into one of HOOK/STORY/PROBLEM/BENEFITS/FEATURES/PROOF/CTA
    - assign a semantic_score in [0,1]
    - decide keep (True/False)
    - give a short reason

    It MUST return a JSON object with keys:
    - slot
    - semantic_score
    - keep
    - reason
    """
    text = (text or "").strip()
    if not text:
        return {
            "slot": "FEATURES",
            "semantic_score": 0.0,
            "keep": False,
            "llm_reason": "Empty text segment",
        }

    system_prompt = (
        "You are a conversion ad editor for TikTok Shop.\n"
        "Classify a single spoken segment from a product video into ONE funnel slot and decide if it should be kept.\n\n"
        "Funnel slots:\n"
        "- HOOK: grabs attention, bold claim, question, pattern interrupt, curiosity.\n"
        "- STORY: short personal story, context, relatable moment, before/after.\n"
        "- PROBLEM: pain, frustration, what goes wrong without the product.\n"
        "- BENEFITS: outcomes, transformations, how life feels with the product.\n"
        "- FEATURES: concrete details (ingredients, materials, specs, usage, dosage).\n"
        "- PROOF: testimonials, social proof, compliments, credibility, numbers.\n"
        "- CTA: clear action: click, buy, add to cart, shop link, etc.\n\n"
        "Rules:\n"
        "- Mark keep=false for obvious flubs, restarts, outtakes, or meta lines like "
        "  'wait, let me try again', 'am I saying it right', 'cut that part', laughing, or trailing off mid-sentence.\n"
        "- Slang is FINE if coherent and usable with the product (never reject just because of slang).\n"
        "- Do NOT censor or sanitize; focus only on usefulness for an ad edit.\n"
        "- semantic_score should be between 0.0 and 1.0 for how strong this line is for its slot.\n\n"
        "Reply as a JSON object with keys: slot, semantic_score, keep, reason.\n"
        "Example:\n"
        "{\n"
        '  \"slot\": \"HOOK\",\n'
        '  \"semantic_score\": 0.9,\n'
        '  \"keep\": true,\n'
        '  \"reason\": \"Strong curiosity hook calling out a specific pain.\"\n'
        "}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        content = resp.choices[0].message.content
        data = json.loads(content)

        slot = str(data.get("slot", "FEATURES")).upper().strip()
        if slot not in FUNNEL_SLOTS:
            # map some common variants
            if slot == "BENEFIT":
                slot = "BENEFITS"
            elif slot == "PROB":
                slot = "PROBLEM"
            else:
                slot = "FEATURES"

        semantic_score = float(data.get("semantic_score", 0.5))
        semantic_score = max(0.0, min(1.0, semantic_score))
        keep = bool(data.get("keep", True))
        reason = str(data.get("reason", "")).strip() or "LLM scoring"

        return {
            "slot": slot,
            "semantic_score": semantic_score,
            "keep": keep,
            "llm_reason": reason,
        }

    except Exception as e:
        logger.exception("LLM scoring failed for segment", extra={"text": text[:120]})
        return {
            "slot": "FEATURES",
            "semantic_score": 0.5,
            "keep": True,
            "llm_reason": "LLM error, fallback classification",
        }


def _normalize_text_for_dedupe(t: str) -> str:
    t = (t or "").lower().strip()
    for ch in [",", ".", "!", "?", ":", ";"]:
        t = t.replace(ch, "")
    t = " ".join(t.split())
    return t


def score_clip_for_funnel(seg_idx: int, seg: Dict[str, Any]) -> Clip:
    """
    Turn a Whisper segment into a Clip with LLM + (dummy) visual scoring.
    """
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start + 2.0))
    text = str(seg.get("text", "")).strip()

    llm_info = llm_score_segment(text)
    slot = llm_info["slot"]
    semantic_score = float(llm_info["semantic_score"])
    keep = bool(llm_info["keep"])
    llm_reason = str(llm_info["llm_reason"]) if "llm_reason" in llm_info else str(llm_info.get("reason", ""))

    # Basic visual / quality scores (we're not running vision models yet)
    visual_score = 1.0
    face_q = 1.0
    scene_q = 1.0
    vtx_sim = semantic_score  # reuse for now

    # Combine semantic + visual
    if VISION_ENABLED:
        score = (1.0 - W_VISION) * semantic_score + W_VISION * visual_score
    else:
        score = semantic_score

    meta = {
        "slot": slot,
        "semantic_score": semantic_score,
        "visual_score": visual_score,
        "score": score,
        "chain_ids": [],
        "keep": keep and score >= VETO_MIN_SCORE,
    }

    clip_id = f"ASR{seg_idx:04d}_c1"

    return Clip(
        id=clip_id,
        slot=slot,
        start=start,
        end=end,
        score=score,
        semantic_score=semantic_score,
        visual_score=visual_score,
        face_q=face_q,
        scene_q=scene_q,
        vtx_sim=vtx_sim,
        chain_ids=[clip_id],
        text=text,
        llm_reason=llm_reason,
        visual_flags={"scene_jump": False, "motion_jump": False},
        meta=meta,
    )


# ------------------------------------------------------------------------------------
# Composer
# ------------------------------------------------------------------------------------

def _compose_funnel(clips: List[Clip]) -> Dict[str, Any]:
    """
    Build a single funnel:
      HOOK → STORY → PROBLEM → BENEFITS → FEATURES → PROOF → CTA

    No hard limits on how many sentences per slot.
    We:
      - filter by meta.keep and EDITDNA_MIN_CLIP_SCORE
      - choose strongest HOOK and CTA
      - keep *all* good STORY/PROBLEM/BENEFITS/FEATURES/PROOF (deduped)
      - preserve chronological order within each slot
    """
    logger.info("🧩 Composing funnel", extra={"total_clips": len(clips)})

    usable = [c for c in clips if c.meta.get("keep", True) and c.score >= EDITDNA_MIN_CLIP_SCORE]
    usable_sorted = sorted(usable, key=lambda c: c.start)

    by_slot: Dict[str, List[Clip]] = {s: [] for s in FUNNEL_SLOTS}
    for c in usable_sorted:
        by_slot.setdefault(c.slot, []).append(c)

    def pick_best(sl: str, min_score: float) -> Optional[Clip]:
        candidates = [c for c in by_slot.get(sl, []) if c.score >= min_score]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.score)

    # Best HOOK & CTA
    hook_clip = pick_best("HOOK", EDITDNA_HOOK_MIN_SCORE)
    cta_clip = pick_best("CTA", EDITDNA_CTA_MIN_SCORE)

    # Dedup within each multi-clip slot
    def dedupe_slot(slot_name: str) -> List[Clip]:
        seen = set()
        out: List[Clip] = []
        for c in by_slot.get(slot_name, []):
            norm = _normalize_text_for_dedupe(c.text)
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            out.append(c)
        return out

    story_clips = dedupe_slot("STORY")
    problem_clips = dedupe_slot("PROBLEM")
    benefit_clips = dedupe_slot("BENEFITS")
    feature_clips = dedupe_slot("FEATURES")
    proof_clips = dedupe_slot("PROOF")

    # Final order
    timeline: List[Clip] = []
    if hook_clip:
        timeline.append(hook_clip)

    timeline.extend(story_clips)
    timeline.extend(problem_clips)
    timeline.extend(benefit_clips)
    timeline.extend(feature_clips)
    timeline.extend(proof_clips)

    if cta_clip:
        timeline.append(cta_clip)

    used_clip_ids = [c.id for c in timeline]

    composer = {
        "hook_id": hook_clip.id if hook_clip else None,
        "story_ids": [c.id for c in story_clips],
        "problem_ids": [c.id for c in problem_clips],
        "benefit_ids": [c.id for c in benefit_clips],
        "feature_ids": [c.id for c in feature_clips],
        "proof_ids": [c.id for c in proof_clips],
        "cta_id": cta_clip.id if cta_clip else None,
        "used_clip_ids": used_clip_ids,
        "min_score": EDITDNA_MIN_CLIP_SCORE,
    }

    # Human-readable debug string
    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====")

    def add_block(title: str, clips_list: List[Clip]):
        lines.append(f"{title}:")
        if not clips_list:
            lines.append("  (none)")
        else:
            for c in clips_list:
                lines.append(f"  [{c.id}] score={c.score:.2f} → {json.dumps(c.text)}")

    add_block("HOOK", [hook_clip] if hook_clip else [])
    add_block("STORY", story_clips)
    add_block("PROBLEM", problem_clips)
    add_block("BENEFITS", benefit_clips)
    add_block("FEATURES", feature_clips)
    add_block("PROOF", proof_clips)
    add_block("CTA", [cta_clip] if cta_clip else [])

    lines.append("")
    lines.append("FINAL ORDER TIMELINE:")
    for idx, c in enumerate(timeline, start=1):
        lines.append(f"{idx}) {c.id} → {json.dumps(c.text)}")

    lines.append("")
    lines.append("=====================================")

    composer_human = "\n".join(lines)
    return composer, composer_human, timeline


# ------------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------------

def _ffmpeg_cut_segment(input_path: str, start: float, end: float, out_path: str) -> None:
    duration = max(0.0, end - start)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        out_path,
    ]
    logger.info("✂️ ffmpeg cut", extra={"cmd": " ".join(cmd)})
    subprocess.run(cmd, check=True)


def _ffmpeg_concat(segments: List[str], out_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for p in segments:
            f.write(f"file '{p}'\n")
        list_path = f.name

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        out_path,
    ]
    logger.info("🎬 ffmpeg concat", extra={"cmd": " ".join(cmd)})
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(list_path)


def _render_funnel_video(input_path: str, session_dir: str, timeline: List[Clip]) -> Optional[str]:
    if not timeline:
        logger.warning("No clips in timeline; skipping render")
        return None

    segments_dir = os.path.join(session_dir, "segments")
    _ensure_dir(segments_dir)
    cut_paths: List[str] = []

    for idx, c in enumerate(timeline, start=1):
        seg_out = os.path.join(segments_dir, f"seg_{idx:03d}.mp4")
        _ffmpeg_cut_segment(input_path, c.start, c.end, seg_out)
        cut_paths.append(seg_out)

    final_path = os.path.join(session_dir, "final.mp4")
    _ffmpeg_concat(cut_paths, final_path)

    return final_path


# ------------------------------------------------------------------------------------
# S3 upload
# ------------------------------------------------------------------------------------

def _upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not set; skipping upload")
        return None

    key = f"{S3_PREFIX}/{session_id}-{os.path.basename(local_path)}"
    logger.info("☁️ Uploading to S3", extra={"bucket": S3_BUCKET, "key": key})

    s3 = boto3.client("s3", region_name=S3_REGION)

    extra_args = {}
    if S3_ACL:
        extra_args["ACL"] = S3_ACL

    try:
        s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs=extra_args)
    except Exception:
        logger.exception("S3 upload failed")
        return None

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=PRESIGN_EXPIRES,
        )
        return url
    except Exception:
        logger.exception("Failed to generate presigned URL; falling back to HTTPS URL")
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


# ------------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------------

def _run_pipeline_impl(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core pipeline.
    job contains:
      - session_id: str
      - file_urls: List[str]  (we use the first one)
    """
    session_id = job["session_id"]
    file_urls: List[str] = job["file_urls"]

    if not file_urls:
        raise ValueError("file_urls must be a non-empty list")

    session_dir = os.path.join(EDITDNA_TMP_ROOT, "editdna", session_id)
    _ensure_dir(session_dir)

    # 1) Download
    input_url = file_urls[0]
    input_local = _download_to_local(session_dir, input_url)

    # 2) ASR
    asr_segments = _run_asr(input_local)
    clips: List[Clip] = []
    for idx, seg in enumerate(asr_segments):
        clips.append(score_clip_for_funnel(idx, seg))

    # 3) Composer
    composer, composer_human, timeline = _compose_funnel(clips)

    # 4) Render
    output_video_local = None
    output_video_url = None

    if composer["used_clip_ids"]:
        try:
            output_video_local = _render_funnel_video(input_local, session_dir, timeline)
        except Exception:
            logger.exception("Render failed; skipping video output")
            output_video_local = None

        if output_video_local and os.path.exists(output_video_local):
            output_video_url = _upload_to_s3(output_video_local, session_id)

    # 5) Prepare response
    clips_payload = [asdict(c) for c in clips]

    slots: Dict[str, List[Dict[str, Any]]] = {s: [] for s in FUNNEL_SLOTS}
    for c in clips:
        slots.setdefault(c.slot, []).append(asdict(c))

    duration_sec = 0.0
    if asr_segments:
        duration_sec = float(asr_segments[-1].get("end", 0.0))

    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips_payload,
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_video_local,
        "output_video_url": output_video_url,
        "asr": ASR_ENABLED,
        "semantic": True,
        "vision": VISION_ENABLED,
    }

    return result


def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Public entrypoint called from tasks.job_render.

    tasks.py does:
        out = pipeline.run_pipeline(session_id=session_id, file_urls=files)
    """
    job = {
        "session_id": session_id,
        "file_urls": file_urls,
    }
    return _run_pipeline_impl(job)
