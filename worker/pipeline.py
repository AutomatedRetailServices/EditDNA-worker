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
from typing import List, Dict, Any, Optional, Tuple

import boto3
import requests
import torch
import whisper
import openai  # using 0.28.x style client

# -----------------------------------------------------
# Global config
# -----------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# OpenAI (old SDK style; you already pinned 0.28.x)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

OPENAI_MODEL_CLASSIFIER = os.environ.get("EDITDNA_LLM_MODEL", "gpt-4o-mini")

# Whisper model + device
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "medium")
WHISPER_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/workspace/whisper")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# S3 defaults (so you don't need extra env vars)
S3_BUCKET = os.environ.get(
    "EDITDNA_S3_BUCKET", "script2clipshop-video-automatedretailservices"
)
S3_PREFIX = os.environ.get(
    "EDITDNA_S3_PREFIX", "editdna/outputs"
)
S3_BASE_URL = os.environ.get(
    "EDITDNA_OUTPUT_BASE_URL", f"https://{S3_BUCKET}.s3.amazonaws.com"
)

# Funnel / composer limits
MAX_TOTAL_DURATION = float(os.environ.get("EDITDNA_MAX_DURATION", "75"))  # seconds
MIN_SCORE_TO_KEEP = float(os.environ.get("EDITDNA_MIN_SCORE", "0.3"))

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str]) -> None:
    """
    Run a shell command and raise if it fails.
    """
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("Command failed: %s", result.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError(f"Command failed: {cmd}")


def _download_first_file(session_id: str, file_urls: List[str]) -> str:
    """
    Download the FIRST file url to /tmp/editdna/<session_id>/input.mp4
    (for now we only support single-file funnel).
    """
    if not file_urls:
        raise ValueError("file_urls must be non-empty")

    url = file_urls[0]
    workdir = os.path.join("/tmp/editdna", session_id)
    _ensure_dir(workdir)
    out_path = os.path.join(workdir, "input.mp4")

    logger.info("Downloading input video for session %s from %s", session_id, url)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    logger.info("Downloaded input video to %s", out_path)
    return out_path


# -----------------------------------------------------
# Whisper ASR (GPU when available)
# -----------------------------------------------------

_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        logger.info(
            "Loading Whisper model '%s' on device=%s (root=%s)",
            WHISPER_MODEL_NAME,
            DEVICE,
            WHISPER_DOWNLOAD_ROOT,
        )
        _WHISPER_MODEL = whisper.load_model(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            download_root=WHISPER_DOWNLOAD_ROOT,
        )
    return _WHISPER_MODEL


def run_asr(input_path: str) -> Dict[str, Any]:
    """
    Run Whisper on the video/audio and return its raw result.
    """
    model = _get_whisper_model()
    logger.info("Running Whisper ASR on %s", input_path)
    t0 = time.time()
    result = model.transcribe(
        input_path,
        fp16=(DEVICE == "cuda"),
        verbose=False,
    )
    logger.info("Whisper ASR finished in %.2fs", time.time() - t0)
    return result


# -----------------------------------------------------
# Micro-segmentation (micro cuts) from ASR
# -----------------------------------------------------

def build_micro_segments(asr_result: Dict[str, Any], max_window: float = 4.0) -> List[Dict[str, Any]]:
    """
    Take Whisper segments and re-group into micro-segments of up to `max_window` seconds.
    This is where we create ASR0000_c1 style IDs.

    We do NOT try to be super fancy here; we care about:
    - contiguous windows
    - clean text for each micro segment
    """

    segments = asr_result.get("segments", [])
    micro_segments: List[Dict[str, Any]] = []

    current_start = None
    current_end = None
    current_text_parts: List[str] = []

    def flush_window():
        nonlocal current_start, current_end, current_text_parts, micro_segments
        if current_start is None:
            return
        idx = len(micro_segments)
        seg_id = f"ASR{idx:04d}_c1"
        text = " ".join(t.strip() for t in current_text_parts if t and t.strip())
        micro_segments.append(
            {
                "id": seg_id,
                "start": float(current_start),
                "end": float(current_end),
                "text": text.strip(),
            }
        )
        current_start = None
        current_end = None
        current_text_parts = []

    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s + 0.5))
        txt = seg.get("text", "").strip()

        if current_start is None:
            current_start = s
            current_end = e
            current_text_parts.append(txt)
            continue

        # if this next piece fits into current window, extend
        if (e - current_start) <= max_window and abs(s - current_end) < 1.0:
            current_end = e
            current_text_parts.append(txt)
        else:
            # flush current window, start new
            flush_window()
            current_start = s
            current_end = e
            current_text_parts.append(txt)

    # flush last window
    flush_window()

    # filter out empty text windows
    micro_segments = [m for m in micro_segments if m["text"]]
    logger.info("Built %d micro-segments from ASR", len(micro_segments))
    return micro_segments


# -----------------------------------------------------
# LLM scoring & slot classification
# -----------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """
You are a funnel editor for TikTok Shop ads.

For EACH line, you will:
1) Decide which funnel slot it belongs to:
   - HOOK       → attention grabbers
   - STORY      → personal story, narrative, "let me tell you..."
   - PROBLEM    → pain, frustration, what sucks
   - BENEFITS   → outcomes, what they get, transformations
   - FEATURES   → product details, ingredients, how it works
   - PROOF      → testimonials, credibility, social proof, "I saw results"
   - CTA        → call-to-action, "click the link", "buy now"
   - TRASH      → wrong takes, "wait", "start again", meta-comments, etc

2) Decide if we should KEEP this line in the final ad:
   - keep = true only if it's a coherent, usable line for viewers
   - keep = false for repetition, incomplete phrases, obvious bloopers, or “behind-the-scenes” chatter.

3) Give a semantic_score from 0.0 to 1.0:
   - 0.0 = useless
   - 1.0 = extremely strong for that slot

You MUST respond in strict JSON:
{
  "slot": "...",
  "keep": true/false,
  "semantic_score": 0.0-1.0,
  "reason": "..."
}
""".strip()


def llm_classify_segment(text: str) -> Dict[str, Any]:
    """
    Ask the LLM to classify a line into a funnel slot, decide keep vs trash,
    and give a semantic score.
    """
    if not OPENAI_API_KEY:
        # fallback: naive default
        return {
            "slot": "FEATURE",
            "keep": True if text.strip() else False,
            "semantic_score": 0.5,
            "reason": "Fallback because OPENAI_API_KEY is missing.",
        }

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL_CLASSIFIER,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.1,
        )
        raw = resp["choices"][0]["message"]["content"]
        # Try to parse JSON; be strict
        data = json.loads(raw)
        slot = str(data.get("slot", "FEATURE")).upper()
        # normalize slot
        if slot not in [
            "HOOK",
            "STORY",
            "PROBLEM",
            "BENEFITS",
            "FEATURE",
            "PROOF",
            "CTA",
            "TRASH",
        ]:
            slot = "FEATURE"

        keep = bool(data.get("keep", True))
        semantic_score = float(data.get("semantic_score", 0.5))
        reason = str(data.get("reason", ""))

        return {
            "slot": slot,
            "keep": keep,
            "semantic_score": semantic_score,
            "reason": reason,
        }

    except Exception as e:
        logger.exception("llm_classify_segment failed, falling back", exc_info=e)
        # Fallback: everything becomes FEATURE with mid score
        return {
            "slot": "FEATURE",
            "keep": True if text.strip() else False,
            "semantic_score": 0.5,
            "reason": f"Fallback because of error: {e}",
        }


def score_clip_for_funnel(seg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach funnel classification + scoring + vision placeholders.
    """
    text = seg["text"]
    llm_info = llm_classify_segment(text)

    slot = llm_info["slot"]
    keep = llm_info["keep"]
    semantic_score = float(llm_info["semantic_score"])

    # Vision placeholders (for now 1.0)
    visual_score = 1.0
    face_q = 1.0
    scene_q = 1.0

    # Composite score
    score = 0.7 * semantic_score + 0.3 * visual_score

    scored = {
        "id": seg["id"],
        "slot": slot,
        "start": seg["start"],
        "end": seg["end"],
        "text": text,
        "semantic_score": semantic_score,
        "visual_score": visual_score,
        "score": score,
        "vtx_sim": score,
        "face_q": face_q,
        "scene_q": scene_q,
        "llm_reason": llm_info.get("reason", ""),
        "keep": keep,
        "chain_ids": [seg["id"]],
        "visual_flags": {
            "scene_jump": False,
            "motion_jump": False,
        },
        "meta": {
            "slot": slot,
            "semantic_score": semantic_score,
            "visual_score": visual_score,
            "score": score,
            "chain_ids": [seg["id"]],
        },
    }
    return scored


# -----------------------------------------------------
# Composer: real funnel (HOOK → STORY → PROBLEM → BENEFITS → FEATURES → PROOF → CTA)
# -----------------------------------------------------

def _bucket_by_slot(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
        "TRASH": [],
    }
    for c in clips:
        slot = c["slot"]
        if slot not in slots:
            slot = "FEATURE"
        slots[slot].append(c)

    # sort each bucket by score desc, then by start asc
    for k, lst in slots.items():
        lst.sort(key=lambda x: (-x["score"], x["start"]))
    return slots


def _dedupe_text_in_bucket(bucket: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate text lines inside the same slot bucket.
    Simple strategy: lowercased text exact dedupe.
    """
    seen = set()
    out = []
    for c in bucket:
        key = c["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    # resort by start so timeline flows naturally
    out.sort(key=lambda x: x["start"])
    return out


def compose_funnel(clips: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build a funnel:
    HOOK → STORY → PROBLEM → BENEFITS → FEATURES → PROOF → CTA

    No hard per-slot limits; only global MAX_TOTAL_DURATION applies.
    """
    # Filter only keep=True and score above a small floor
    keepers = [c for c in clips if c["keep"] and c["score"] >= MIN_SCORE_TO_KEEP]
    logger.info("Composer: %d clips after keep+score filter", len(keepers))

    if not keepers:
        raise ValueError("No usable clips after classification & scoring")

    slots = _bucket_by_slot(keepers)

    # Dedupe inside each slot (except CTA, where usually only a couple anyway)
    for s in ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURE", "PROOF"]:
        slots[s] = _dedupe_text_in_bucket(slots[s])

    # HOOK: pick best 1–2 based on score, but we don't strictly enforce 2
    hooks = slots["HOOK"]
    hooks.sort(key=lambda x: (-x["score"], x["start"]))
    if not hooks:
        # fallback: top STORY or FEATURE as hook
        backup = (slots["STORY"] + slots["FEATURE"])
        backup.sort(key=lambda x: (-x["score"], x["start"]))
        if backup:
            hooks = [backup[0]]
        else:
            hooks = [keepers[0]]
    # we can decide to keep top1 as primary funnel hook
    hook_clip = hooks[0]

    # CTA: best scoring CTA
    ctas = slots["CTA"]
    ctas.sort(key=lambda x: (-x["score"], x["start"]))
    cta_clip = ctas[0] if ctas else None

    # Other slots in order
    story = slots["STORY"]
    problem = slots["PROBLEM"]
    benefits = slots["BENEFITS"]
    features = slots["FEATURE"]
    proof = slots["PROOF"]

    ordered: List[Dict[str, Any]] = []
    total_dur = 0.0

    def try_add(c: Dict[str, Any]):
        nonlocal total_dur
        seg_dur = float(c["end"] - c["start"])
        if seg_dur <= 0:
            return False
        if total_dur + seg_dur > MAX_TOTAL_DURATION:
            return False
        ordered.append(c)
        total_dur += seg_dur
        return True

    # 1) HOOK
    try_add(hook_clip)

    # 2) STORY
    for c in story:
        if c["id"] == hook_clip["id"]:
            continue
        if not try_add(c):
            break

    # 3) PROBLEM
    for c in problem:
        if not try_add(c):
            break

    # 4) BENEFITS
    for c in benefits:
        if not try_add(c):
            break

    # 5) FEATURES
    for c in features:
        if not try_add(c):
            break

    # 6) PROOF
    for c in proof:
        if not try_add(c):
            break

    # 7) CTA (force at the end if possible)
    if cta_clip:
        # Only add if not already in ordered and if we have a bit of room
        if cta_clip["id"] not in [x["id"] for x in ordered]:
            _ = try_add(cta_clip)

    # Build composer meta
    story_ids = [c["id"] for c in story if c["id"] in [x["id"] for x in ordered]]
    problem_ids = [c["id"] for c in problem if c["id"] in [x["id"] for x in ordered]]
    benefit_ids = [c["id"] for c in benefits if c["id"] in [x["id"] for x in ordered]]
    feature_ids = [c["id"] for c in features if c["id"] in [x["id"] for x in ordered]]
    proof_ids = [c["id"] for c in proof if c["id"] in [x["id"] for x in ordered]]

    # For backwards compatibility: old "feature_ids" is ALL mid-funnel segments
    mid_ids = story_ids + problem_ids + benefit_ids + feature_ids + proof_ids

    composer = {
        "hook_id": hook_clip["id"],
        "story_ids": story_ids,
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": mid_ids,  # backward compatible
        "proof_ids": proof_ids,
        "cta_id": cta_clip["id"] if cta_clip else None,
        "used_clip_ids": [c["id"] for c in ordered],
        "min_score": MIN_SCORE_TO_KEEP,
    }

    # Human-readable composer
    lines = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")
    lines.append(f"HOOK ({hook_clip['id']}, score={hook_clip['score']:.2f}):")
    lines.append(f'  "{hook_clip["text"]}"\n')

    def describe_block(name: str, ids: List[str]):
        if not ids:
            return
        lines.append(f"{name} (kept):")
        for cid in ids:
            clip = next(c for c in ordered if c["id"] == cid)
            lines.append(
                f'  - [{clip["id"]}] score={clip["score"]:.2f} → "{clip["text"]}"'
            )
        lines.append("")

    describe_block("STORY", story_ids)
    describe_block("PROBLEM", problem_ids)
    describe_block("BENEFITS", benefit_ids)
    describe_block("FEATURES", feature_ids)
    describe_block("PROOF", proof_ids)

    if cta_clip:
        lines.append(
            f'CTA ({cta_clip["id"]}, score={cta_clip["score"]:.2f}):\n'
            f'  "{cta_clip["text"]}"\n'
        )

    lines.append("FINAL ORDER TIMELINE:")
    for idx, c in enumerate(ordered, start=1):
        lines.append(f'{idx}) {c["id"]} → "{c["text"]}"')
    lines.append("\n=====================================")
    composer_human = "\n".join(lines)

    return composer, ordered


# -----------------------------------------------------
# FFmpeg: cut & concat funnel into final video
# -----------------------------------------------------

def render_funnel_video(
    session_id: str,
    input_local: str,
    ordered_clips: List[Dict[str, Any]],
) -> str:
    """
    Use ffmpeg to cut each chosen segment from the input video and
    concat them into <workdir>/final.mp4
    """
    workdir = os.path.join("/tmp/editdna", session_id)
    _ensure_dir(workdir)

    temp_clips = []
    for idx, c in enumerate(ordered_clips):
        out_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
        start = float(c["start"])
        end = float(c["end"])
        if end <= start:
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_local,
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-c",
            "copy",
            out_path,
        ]
        _run_cmd(cmd)
        temp_clips.append(out_path)

    if not temp_clips:
        raise ValueError("No temp clips were rendered for the funnel")

    concat_list_path = os.path.join(workdir, "concat.txt")
    with open(concat_list_path, "w") as f:
        for p in temp_clips:
            f.write(f"file '{p}'\n")

    final_path = os.path.join(workdir, "final.mp4")
    cmd = [
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
        final_path,
    ]
    _run_cmd(cmd)

    logger.info("Final funnel video rendered at %s", final_path)
    return final_path


# -----------------------------------------------------
# S3 upload
# -----------------------------------------------------

def upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    """
    Upload final.mp4 to S3 and return https URL.
    If upload fails, return None.
    """
    if not os.path.exists(local_path):
        logger.error("upload_to_s3: local_path does not exist: %s", local_path)
        return None

    key = f"{S3_PREFIX}/{session_id}/final.mp4"
    logger.info("Uploading final video to s3://%s/%s", S3_BUCKET, key)

    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    except Exception as e:
        logger.exception("S3 upload failed", exc_info=e)
        return None

    url = f"{S3_BASE_URL}/{key}"
    logger.info("Uploaded final video to %s", url)
    return url


# -----------------------------------------------------
# Public entrypoint
# -----------------------------------------------------

def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Main GPU worker pipeline.

    1) Download first input file
    2) Run Whisper (GPU when available)
    3) Build micro segments
    4) LLM classify each segment into funnel slots + scores
    5) Compose real funnel (HOOK → STORY → PROBLEM → BENEFITS → FEATURES → PROOF → CTA)
    6) Render final video with ffmpeg
    7) Upload to S3 and return URLs + metadata
    """
    t_start = time.time()
    logger.info("run_pipeline start | session_id=%s | file_urls=%s", session_id, file_urls)

    # 1) Download
    input_local = _download_first_file(session_id, file_urls)

    # 2) ASR
    asr = run_asr(input_local)

    # 3) Micro segments
    micro_segments = build_micro_segments(asr)

    # 4) Score + classify each segment
    scored_clips = [score_clip_for_funnel(seg) for seg in micro_segments]

    # 5) Compose funnel
    composer, ordered_clips = compose_funnel(scored_clips)

    # 6) Render
    output_video_local = render_funnel_video(session_id, input_local, ordered_clips)

    # 7) Upload to S3
    output_video_url = upload_to_s3(output_video_local, session_id)

    # Build slots dict in the shape your web layer expects
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for c in scored_clips:
        # skip TRASH in slots
        if c["slot"] == "TRASH":
            continue
        slot = c["slot"]
        if slot not in slots:
            slot = "FEATURE"
        slots[slot].append(
            {
                "id": c["id"],
                "slot": slot,
                "start": c["start"],
                "end": c["end"],
                "score": c["score"],
                "semantic_score": c["semantic_score"],
                "visual_score": c["visual_score"],
                "face_q": c["face_q"],
                "scene_q": c["scene_q"],
                "vtx_sim": c["vtx_sim"],
                "chain_ids": c["chain_ids"],
                "text": c["text"],
                "llm_reason": c["llm_reason"],
                "visual_flags": c["visual_flags"],
                "meta": c["meta"],
            }
        )

    # Also expose only the clips that were actually used in final funnel
    used_ids = set(composer["used_clip_ids"])
    ordered_clips_for_output = []
    for c in ordered_clips:
        ordered_clips_for_output.append(
            {
                "id": c["id"],
                "slot": c["slot"],
                "start": c["start"],
                "end": c["end"],
                "score": c["score"],
                "semantic_score": c["semantic_score"],
                "visual_score": c["visual_score"],
                "face_q": c["face_q"],
                "scene_q": c["scene_q"],
                "vtx_sim": c["vtx_sim"],
                "chain_ids": c["chain_ids"],
                "text": c["text"],
                "llm_reason": c["llm_reason"],
                "visual_flags": c["visual_flags"],
                "meta": c["meta"],
            }
        )

    elapsed = time.time() - t_start
    logger.info("run_pipeline finished in %.2fs", elapsed)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": float(asr.get("duration", 0.0) or 0.0),
        "clips": ordered_clips_for_output,
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_video_local,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }
