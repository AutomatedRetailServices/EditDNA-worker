import os
import math
import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests
import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips

from . import llm  # worker.llm
from .vision_v3 import visual_brain  # V3 visual scoring

logger = logging.getLogger(__name__)

# Whisper model name (for ASR)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

# S3 output config
OUTPUT_BUCKET = os.getenv(
    "EDITDNA_OUTPUT_BUCKET",
    "script2clipshop-video-automatedretailservices",
)
OUTPUT_PREFIX = os.getenv(
    "EDITDNA_OUTPUT_PREFIX",
    "editdna/outputs",
)

# Composer config
MIN_CLIP_SCORE = float(os.getenv("EDITDNA_MIN_CLIP_SCORE", "0.5"))
MAX_FEATURE_CLIPS = int(os.getenv("EDITDNA_MAX_FEATURE_CLIPS", "8"))

# Score mixing weights
SEMANTIC_WEIGHT = float(os.getenv("EDITDNA_SEMANTIC_WEIGHT", "0.7"))
VISUAL_WEIGHT = float(os.getenv("EDITDNA_VISUAL_WEIGHT", "0.3"))

try:
    import whisper
except Exception as e:
    whisper = None
    logger.error(f"Whisper import failed: {e}")


@dataclass
class Clause:
    id: str
    start: float
    end: float
    text: str
    slot_hint: str = "STORY"  # HOOK / PROBLEM / FEATURE / PROOF / CTA / STORY
    face_q: float = 1.0
    scene_q: float = 1.0
    semantic_score: float = 0.0
    visual_score: float = 0.0
    vtx_sim: float = 0.0      # final combined score
    chain_ids: Optional[List[str]] = None
    llm_reason: str = ""      # why it got that score
    visual_flags: Optional[Dict[str, bool]] = None


_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if whisper is None:
        raise RuntimeError("whisper is not available; check requirements.")
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


def _ensure_tmp_dir() -> str:
    base = "/tmp/editdna"
    os.makedirs(base, exist_ok=True)
    return base


def _download_to_local(url: str, session_id: str) -> str:
    """
    Download the first input video to /tmp/editdna/<session_id>_input.mp4
    """
    base = _ensure_tmp_dir()
    local_path = os.path.join(base, f"{session_id}_input.mp4")

    logger.info("Downloading input video", extra={"url": url, "local_path": local_path})
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


def _get_duration_sec(path: str) -> float:
    try:
        with VideoFileClip(path) as clip:
            return float(clip.duration)
    except Exception as e:
        logger.warning(f"Failed to read duration: {e}")
        return 0.0


def _grab_frame_b64(path: str, t_sec: float) -> Optional[str]:
    """
    Grab a single RGB frame at time t_sec, JPEG-encode, return base64 string.
    If anything fails, return None (pipeline keeps going).
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _run_asr(path: str) -> List[Clause]:
    """
    Run Whisper on the video, return list of Clause objects.
    We keep segmentation as Whisper gives it.
    """
    model = _get_whisper_model()
    result = model.transcribe(path, fp16=False)
    segments = result.get("segments", []) or []

    clauses: List[Clause] = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        clause_id = f"ASR{idx:04d}_c1"

        # very simple slot hint:
        #   - first segment → HOOK
        #   - last segment  → CTA
        #   - others        → STORY/FEATURE
        if idx == 0:
            slot_hint = "HOOK"
        elif idx == len(segments) - 1:
            slot_hint = "CTA"
        else:
            slot_hint = "STORY"

        clauses.append(
            Clause(
                id=clause_id,
                start=start,
                end=end,
                text=text,
                slot_hint=slot_hint,
                face_q=1.0,
                scene_q=1.0,
                semantic_score=0.0,
                visual_score=0.0,
                vtx_sim=0.0,
                chain_ids=[clause_id],
                llm_reason="",
                visual_flags={"scene_jump": False, "motion_jump": False},
            )
        )
    return clauses


def _build_slots(clauses: List[Clause]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the `slots` dict similar to your prior JSON:
    {
      "HOOK": [...],
      "PROBLEM": [...],
      "FEATURE": [...],
      "PROOF": [...],
      "CTA": [...]
    }
    """
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for c in clauses:
        if c.slot_hint == "HOOK":
            key = "HOOK"
        elif c.slot_hint == "CTA":
            key = "CTA"
        elif c.slot_hint == "PROBLEM":
            key = "PROBLEM"
        elif c.slot_hint == "PROOF":
            key = "PROOF"
        else:
            key = "FEATURE"

        slots[key].append(
            {
                "id": c.id,
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "meta": {
                    "slot": key,
                    "score": c.vtx_sim,
                    "semantic_score": c.semantic_score,
                    "visual_score": c.visual_score,
                    "chain_ids": c.chain_ids or [c.id],
                },
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": c.vtx_sim,
                "has_product": False,
                "ocr_hit": 0,
                "visual_flags": c.visual_flags or {
                    "scene_jump": False,
                    "motion_jump": False,
                },
            }
        )

    return slots


def _pick_best_hook(slots: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    hooks = slots.get("HOOK") or []
    if not hooks:
        return None
    best = sorted(hooks, key=lambda h: h["meta"]["score"], reverse=True)[0]
    if best["meta"]["score"] < MIN_CLIP_SCORE:
        return None
    return best["id"]


def _pick_best_cta(slots: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    ctas = slots.get("CTA") or []
    if not ctas:
        return None
    best = sorted(ctas, key=lambda c: c["meta"]["score"], reverse=True)[0]
    if best["meta"]["score"] < MIN_CLIP_SCORE:
        return None
    return best["id"]


def _is_good_clause_for_funnel(c: Clause, min_semantic: float = 0.6, min_words: int = 4) -> bool:
    """
    Heuristic filter so bad / filler / broken lines don't end up in the final video.
    Keeps slang & spicy language as long as it's coherent and complete.
    """
    text = (c.text or "").strip()
    if not text:
        return False

    words = text.split()
    # 1) Require minimum semantic strength
    if c.semantic_score < min_semantic:
        return False

    # 2) Very short one-word / two-word fillers are usually trash
    if len(words) < min_words:
        return False

    lower = text.lower()

    # 3) Obvious partial/blooper cues
    if "..." in text and "wait" in lower:
        # e.g. "These pineapple flavored...wait."
        return False

    # 4) Generic filler that almost never sells
    filler_exact = {
        "okay.",
        "ok.",
        "yeah.",
        "wait.",
    }
    if lower in filler_exact:
        return False

    # 5) (Optional) brand-safety word blacklist for final video.
    #    For now, we don't ban slang like "wet wet" so TikTok tone stays.
    #    You can add hard-banned phrases here if needed.
    banned_contains: List[str] = [
        # "kuchigai stay",  # example: uncomment to force-drop this from final edit
    ]
    for bad in banned_contains:
        if bad in lower:
            return False

    return True


def _build_composer(
    clauses: List[Clause],
    slots: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], str, List[Clause]]:
    """
    Decide which clips to keep (HOOK, FEATURES, CTA), build a human-readable summary,
    and return:
      - composer dict
      - composer_human string
      - ordered list of Clause objects in final timeline order
    """
    id_to_clause: Dict[str, Clause] = {c.id: c for c in clauses}

    # HOOK: keep existing HOOK logic (we let spicy / imperfect hooks live)
    hook_id = _pick_best_hook(slots)

    # FEATURES: filter using semantic + structure heuristics
    raw_features = slots.get("FEATURE") or []
    feature_candidates: List[Tuple[str, float, float]] = []  # (id, score, start)

    for f in raw_features:
        cid = f["id"]
        meta = f.get("meta") or {}
        score = float(meta.get("score", 0.0))
        if score < MIN_CLIP_SCORE:
            continue
        clause = id_to_clause.get(cid)
        if not clause:
            continue
        if not _is_good_clause_for_funnel(clause):
            continue
        feature_candidates.append((cid, score, clause.start))

    # Sort by score desc, then by start time to stabilize order
    feature_candidates.sort(key=lambda x: (-x[1], x[2]))
    feature_ids = [cid for cid, _, _ in feature_candidates[:MAX_FEATURE_CLIPS]]

    # CTA: as before (score-based)
    cta_id = _pick_best_cta(slots)

    used_ids: List[str] = []
    ordered_clauses: List[Clause] = []

    # Final timeline: HOOK -> FEATURES (original temporal order) -> CTA
    if hook_id and hook_id in id_to_clause:
        used_ids.append(hook_id)
        ordered_clauses.append(id_to_clause[hook_id])

    feature_clauses = [id_to_clause[fid] for fid in feature_ids if fid in id_to_clause]
    feature_clauses = sorted(feature_clauses, key=lambda c: c.start)
    for fc in feature_clauses:
        if fc.id not in used_ids:
            used_ids.append(fc.id)
            ordered_clauses.append(fc)

    if cta_id and cta_id in id_to_clause:
        if cta_id not in used_ids:
            used_ids.append(cta_id)
            ordered_clauses.append(id_to_clause[cta_id])

    composer = {
        "hook_id": hook_id,
        "feature_ids": feature_ids,
        "cta_id": cta_id,
        "used_clip_ids": used_ids,
        "min_score": MIN_CLIP_SCORE,
    }

    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")

    if hook_id and hook_id in id_to_clause:
        hc = id_to_clause[hook_id]
        lines.append(f"HOOK ({hc.id}, score={hc.vtx_sim:.2f}):")
        lines.append(f'  "{hc.text}"\n')
    else:
        lines.append("HOOK: NONE\n")

    if feature_ids:
        lines.append("FEATURES (kept):")
        for fid in feature_ids:
            if fid not in id_to_clause:
                continue
            fc = id_to_clause[fid]
            lines.append(
                f'  - [{fc.id}] score={fc.vtx_sim:.2f} → "{fc.text}"'
            )
        lines.append("")
    else:
        lines.append("FEATURES: NONE\n")

    if cta_id and cta_id in id_to_clause:
        cc = id_to_clause[cta_id]
        lines.append(f'CTA ({cc.id}, score={cc.vtx_sim:.2f}):')
        lines.append(f'  "{cc.text}"\n')
    else:
        lines.append("CTA: NONE\n")

    lines.append("FINAL ORDER TIMELINE:")
    if ordered_clauses:
        for idx, c in enumerate(ordered_clauses, start=1):
            lines.append(f'{idx}) {c.id} → "{c.text}"')
    else:
        lines.append("(no clips selected)")

    lines.append("\n=====================================")

    composer_human = "\n".join(lines)
    return composer, composer_human, ordered_clauses


def _render_final_video(
    input_local: str,
    session_id: str,
    ordered_clauses: List[Clause],
) -> Tuple[str, Optional[str]]:
    """
    Render the final stitched video to /tmp/editdna/<session_id>_edit.mp4
    Upload to S3 and return (local_path, s3_url_or_none).
    """
    base = _ensure_tmp_dir()
    output_local = os.path.join(base, f"{session_id}_edit.mp4")

    if not ordered_clauses:
        logger.warning("No clauses selected for final edit; skipping render.")
        return output_local, None

    try:
        with VideoFileClip(input_local) as clip:
            subclips = []
            for c in ordered_clauses:
                start = max(0.0, c.start)
                end = min(float(clip.duration), c.end)
                if end <= start:
                    continue
                sub = clip.subclip(start, end)
                subclips.append(sub)

            if not subclips:
                logger.warning("No valid subclips generated; skipping render.")
                return output_local, None

            final = concatenate_videoclips(subclips, method="compose")
            final.write_videofile(
                output_local,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(base, f"{session_id}_temp_audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None,
            )
    except Exception as e:
        logger.exception(f"Failed to render final video: {e}")
        return output_local, None

    # Upload to S3
    try:
        s3 = boto3.client("s3")
        key = f"{OUTPUT_PREFIX}/{session_id}/final.mp4"
        logger.info(
            "Uploading final video to S3",
            extra={"bucket": OUTPUT_BUCKET, "key": key},
        )
        s3.upload_file(output_local, OUTPUT_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
        s3_url = f"https://{OUTPUT_BUCKET}.s3.amazonaws.com/{key}"
        return output_local, s3_url
    except Exception as e:
        logger.exception(f"Failed to upload final video to S3: {e}")
        return output_local, None


def run_pipeline(
    *,
    session_id: str,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry used by tasks.job_render.

    Steps:
    1) Download first input video from file_urls to /tmp
    2) Get duration
    3) Run Whisper ASR → segments (clauses)
    4) For each segment:
         - grab mid-frame (for LLM)
         - LLM semantic scoring
         - visual_brain V3 scoring
         - combine into final score
    5) Build clips + slots structure
    6) Run funnel composer (HOOK, FEATURES, CTA)
    7) Render final stitched video according to composer
    8) Upload to S3 and return output_video_url
    """
    if not file_urls:
        raise ValueError("run_pipeline: file_urls is required and must be non-empty")

    input_url = file_urls[0]
    logger.info(
        "Starting pipeline",
        extra={"session_id": session_id, "input_url": input_url},
    )

    input_local = _download_to_local(input_url, session_id)
    duration = _get_duration_sec(input_local)

    # 1) ASR → Clauses
    clauses = _run_asr(input_local)

    clips: List[Dict[str, Any]] = []

    # 2) Scoring loop: semantic (LLM) + visual (V3)
    for c in clauses:
        mid_t = (c.start + c.end) / 2.0
        frame_b64 = _grab_frame_b64(input_local, mid_t)

        # Semantic scoring (existing LLM call, unchanged signature)
        sem_score, reason = llm.score_clause_multimodal(
            text=c.text,
            frame_b64=frame_b64,
            slot_hint=c.slot_hint,
        )
        c.semantic_score = float(sem_score)
        c.llm_reason = reason

        # Visual scoring (new V3 brain)
        vis_score, vis_flags = visual_brain.score_segment(
            path=input_local,
            start=c.start,
            end=c.end,
        )
        c.visual_score = float(vis_score)
        c.visual_flags = vis_flags

        # Combine
        final_score = (
            SEMANTIC_WEIGHT * c.semantic_score
            + VISUAL_WEIGHT * c.visual_score
        )
        final_score = max(0.0, min(1.0, final_score))
        c.vtx_sim = final_score

        # Normalize slot for clip-level metadata
        if c.slot_hint in ("HOOK", "CTA", "PROBLEM", "FEATURE", "PROOF"):
            clip_slot = c.slot_hint
        else:
            clip_slot = "STORY"

        clips.append(
            {
                "id": c.id,
                "slot": clip_slot,
                "start": c.start,
                "end": c.end,
                "score": final_score,
                "semantic_score": c.semantic_score,
                "visual_score": c.visual_score,
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": final_score,
                "chain_ids": c.chain_ids or [c.id],
                "text": c.text,
                "llm_reason": c.llm_reason,
                "visual_flags": c.visual_flags,
            }
        )

    # 3) Build slots structure with updated scores
    slots = _build_slots(clauses)

    # 4) Composer
    composer, composer_human, ordered_clauses = _build_composer(clauses, slots)

    # 5) Render final stitched video and upload to S3
    output_video_local, output_video_url = _render_final_video(
        input_local=input_local,
        session_id=session_id,
        ordered_clauses=ordered_clauses,
    )

    return {
        "duration_sec": duration,
        "clips": clips,
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_video_local,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }
