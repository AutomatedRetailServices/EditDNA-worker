import os
import math
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import whisper
import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips

from . import llm
from . import vision_clip
from . import ocr
from . import object_detect


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

# Quality knobs
MIN_CLIP_SCORE = float(os.getenv("EDITDNA_MIN_CLIP_SCORE", "0.5"))
MAX_FEATURE_CLIPS = int(os.getenv("EDITDNA_MAX_FEATURE_CLIPS", "6"))
FORCE_HOOK_CTA = bool(int(os.getenv("EDITDNA_FORCE_HOOK_CTA", "0")))  # 0 or 1


@dataclass
class Clause:
    id: str
    start: float
    end: float
    text: str
    slot_hint: str = "STORY"
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0  # we reuse this as "LLM score"
    chain_ids: Optional[List[str]] = None

    # V2 visual + meta fields (filled later)
    visual_ok: bool = True
    visual_internal_sim: float = 1.0
    clip_vec: Optional[np.ndarray] = None
    ocr_hit: int = 0
    has_product: bool = False
    llm_reason: str = ""


_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


def _ensure_tmp_dir() -> str:
    root = os.getenv("EDITDNA_TMP_DIR", "/tmp/editdna")
    os.makedirs(root, exist_ok=True)
    return root


def _download_to_local(session_id: str, url: str) -> str:
    tmp_dir = _ensure_tmp_dir()
    local_path = os.path.join(tmp_dir, f"{session_id}_input.mp4")

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
    except Exception:
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
                vtx_sim=0.0,
                chain_ids=[clause_id],
            )
        )
    return clauses


def _apply_llm_scoring(path: str, clauses: List[Clause]) -> None:
    """
    For each Clause, grab a mid-frame, send text+frame to LLM judge,
    and fill vtx_sim (score) + llm_reason.
    """
    for c in clauses:
        mid_t = (c.start + c.end) / 2.0
        frame_b64 = _grab_frame_b64(path, mid_t)

        score, reason = llm.score_clause_multimodal(
            text=c.text,
            frame_b64=frame_b64,
            slot_hint=c.slot_hint,
        )

        c.vtx_sim = float(score)
        c.llm_reason = reason


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
    We keep ALL clauses here (even low-score) so you can inspect.
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
                    "chain_ids": c.chain_ids or [c.id],
                },
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": c.vtx_sim,
                "has_product": c.has_product,
                "ocr_hit": c.ocr_hit,
            }
        )

    return slots


def _select_funnel_clauses(clauses: List[Clause]) -> Dict[str, Any]:
    """
    Decide which clips to keep in the final funnel.
    Uses:
      - LLM score (vtx_sim)
      - visual_ok (from CLIP internal sim)
    """
    # Only consider clauses that scored at all
    scored = [c for c in clauses if c.vtx_sim is not None]

    # Split by slot hints
    all_hooks = [c for c in scored if c.slot_hint == "HOOK"]
    all_ctas = [c for c in scored if c.slot_hint == "CTA"]
    all_features = [c for c in scored if c.slot_hint not in ("HOOK", "CTA")]

    # Apply threshold + visual_ok for "good" ones
    hooks = [c for c in all_hooks if c.vtx_sim >= MIN_CLIP_SCORE and c.visual_ok]
    ctas = [c for c in all_ctas if c.vtx_sim >= MIN_CLIP_SCORE and c.visual_ok]
    features = [
        c for c in all_features if c.vtx_sim >= MIN_CLIP_SCORE and c.visual_ok
    ]

    # HOOK: best above threshold, or best overall if FORCE_HOOK_CTA
    hook: Optional[Clause] = None
    if hooks:
        hook = max(hooks, key=lambda c: c.vtx_sim)
    elif FORCE_HOOK_CTA and all_hooks:
        hook = max(all_hooks, key=lambda c: c.vtx_sim)

    # CTA: same idea
    cta: Optional[Clause] = None
    if ctas:
        cta = max(ctas, key=lambda c: c.vtx_sim)
    elif FORCE_HOOK_CTA and all_ctas:
        cta = max(all_ctas, key=lambda c: c.vtx_sim)

    # Features: strongest first
    features_sorted = sorted(
        features,
        key=lambda c: (-c.vtx_sim, c.start),
    )
    features_selected = features_sorted[:MAX_FEATURE_CLIPS]

    # Timeline: hook, features, cta → sorted by time
    timeline_clauses: List[Clause] = []
    if hook:
        timeline_clauses.append(hook)
    timeline_clauses.extend(features_selected)
    if cta:
        timeline_clauses.append(cta)

    timeline_clauses = sorted(timeline_clauses, key=lambda c: c.start)

    return {
        "hook": hook,
        "features": features_selected,
        "cta": cta,
        "timeline": timeline_clauses,
    }


def _render_timeline(input_local: str, session_id: str, timeline: List[Clause]) -> str:
    """
    Render final MP4 from the selected timeline clauses.
    Returns local path to edited file.
    """
    tmp_dir = _ensure_tmp_dir()
    out_path = os.path.join(tmp_dir, f"{session_id}_edit.mp4")

    if not timeline:
        # no clips picked, just return original for now
        return input_local

    with VideoFileClip(input_local) as base_clip:
        subclips = []
        for c in timeline:
            # Clip range safety
            start = max(0.0, c.start)
            end = max(start + 0.05, min(float(base_clip.duration), c.end))
            subclips.append(base_clip.subclip(start, end))

        final = concatenate_videoclips(subclips)
        final.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(tmp_dir, f"{session_id}_audio_temp.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )

    return out_path


def _upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    """
    Upload final video to S3 and return HTTPS URL.
    Requires:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION
      - EDITDNA_S3_BUCKET
      - EDITDNA_S3_PREFIX (e.g. 'editdna/outputs')
    """
    bucket = os.getenv("EDITDNA_S3_BUCKET")
    prefix = os.getenv("EDITDNA_S3_PREFIX", "editdna/outputs")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not bucket:
        return None

    key = f"{prefix}/{session_id}/final.mp4"

    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})

    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _compose_human_summary(funnel: Dict[str, Any]) -> str:
    parts = ["===== EDITDNA FUNNEL COMPOSER =====", ""]

    hook = funnel.get("hook")
    feats: List[Clause] = funnel.get("features") or []
    cta = funnel.get("cta")

    if hook:
        parts.append(f"HOOK ({hook.id}, score={hook.vtx_sim:.2f}):")
        parts.append(f'  "{hook.text}"')
        parts.append("")
    else:
        parts.append("HOOK: NONE")
        parts.append("")

    parts.append("FEATURES (kept):")
    if feats:
        for c in feats:
            parts.append(
                f"  - [{c.id}] score={c.vtx_sim:.2f} → \"{c.text}\""
            )
    else:
        parts.append("  (none)")
    parts.append("")

    if cta:
        parts.append(f"CTA ({cta.id}, score={cta.vtx_sim:.2f}):")
        parts.append(f'  "{cta.text}"')
        parts.append("")
    else:
        parts.append("CTA: NONE")
        parts.append("")

    # Timeline
    parts.append("FINAL ORDER TIMELINE:")
    timeline: List[Clause] = funnel.get("timeline") or []
    if timeline:
        for idx, c in enumerate(timeline, start=1):
            parts.append(f'{idx}) {c.id} → "{c.text}"')
    else:
        parts.append("  (no clips)")

    parts.append("")
    parts.append("=====================================")

    return "\n".join(parts)


def run_pipeline(
    *,
    session_id: str,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry used by tasks.job_render.

    Steps:
    1) Download first video URL to local
    2) Run Whisper ASR → segments
    3) Visual Brain V2:
         - CLIP internal coherence check
         - CTA heuristic (OCR stub)
         - product keyword heuristic
    4) LLM multimodal scorer for each clause
    5) Composer: pick HOOK / FEATURES / CTA
    6) Render timeline → edited MP4
    7) Upload to S3 and return output_video_url
    """
    if not file_urls:
        raise ValueError("file_urls must contain at least one video URL")

    source_url = file_urls[0]
    input_local = _download_to_local(session_id, source_url)
    duration = _get_duration_sec(input_local)

    # 1) ASR
    clauses = _run_asr(input_local)

    # 2) Visual Brain (V2) – scene coherence + product flags + CTA-ish detection
    vision_clip.enrich_clauses_with_vision(input_local, clauses)
    ocr.enrich_clauses_with_ocr(clauses)
    object_detect.enrich_clauses_with_product_flags(clauses)

    # 3) LLM scoring
    _apply_llm_scoring(input_local, clauses)

    # 4) Build clips structure (for debugging / analysis)
    clips: List[Dict[str, Any]] = []
    for c in clauses:
        clips.append(
            {
                "id": c.id,
                "slot": c.slot_hint,
                "start": c.start,
                "end": c.end,
                "score": c.vtx_sim,
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": c.vtx_sim,
                "chain_ids": c.chain_ids or [c.id],
                "text": c.text,
                "llm_reason": c.llm_reason,
                "visual_ok": c.visual_ok,
                "visual_internal_sim": c.visual_internal_sim,
                "has_product": c.has_product,
                "ocr_hit": c.ocr_hit,
            }
        )

    slots = _build_slots(clauses)

    # 5) Composer
    funnel = _select_funnel_clauses(clauses)
    composer_human = _compose_human_summary(funnel)

    hook = funnel.get("hook")
    feats: List[Clause] = funnel.get("features") or []
    cta = funnel.get("cta")
    timeline: List[Clause] = funnel.get("timeline") or []

    hook_id = hook.id if hook else None
    feature_ids = [c.id for c in feats]
    cta_id = cta.id if cta else None
    used_clip_ids = [c.id for c in timeline]

    # 6) Render
    output_video_local = _render_timeline(input_local, session_id, timeline)

    # 7) Upload
    output_video_url = _upload_to_s3(output_video_local, session_id)

    return {
        "duration_sec": duration,
        "clips": clips,
        "slots": slots,
        "composer": {
            "hook_id": hook_id,
            "feature_ids": feature_ids,
            "cta_id": cta_id,
            "used_clip_ids": used_clip_ids,
            "min_score": MIN_CLIP_SCORE,
        },
        "composer_human": composer_human,
        "output_video_local": output_video_local,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }
