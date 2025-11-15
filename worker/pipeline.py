import os
import math
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import whisper
import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips

from . import llm  # worker.llm


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

# Simple composer knobs
MIN_CLIP_SCORE = float(os.getenv("EDITDNA_MIN_CLIP_SCORE", "0.5"))
MAX_FEATURE_CLIPS = int(os.getenv("EDITDNA_MAX_FEATURE_CLIPS", "6"))


@dataclass
class Clause:
    id: str
    start: float
    end: float
    text: str
    slot_hint: str = "STORY"
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    chain_ids: Optional[List[str]] = None


_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model


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

        # simple slot hint
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


def _build_slots(clauses: List[Clause]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the `slots` dict:
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
                "has_product": False,
                "ocr_hit": 0,
            }
        )

    return slots


def _select_funnel_clauses(clauses: List[Clause]) -> Dict[str, Any]:
    """
    Very simple funnel composer:
    - Best HOOK by score
    - Top N FEATURE/STORY clips by score
    - Best CTA by score
    Only clips with score >= MIN_CLIP_SCORE are allowed.
    """
    scored = [c for c in clauses if c.vtx_sim is not None]

    hooks = [c for c in scored if c.slot_hint == "HOOK" and c.vtx_sim >= MIN_CLIP_SCORE]
    ctas = [c for c in scored if c.slot_hint == "CTA" and c.vtx_sim >= MIN_CLIP_SCORE]
    features = [
        c
        for c in scored
        if c.slot_hint not in ("HOOK", "CTA") and c.vtx_sim >= MIN_CLIP_SCORE
    ]

    hook = max(hooks, key=lambda c: c.vtx_sim) if hooks else None
    cta = max(ctas, key=lambda c: c.vtx_sim) if ctas else None

    features_sorted = sorted(features, key=lambda c: (-c.vtx_sim, c.start))
    features_selected = features_sorted[:MAX_FEATURE_CLIPS]

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


def _render_funnel_video(
    input_local: str,
    timeline: List[Clause],
    session_id: str,
) -> Optional[str]:
    """
    Render a simple stitched video from the selected clauses.
    Returns the local output path, or None if anything fails.
    """
    if not timeline:
        return None

    try:
        os.makedirs("/tmp/editdna", exist_ok=True)
        out_path = os.path.join("/tmp/editdna", f"{session_id}_edit.mp4")

        base = VideoFileClip(input_local)
        subclips = []
        for c in timeline:
            start = max(0.0, c.start)
            end = max(start + 0.05, c.end)
            sub = base.subclip(start, end)
            subclips.append(sub)

        final = concatenate_videoclips(subclips, method="compose")
        final.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join("/tmp/editdna", f"{session_id}_temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )

        final.close()
        for s in subclips:
            s.close()
        base.close()

        return out_path
    except Exception:
        return None


def _upload_to_s3(local_path: str, s3_prefix: str, session_id: str) -> Optional[str]:
    """
    Upload the rendered video to S3 and return public URL.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket or not local_path or not os.path.exists(local_path):
        return None

    key = f"editdna/outputs/{s3_prefix or session_id}/final.mp4"

    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    except Exception:
        return None


# -----------------------------
# HUMAN-READABLE COMPOSER SUMMARY
# -----------------------------
def _make_human_readable_summary(funnel: Dict[str, Any]) -> str:
    lines = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")

    # HOOK
    hook = funnel.get("hook")
    if hook:
        lines.append(f"HOOK ({hook.id}, score={hook.vtx_sim:.2f}):")
        lines.append(f'  "{hook.text}"\n')
    else:
        lines.append("HOOK: NONE\n")

    # FEATURES
    lines.append("FEATURES (kept):")
    for f in funnel.get("features", []):
        lines.append(f"  - [{f.id}] score={f.vtx_sim:.2f} → \"{f.text}\"")
    lines.append("")

    # CTA
    cta = funnel.get("cta")
    if cta:
        lines.append(f"CTA ({cta.id}, score={cta.vtx_sim:.2f}):")
        lines.append(f'  "{cta.text}"\n')
    else:
        lines.append("CTA: NONE\n")

    # FINAL ORDER
    lines.append("FINAL ORDER TIMELINE:")
    for i, c in enumerate(funnel.get("timeline", []), start=1):
        lines.append(f"{i}) {c.id} → \"{c.text}\"")
    lines.append("\n=====================================")

    return "\n".join(lines)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(
    *,
    input_local: str,
    session_id: str,
    s3_prefix: str,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main pipeline:
    1) ASR
    2) LLM scoring
    3) Slots
    4) Composer
    5) Rendering
    6) S3 upload
    7) Human summary
    """
    duration = _get_duration_sec(input_local)
    clauses = _run_asr(input_local)

    clips: List[Dict[str, Any]] = []

    # 3) score each clause
    for c in clauses:
        mid_t = (c.start + c.end) / 2.0
        frame_b64 = _grab_frame_b64(input_local, mid_t)

        score, reason = llm.score_clause_multimodal(
            text=c.text,
            frame_b64=frame_b64,
            slot_hint=c.slot_hint,
        )

        c.vtx_sim = score

        clips.append(
            {
                "id": c.id,
                "slot": c.slot_hint,
                "start": c.start,
                "end": c.end,
                "score": score,
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": score,
                "chain_ids": c.chain_ids or [c.id],
                "text": c.text,
                "llm_reason": reason,
            }
        )

    # 4) build slots
    slots = _build_slots(clauses)

    # 5) compose
    funnel = _select_funnel_clauses(clauses)
    timeline = funnel["timeline"]
    used_clip_ids = [c.id for c in timeline]

    # human-readable summary
    composer_human = _make_human_readable_summary(funnel)

    # 6) render
    output_local = _render_funnel_video(
        input_local=input_local,
        timeline=timeline,
        session_id=session_id,
    )

    # 7) upload
    output_url = None
    if output_local:
        output_url = _upload_to_s3(
            local_path=output_local,
            s3_prefix=s3_prefix,
            session_id=session_id,
        )

    return {
        "duration_sec": duration,
        "clips": clips,
        "slots": slots,
        "composer": {
            "hook_id": funnel["hook"].id if funnel["hook"] else None,
            "feature_ids": [f.id for f in funnel["features"]],
            "cta_id": funnel["cta"].id if funnel["cta"] else None,
            "used_clip_ids": used_clip_ids,
            "min_score": MIN_CLIP_SCORE,
        },
        "composer_human": composer_human,
        "output_video_local": output_local,
        "output_video_url": output_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }
