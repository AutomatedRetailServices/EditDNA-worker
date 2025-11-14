import os
import math
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import whisper
from moviepy.editor import VideoFileClip

from . import llm  # worker.llm


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")


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
        # BGR -> RGB not strictly needed for JPEG; we just encode
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
                vtx_sim=0.0,
                chain_ids=[clause_id],
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
        else:
            # put everything else as FEATURE for now
            key = "FEATURE"

        slots[key].append(
            {
                "id": c.id,
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "meta": {
                    "slot": key,
                    "score": c.vtx_sim,  # we will overwrite later with LLM score
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


def run_pipeline(
    *,
    input_local: str,
    session_id: str,
    s3_prefix: str,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry used by tasks.job_render.

    Steps:
    1) Get duration
    2) Run Whisper ASR → segments
    3) For each segment: grab mid-frame, send to LLM judge
    4) Build clips + slots structure
    """
    duration = _get_duration_sec(input_local)
    clauses = _run_asr(input_local)

    clips: List[Dict[str, Any]] = []

    for c in clauses:
        mid_t = (c.start + c.end) / 2.0
        frame_b64 = _grab_frame_b64(input_local, mid_t)

        # Ask LLM to judge this sentence (with optional frame)
        score, reason = llm.score_clause_multimodal(
            text=c.text,
            frame_b64=frame_b64,
            slot_hint=c.slot_hint,
        )

        c.vtx_sim = score  # reuse this field to store "LLM score"

        clips.append(
            {
                "id": c.id,
                "slot": "STORY",
                "start": c.start,
                "end": c.end,
                "score": score,
                "face_q": c.face_q,
                "scene_q": c.scene_q,
                "vtx_sim": score,
                "chain_ids": c.chain_ids or [c.id],
                "text": c.text,
            }
        )

    slots = _build_slots(clauses)

    return {
        "duration_sec": duration,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,  # we used text + LLM
        "vision": True,    # we attempted to use frames
    }
