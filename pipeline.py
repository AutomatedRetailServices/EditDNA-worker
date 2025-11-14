import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from .asr import transcribe  # returns list[{id,start,end,text,slot?}]
from .vision import analyze_frames  # returns {seg_id:{face_q,scene_q,vtx_sim}}
from .llm import filter_segments_with_llm  # returns {seg_id:{keep_score,brief_reason}}
from .utils import ffprobe_duration, now_utc_iso, ensure_float, MAX_TAKE_SEC

# Hard safety only (not a target). Keep generous so long inputs can breathe.
HARD_CAP_SEC = 120.0

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    slot: str = "STORY"
    face_q: float = 0.0
    scene_q: float = 0.0
    vtx_sim: float = 0.0

def _rank_score(keep: float, face_q: float, scene_q: float) -> float:
    """Weighted ranking for segment selection."""
    return keep * 0.70 + face_q * 0.15 + scene_q * 0.15

def _dynamic_target(media_duration: float, payload: Dict[str, Any]) -> float:
    """
    Fluid target derived from input duration.
    - If caller *explicitly* passes target_duration_sec in payload, respect it.
    - Else compute:
        < 15s raw     -> target = raw (keep it all)
        15–90s raw    -> target ≈ 0.80 * raw
        > 90s raw     -> cap near 90s
    """
    # Optional explicit override if you ever want it:
    td = payload.get("target_duration_sec")
    if td is not None:
        try:
            td = float(td)
            return max(5.0, min(td, HARD_CAP_SEC))
        except Exception:
            pass

    md = float(media_duration or 0.0)
    if md <= 0.0:
        return 30.0  # unknown length: sane default
    if md < 15.0:
        return md
    if md <= 90.0:
        return md * 0.80
    return 90.0

def _min_out_sec(dynamic_target: float) -> float:
    """
    Minimum acceptable assembled length so we don't collapse too short.
    Aim for ~60% of dynamic target, clamped to [10s, 25s].
    """
    return max(10.0, min(25.0, dynamic_target * 0.60))

def assemble_clips(
    asr_segments: List[Dict[str, Any]],
    llm_decisions: Dict[str, Dict[str, Any]],
    vision_scores: Dict[str, Dict[str, float]],
    payload: Dict[str, Any],
    media_duration: float,
) -> List[Dict[str, Any]]:
    target = _dynamic_target(media_duration, payload)
    min_out = _min_out_sec(target)

    # 1) Rank segments
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for seg in asr_segments or []:
        sid = seg.get("id")
        keep = float((llm_decisions.get(sid, {}) or {}).get("keep_score", 0.0))
        v = vision_scores.get(sid, {}) or {}
        face_q = float(v.get("face_q", 0.0))
        scene_q = float(v.get("scene_q", 0.0))
        score = _rank_score(keep, face_q, scene_q)
        ranked.append((score, seg))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # 2) Greedy fill toward dynamic target
    clips: List[Dict[str, Any]] = []
    total = 0.0
    for _, seg in ranked:
        s = ensure_float(seg.get("start", 0.0))
        e = ensure_float(seg.get("end", 0.0))
        dur = max(0.0, e - s)
        if dur <= 0:
            continue
        if total + dur > HARD_CAP_SEC:
            break
        sid = seg.get("id") or f"SEG_{len(clips)}"
        v = vision_scores.get(sid, {}) or {}
        clips.append({
            "id": sid,
            "slot": seg.get("slot") or "STORY",
            "start": s,
            "end": e,
            "text": seg.get("text", ""),
            "face_q": float(v.get("face_q", 0.0)),
            "scene_q": float(v.get("scene_q", 0.0)),
            "vtx_sim": float(v.get("vtx_sim", 0.0)),
        })
        total += dur
        if total >= target:
            break

    # 3) Progressive fallback (NO 5s hard cap ever)
    if total < min_out:
        # Fallback A: pick longest readable ASR segments (ignore LLM)
        alt = sorted(
            asr_segments or [],
            key=lambda s: max(0.0, ensure_float(s.get("end", 0.0)) - ensure_float(s.get("start", 0.0))),
            reverse=True,
        )
        clips, total = [], 0.0
        for seg in alt:
            s = ensure_float(seg.get("start", 0.0))
            e = ensure_float(seg.get("end", 0.0))
            dur = max(0.0, e - s)
            if dur <= 0:
                continue
            sid = seg.get("id") or f"SEG_{len(clips)}"
            v = vision_scores.get(sid, {}) or {}
            clips.append({
                "id": sid,
                "slot": seg.get("slot") or "STORY",
                "start": s,
                "end": e,
                "text": seg.get("text", ""),
                "face_q": float(v.get("face_q", 0.0)),
                "scene_q": float(v.get("scene_q", 0.0)),
                "vtx_sim": float(v.get("vtx_sim", 0.0)),
            })
            total += dur
            if total >= min_out:
                break

    # 4) Last resort: raw window up to min(target, media)
    if not clips:
        end_t = max(10.0, min(float(media_duration or 0.0), float(target), HARD_CAP_SEC))
        clips = [{
            "id": "RAW_WINDOW",
            "slot": "STORY",
            "start": 0.0,
            "end": end_t,
            "text": "",
            "face_q": 0.0,
            "scene_q": 0.0,
            "vtx_sim": 0.0,
        }]

    return clips

def _clip_meta(c: Dict[str, Any], slot: str) -> Dict[str, Any]:
    return {
        "id": c["id"],
        "start": c["start"],
        "end": c["end"],
        "text": c.get("text", ""),
        "meta": {
            "slot": slot,
            "score": 0.0,
            "chain_ids": [c["id"]],
        },
        "face_q": c.get("face_q", 0.0),
        "scene_q": c.get("scene_q", 0.0),
        "vtx_sim": c.get("vtx_sim", 0.0),
        "has_product": False,
        "ocr_hit": 0,
    }

def _slots_from_clips(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simple placeholder slotting:
      - first  clip -> HOOK
      - middle clips -> FEATURE
      - last  clip -> CTA
    Replace with your real slotter if you have one.
    """
    slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    if not clips:
        return slots
    if len(clips) >= 1:
        slots["HOOK"].append(_clip_meta(clips[0], "HOOK"))
    for c in clips[1:-1]:
        slots["FEATURE"].append(_clip_meta(c, "FEATURE"))
    if len(clips) >= 2:
        slots["CTA"].append(_clip_meta(clips[-1], "CTA"))
    return slots

def run_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates: ASR -> Vision -> LLM -> Assembly (fluid duration).
    payload keys: { input_url or input_local, (optional) target_duration_sec }
    """
    payload = payload or {}
    input_local = payload.get("input_local")
    input_url = payload.get("input_url")
    if not input_local and not input_url:
        raise ValueError("payload requires input_local or input_url")

    media_path = input_local or input_url
    media_duration = ffprobe_duration(media_path)

    # 1) ASR
    asr_segments = transcribe(media_path)  # [{id,start,end,text,slot?}]
    # 2) Vision
    vision_scores = analyze_frames(media_path, asr_segments)
    # 3) LLM (always-on scoring; soft-keeps on errors)
    llm_decisions = filter_segments_with_llm(asr_segments, media_path)

    # 4) Assemble (fluid)
    clips = assemble_clips(asr_segments, llm_decisions, vision_scores, payload, media_duration)
    dur = sum(max(0.0, ensure_float(c["end"]) - ensure_float(c["start"])) for c in clips)

    # 5) Slots
    slots = _slots_from_clips(clips)

    result = {
        "ok": True,
        "session_id": payload.get("session_id", "funnel-test-1"),
        "input_local": input_local or "",
        "duration_sec": round(dur, 3),
        "s3_key": payload.get("s3_key", ""),
        "s3_url": payload.get("s3_url", ""),
        "https_url": payload.get("https_url", ""),
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": True,
        "ts": now_utc_iso(),
    }
    return result
