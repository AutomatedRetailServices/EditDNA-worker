# /workspace/EditDNA-worker/pipeline.py
import os
import re
import uuid
import json
import tempfile
from typing import List, Dict, Any

from worker import asr   # your asr.py that returns segments
from worker import s3    # your s3 helper that uploads
from worker import video # your ffmpeg / concat helpers


# ---------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------
def _tmpfile(suffix=".mp4"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


# ---------------------------------------------------------------------
# HARD BLOCKLIST:
# things that very clearly mean “bad take / restart”
# THIS is what your last run was full of (“wait i’m gonna say that right”)
# ---------------------------------------------------------------------
HARD_BAD_PHRASES = [
    "wait i'm gonna say that right",
    "wait i’m gonna say that right",
    "wait i'm gonna say that again",
    "wait i don't",
    "wait i dont",
    "why can't i remember",
    "why cant i remember",
    "i'm gonna say that right",
    "im gonna say that right",
    "i'm gonna say that again",
    "that one good i think they're really good",  # that weird line your ASR keeps giving
    "moisture control odor control wait",         # the messed-up line in last JSON
]

# words that on their own are suspicious but we only drop if the line is short
SOFT_BAD_TOKENS = [
    "wait",
    "let me say that again",
    "i'm gonna",
    "im gonna",
    "i don't",
    "i dont",
]

# ---------------------------------------------------------------------
# this is to NOT kill normal slang / tiktok talk
# we just let these pass even if they look casual
# ---------------------------------------------------------------------
SLANG_OK = [
    "for the girls",
    "the girls only",
    "this is for the girls",
    "wet wet",
]


def is_hard_bad(text: str) -> bool:
    t = _norm(text)
    for bad in HARD_BAD_PHRASES:
        if bad in t:
            return True
    return False


def is_slang_ok(text: str) -> bool:
    t = _norm(text)
    for s in SLANG_OK:
        if s in t:
            return True
    return False


def is_soft_bad(text: str) -> bool:
    """
    Short clause + contains a soft bad token = we drop it.
    This is to catch e.g. "wait" / "i don't..." fragments.
    """
    t = _norm(text)
    if len(t) < 15:  # very short
        for tok in SOFT_BAD_TOKENS:
            if tok in t:
                return True
    return False


def classify_clause(text: str) -> str:
    """
    Very simple slot classification like before.
    """
    t = _norm(text)
    # CTA-ish
    if "click the link" in t or "get yours today" in t or "grab one" in t:
        return "CTA"
    # Hook-y
    if "why not" in t or "worry no more" in t or "because i found" in t:
        return "HOOK"
    # Feature-ish
    return "FEATURE"


# ---------------------------------------------------------------------
# main entry point used by tasks.job_render()
# ---------------------------------------------------------------------
def run_pipeline(local_video_path: str,
                 session_id: str,
                 s3_prefix: str = "editdna/outputs/") -> Dict[str, Any]:
    """
    1. run ASR -> list of segments with (start, end, text)
    2. split those segments into clauses (we already saw that in your last JSON)
    3. filter out obvious bad / restarts
    4. slot them (HOOK / FEATURE / CTA)
    5. upload final video (you already rendered it earlier) – here we just return metadata
    """
    # 1) ASR
    asr_result = asr.transcribe(local_video_path)
    # expected: asr_result = [{"start": ..., "end": ..., "text": "..."}]

    all_clauses: List[Dict[str, Any]] = []
    for seg in asr_result:
        seg_text = seg["text"].strip()
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # naive clause split: on " , " and " . " and " and "
        # (we're keeping it simple so it matches what you're running now)
        raw = re.split(r"(?:,| and |\. )", seg_text)
        # but we need to map time across subclauses
        seg_duration = max(0.001, seg_end - seg_start)
        per_clause = seg_duration / max(1, len(raw))

        for i, piece in enumerate(raw):
            piece = piece.strip()
            if not piece:
                continue

            c_start = seg_start + i * per_clause
            c_end = min(seg_end, c_start + per_clause)

            all_clauses.append({
                "id": f"{seg.get('id','ASR')}_c{i+1}",
                "start": c_start,
                "end": c_end,
                "text": piece,
            })

    # 2) filter
    cleaned: List[Dict[str, Any]] = []
    for c in all_clauses:
        txt = c["text"]

        # keep slang even if looks weird
        if is_slang_ok(txt):
            cleaned.append(c)
            continue

        # kill obviously bad / restart clauses
        if is_hard_bad(txt):
            continue

        if is_soft_bad(txt):
            continue

        # also kill lines that start with "these pineapple flavored wait"
        if _norm(txt).startswith("these pineapple flavored wait"):
            continue

        # also kill lines that contain two "wait" words
        if _norm(txt).count("wait") >= 2:
            continue

        cleaned.append(c)

    # 3) slotting
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for c in cleaned:
        slot = classify_clause(c["text"])
        c_out = {
            "id": c["id"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "meta": {
                "slot": slot,
                "score": 2.5,
                "chain_ids": [c["id"]],
            },
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "has_product": False,
            "ocr_hit": 0,
        }
        slots.setdefault(slot, []).append(c_out)

    # 4) build final clips list (what your API returns)
    clips_out = []
    for c in cleaned:
        clips_out.append({
            "id": c["id"],
            "slot": classify_clause(c["text"]),
            "start": c["start"],
            "end": c["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [c["id"]],
            "text": c["text"],
        })

    # 5) upload final rendered video (you already rendered to local_video_path earlier)
    # we keep the same pattern as your logs
    basename = f"{uuid.uuid4().hex}_{int(float(os.path.getmtime(local_video_path)))}.mp4"
    s3_key = s3_prefix.rstrip("/") + "/" + basename
    s3_url = s3.upload_file(local_video_path, s3_key)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": video.get_duration(local_video_path),
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": s3_url,
        "clips": clips_out,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
