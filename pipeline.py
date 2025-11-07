import os
import json
import time
import tempfile
from worker import asr, s3_utils

# ----------------------------------------------------
# micro semantic slot block
# ----------------------------------------------------
def micro_semantic_filter(text: str) -> bool:
    """
    Return True if we should KEEP this segment.
    Return False if we should DROP it (clearly broken).
    """
    if not text:
        return False

    low = text.lower().strip()

    # 1) Kill obvious blooper / redo lines
    BLOOPERS = [
        "wait i'm gonna say that right",
        "wait i don't",
        "why can't i remember",
        "let me say that again",
        "i'm gonna say that right",
    ]
    for b in BLOOPERS:
        if b in low:
            return False

    # 2) Kill super-short, meaningless fragments
    BAD_SHORT = {
        "must anyway",
        "i found",
        "i found.",
    }
    if low in BAD_SHORT:
        return False

    # 3) Soft-handle ASR-misspelled slang like “kuchigai”, “utas”
    # Drop only if the ENTIRE line is that nonsense, not if it's part of a real sentence
    BAD_TOKENS = ["kuchigai", "utas"]
    for tok in BAD_TOKENS:
        if low == tok:
            return False
        # if token is inside a real sentence, keep it
        # e.g., "your kuchigai won't..." should stay

    # ✅ Keep normal English + TikTok slang + full sentences
    return True


# ----------------------------------------------------
# main pipeline
# ----------------------------------------------------
def run_pipeline(local_video_path: str, s3_prefix: str, session_id: str):
    start_time = time.time()
    print(f"[pipeline] 🔹 starting run_pipeline for {local_video_path}")

    # --- STEP 1: Run ASR ---
    print("[pipeline] 🧠 Running ASR (dual mode)")
    asr_segments = []

    try:
        # Whisper result — returns list of dicts [{"text": ..., "start": ..., "end": ...}]
        asr_segments = asr.transcribe(local_video_path)
    except Exception as e:
        print(f"[pipeline] ❌ ASR failed: {e}")
        return {"ok": False, "error": str(e)}

    if not asr_segments:
        print("[pipeline] ⚠️ No ASR segments returned.")
        return {"ok": False, "error": "no asr segments"}

    print(f"[pipeline] ✅ got {len(asr_segments)} ASR segments")

    # --- STEP 2: Segment Filtering + Slot Detection ---
    clips = []
    for idx, seg in enumerate(asr_segments):
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue

        # 🔴 micro semantic gate
        if not micro_semantic_filter(seg_text):
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))
        low = seg_text.lower()

        # --- SLOT DETECTION ---
        slot = "FEATURE"
        if any(k in low for k in ["if you don't have", "why not", "did you know", "stop scrolling"]):
            slot = "HOOK"
        elif any(k in low for k in ["click the link", "grab yours", "get yours", "buy now", "shop now", "i left it for you"]):
            slot = "CTA"
        elif any(k in low for k in ["works", "see", "tried", "proven", "really good", "great quality", "i love"]):
            slot = "PROOF"

        clip = {
            "id": f"ASR{idx:04d}",
            "slot": slot,
            "start": start,
            "end": end,
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [f"ASR{idx:04d}"],
            "text": seg_text,
        }

        clips.append(clip)

    # --- STEP 3: Upload video to S3 ---
    try:
        s3_key = f"{s3_prefix}{session_id}_{os.urandom(8).hex()}.mp4"
        s3_url = s3_utils.upload_file_to_s3(local_video_path, s3_key)
        https_url = s3_url
    except Exception as e:
        print(f"[pipeline] ⚠️ Failed to upload to S3: {e}")
        s3_key = s3_url = https_url = None

    elapsed = time.time() - start_time
    print(f"[pipeline] ✅ finished in {elapsed:.2f}s")

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": float(asr_segments[-1].get("end", 0.0)),
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": clips,
        "slots": _organize_slots(clips),
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": elapsed,
    }


# ----------------------------------------------------
# helper: slot grouping
# ----------------------------------------------------
def _organize_slots(clips):
    slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    for c in clips:
        slots.setdefault(c["slot"], []).append(c)
    return slots
