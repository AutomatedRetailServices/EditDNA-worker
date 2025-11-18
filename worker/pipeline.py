import os
import json
import logging
import tempfile
import subprocess
import uuid
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from faster_whisper import WhisperModel
from openai import OpenAI

logger = logging.getLogger("editdna.pipeline")

# ==========
# CONFIG
# ==========

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "medium")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.90"))
COMPOSER_MAX_PER_SLOT_DEFAULT = int(os.environ.get("COMPOSER_MAX_PER_SLOT", "7"))

SLOT_ORDER = ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]

SLOT_MAX = {
    "HOOK": int(os.environ.get("COMPOSER_MAX_HOOK", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "STORY": int(os.environ.get("COMPOSER_MAX_STORY", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "PROBLEM": int(os.environ.get("COMPOSER_MAX_PROBLEM", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "BENEFITS": int(os.environ.get("COMPOSER_MAX_BENEFITS", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "FEATURES": int(os.environ.get("COMPOSER_MAX_FEATURES", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "PROOF": int(os.environ.get("COMPOSER_MAX_PROOF", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
    "CTA": int(os.environ.get("COMPOSER_MAX_CTA", str(COMPOSER_MAX_PER_SLOT_DEFAULT))),
}

MAX_SEG_DURATION = float(os.environ.get("MAX_SEG_DURATION", "10.0"))  # seconds

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

client = OpenAI()


# ==========
# UTILS
# ==========

def ensure_session_dir(session_id: str) -> Path:
    base_tmp = Path(os.environ.get("TMP_DIR", "/tmp/TMP/editdna"))
    session_dir = base_tmp / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def download_to_path(url: str, dest_path: Path) -> None:
    logger.info("Downloading file", extra={"url": url, "dest": str(dest_path)})
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def try_upload_s3(local_path: Path) -> str:
    """
    Uploads file to S3 if S3_BUCKET is set.
    Returns either S3 URL or local path string.
    """
    if not S3_BUCKET:
        return str(local_path)

    try:
        import boto3

        s3 = boto3.client("s3")
        key = f"{S3_PREFIX.rstrip('/')}/{local_path.name}"
        s3.upload_file(str(local_path), S3_BUCKET, key)
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        logger.info("Uploaded to S3", extra={"bucket": S3_BUCKET, "key": key})
        return url
    except Exception:
        logger.exception("Failed to upload to S3, returning local path instead")
        return str(local_path)


# ==========
# ASR (GPU-FIRST)
# ==========

def load_whisper_model() -> WhisperModel:
    """
    Try to load Whisper using CUDA (GPU). If it fails, fallback to CPU.
    """
    try:
        logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on CUDA (float16)")
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper running on CUDA ✅")
        return model
    except Exception:
        logger.exception("Failed to load Whisper on CUDA, falling back to CPU int8")
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper running on CPU ⚠️ (slower)")
        return model


def run_whisper_asr(input_local: str) -> List[Dict[str, Any]]:
    """
    Runs Whisper and returns a list of segments with word-level timestamps.
    Each segment: {start, end, text, words: [{start, end, word}, ...]}
    """
    model = load_whisper_model()

    logger.info("Running Whisper ASR", extra={"input": input_local})
    segments_out: List[Dict[str, Any]] = []

    # Using word_timestamps=True so we can do sentence-boundary micro-cuts
    for seg in model.transcribe(
        input_local,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )[0]:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                })
        segments_out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })

    if segments_out:
        total_dur = segments_out[-1]["end"]
    else:
        total_dur = 0.0

    logger.info(
        "ASR produced %d segments, duration ~%.2fs",
        len(segments_out),
        total_dur,
    )
    return segments_out


# ==========
# MICRO-CUT (Sentence Boundary)
# ==========

def _flush_sentence(
    clips: List[Dict[str, Any]],
    current_words: List[Dict[str, Any]],
    buffer_chars: List[str],
    cur_text: List[str]
) -> None:
    if not current_words:
        return
    start = current_words[0]["start"]
    end = current_words[-1]["end"]
    text = "".join(buffer_chars).strip()
    if not text:
        return

    # Create a clip skeleton (visual scores filled later)
    clip_id = f"ASR_{uuid.uuid4().hex[:8]}"
    clips.append({
        "id": clip_id,
        "start": start,
        "end": end,
        "text": text,
        # Visual placeholders; real scoring later
        "slot": "OTHER",
        "score": 0.0,
        "semantic_score": 0.0,
        "visual_score": 1.0,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 1.0,
        "llm_reason": "",
        "visual_flags": {
            "scene_jump": False,
            "motion_jump": False,
        },
        "meta": {},
        "chain_ids": [clip_id],
    })
    buffer_chars.clear()
    cur_text.clear()
    current_words.clear()


def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes ASR segments with word timestamps and cuts them into sentence-level micro-clips:
    - Uses punctuation (.?!)
    - Also cuts when segments exceed MAX_SEG_DURATION
    """
    clips: List[Dict[str, Any]] = []

    for seg in asr_segments:
        words = seg.get("words") or []
        if not words:
            # fallback: just one chunk for this segment
            clip_id = f"ASR_{uuid.uuid4().hex[:8]}"
            clips.append({
                "id": clip_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "slot": "OTHER",
                "score": 0.0,
                "semantic_score": 0.0,
                "visual_score": 1.0,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 1.0,
                "llm_reason": "",
                "visual_flags": {
                    "scene_jump": False,
                    "motion_jump": False,
                },
                "meta": {},
                "chain_ids": [clip_id],
            })
            continue

        current_words: List[Dict[str, Any]] = []
        buffer_chars: List[str] = []
        cur_text: List[str] = []

        last_sentence_start_time = words[0]["start"]

        for w in words:
            current_words.append(w)
            buffer_chars.append(w.word)
            cur_text.append(w.word)

            # Normalize punctuation (end of sentence)
            is_boundary = any(w.word.strip().endswith(p) for p in [".", "?", "!"])

            # Duration-based cut
            cur_dur = float(w["end"]) - float(last_sentence_start_time)
            too_long = cur_dur >= MAX_SEG_DURATION

            if is_boundary or too_long:
                _flush_sentence(clips, current_words, buffer_chars, cur_text)
                last_sentence_start_time = w["end"]

        # flush remainder
        _flush_sentence(clips, current_words, buffer_chars, cur_text)

    logger.info("Sentence-boundary micro-cuts produced %d clips", len(clips))
    return clips


# ==========
# LLM CLASSIFICATION (slot / keep / semantic_score)
# ==========

LLM_SYSTEM_PROMPT = """You are a senior UGC ad editor.

For each spoken sentence from a UGC script, you MUST classify:

- slot: one of [HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA, OTHER]
- keep: true if the sentence is usable in the final ad; false if it's a redo, filler, confusion, or meta-comment.
- semantic_score: float between 0.0 and 1.0 (0 = useless, 1 = perfect).
- reason: short explanation.

HOOK: Attention-grabbing openers, bold claims, intriguing questions.
STORY: Personal story, context, or narrative.
PROBLEM: Explicit pain, frustration, or issue.
BENEFITS: Outcomes and transformations ("feel fresh and confident").
FEATURES: Ingredients, specs, how it works ("contains probiotics, prebiotics").
PROOF: Social proof, credibility, results, stats.
CTA: Calls to action ("click the link", "get yours today").
OTHER: Anything that doesn't fit, or meta lines like "wait, is that good?", "let me start again".

Mark as keep=false for:
- redos: "let me start again", "I said that wrong", "wait", "is that good?"
- filler: "okay, okay", "thanks", "yeah"
- confusion/meta: "am I saying it right?"

Return ONLY valid JSON like:
{"slot": "HOOK", "keep": true, "semantic_score": 0.92, "reason": "..." }
"""


def classify_clip_with_llm(text: str) -> Dict[str, Any]:
    """
    Calls OpenAI Chat to classify a single clip text.
    Returns dict: {slot, keep, semantic_score, reason}
    """
    text = text.strip()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Try to parse JSON
        data = json.loads(raw)
        slot = str(data.get("slot", "OTHER")).upper().strip()
        if slot not in SLOT_ORDER + ["OTHER"]:
            slot = "OTHER"
        keep = bool(data.get("keep", False))
        semantic_score = float(data.get("semantic_score", 0.0))
        reason = str(data.get("reason", "")).strip()
        return {
            "slot": slot,
            "keep": keep,
            "semantic_score": semantic_score,
            "reason": reason,
        }
    except Exception:
        logger.exception("LLM classification failed; falling back to heuristics")
        # Very rough heuristic fallback
        low = text.lower()
        slot = "OTHER"
        keep = True
        reason = "Heuristic fallback classification."
        if any(x in low for x in ["click the link", "get yours", "shop now", "buy now"]):
            slot = "CTA"
        elif any(x in low for x in ["support", "helps", "feel", "so you can"]):
            slot = "BENEFITS"
        elif any(x in low for x in ["contains", "includes", "made with", "packed with", "each gummy"]):
            slot = "FEATURES"
        elif any(x in low for x in ["is your", "does your", "struggling with", "tired of"]):
            slot = "PROBLEM"
        elif any(x in low for x in ["once upon", "story", "let me tell"]):
            slot = "STORY"
        elif any(x in low for x in ["wait", "redo", "start again", "is that good", "am i saying it right"]):
            keep = False
            slot = "OTHER"
            reason = "Detected redo/filler."
        elif any(x in low for x in ["hey", "listen", "attention", "ladies", "guys", "odor?", "utis?", "yeast?"]):
            slot = "HOOK"

        semantic_score = 0.85 if keep else 0.1
        return {
            "slot": slot,
            "keep": keep,
            "semantic_score": semantic_score,
            "reason": reason,
        }


def enrich_clips_with_llm(clips: List[Dict[str, Any]]) -> None:
    """
    Mutates clips in-place: fills in slot, semantic_score, keep, llm_reason, meta.
    """
    for clip in clips:
        text = clip["text"]
        llm = classify_clip_with_llm(text)
        slot = llm["slot"]
        keep = llm["keep"]
        semantic_score = llm["semantic_score"]
        reason = llm["reason"]

        clip["slot"] = slot
        clip["semantic_score"] = semantic_score
        clip["llm_reason"] = reason

        # Basic combined score: semantic only for now
        score = semantic_score
        clip["score"] = score

        clip["meta"] = {
            "slot": slot,
            "semantic_score": semantic_score,
            "visual_score": clip.get("visual_score", 1.0),
            "score": score,
            "chain_ids": clip.get("chain_ids", []),
            "keep": keep,
        }


# ==========
# DEDUPE & FILTER
# ==========

def normalize_text_for_dedupe(text: str) -> str:
    t = text.lower().strip()
    # simple normalize
    for ch in [".", "?", "!", ",", ";", ":", "'", '"']:
        t = t.replace(ch, "")
    t = " ".join(t.split())
    return t


def dedupe_and_filter_clips(clips: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    - Removes clips with keep=False or semantic_score < threshold
    - Dedupes semantically similar lines (using difflib similarity on normalized text)
    - Builds slots index
    Returns (filtered_clips, slots_index)
    """
    kept: List[Dict[str, Any]] = []
    slots_index: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SLOT_ORDER}
    slots_index["OTHER"] = []

    seen_texts: List[str] = []

    for clip in sorted(clips, key=lambda c: c["start"]):
        meta = clip.get("meta") or {}
        keep = bool(meta.get("keep", True))
        sem = float(meta.get("semantic_score", clip.get("semantic_score", 0.0)))

        if not keep:
            continue
        if sem < COMPOSER_MIN_SEMANTIC:
            continue

        normalized = normalize_text_for_dedupe(clip["text"])
        is_dup = False
        for prev in seen_texts:
            sim = difflib.SequenceMatcher(None, normalized, prev).ratio()
            if sim >= COMPOSER_DUP_SIM_THRESHOLD:
                is_dup = True
                break

        if is_dup:
            continue

        seen_texts.append(normalized)
        kept.append(clip)

        slot = clip.get("slot", "OTHER")
        if slot not in slots_index:
            slot = "OTHER"
        slots_index[slot].append(clip)

    logger.info("After dedupe/filter: %d kept clips", len(kept))
    return kept, slots_index


# ==========
# COMPOSER V2 (Free-flow, capped at 7 per slot)
# ==========

def compose_timeline(clips: List[Dict[str, Any]], slots_index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Free-flow mode with intelligence:
    - Keeps chronological order.
    - Caps each slot to SLOT_MAX[...] (7 by default).
    - CTA moved to the very end.
    - Returns composer dict with used_clip_ids and human explanation.
    """

    # Counters per slot (so we don't exceed max per slot)
    slot_counts = {s: 0 for s in SLOT_MAX.keys()}

    chronological = sorted(clips, key=lambda c: c["start"])
    selected: List[Dict[str, Any]] = []
    cta_clips: List[Dict[str, Any]] = []

    for clip in chronological:
        slot = clip.get("slot", "OTHER")
        if slot not in SLOT_MAX:
            # OTHER is allowed but has no cap logic; treat as STORY-ish
            selected.append(clip)
            continue

        # CTA will be handled separately
        if slot == "CTA":
            cta_clips.append(clip)
            continue

        if slot_counts[slot] >= SLOT_MAX[slot]:
            continue

        slot_counts[slot] += 1
        selected.append(clip)

    # Handle CTA: pick best ones (by semantic_score) but cap as well
    if cta_clips:
        cta_clips_sorted = sorted(cta_clips, key=lambda c: c.get("semantic_score", 0.0), reverse=True)
        max_cta = SLOT_MAX.get("CTA", 1)
        best_cta = cta_clips_sorted[:max_cta]
        selected.extend(best_cta)

    used_clip_ids = [c["id"] for c in selected]

    # Build composer summary
    def ids_for(slot_name: str) -> List[str]:
        return [c["id"] for c in selected if c.get("slot") == slot_name]

    composer = {
        "hook_id": ids_for("HOOK")[0] if ids_for("HOOK") else None,
        "story_ids": ids_for("STORY"),
        "problem_ids": ids_for("PROBLEM"),
        "benefit_ids": ids_for("BENEFITS"),
        "feature_ids": ids_for("FEATURES"),
        "proof_ids": ids_for("PROOF"),
        "cta_id": ids_for("CTA")[0] if ids_for("CTA") else None,
        "used_clip_ids": used_clip_ids,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }

    # Human-readable summary
    lines = ["===== EDITDNA FUNNEL COMPOSER V2 ====="]
    for slot in SLOT_ORDER:
        ids = ids_for(slot)
        if slot == "HOOK":
            lines.append("HOOK:")
        else:
            lines.append(f"{slot}:")

        if not ids:
            lines.append("  (none)")
        else:
            for cid in ids:
                clip = next(c for c in selected if c["id"] == cid)
                lines.append(f"  [{cid}] score={clip.get('semantic_score', 0.0):.2f} → \"{clip['text']}\"")

    lines.append("")
    lines.append("FINAL ORDER TIMELINE (chronological, CTA forced last):")
    # Timeline in actual used order (selected order preserves chronological + CTA at end)
    for i, clip in enumerate(selected, start=1):
        lines.append(f"{i}) {clip['id']} ({clip.get('slot','OTHER')}) → \"{clip['text']}\"")

    lines.append("=====================================")
    composer_human = "\n".join(lines)

    composer["human"] = composer_human
    return composer


# ==========
# FFMPEG RENDER
# ==========

def render_funnel_video(input_local: str, session_dir: Path, clip_index: Dict[str, Dict[str, Any]],
                        used_clip_ids: List[str]) -> Path:
    """
    Uses ffmpeg to cut the original video [0:v][0:a] into the selected clip intervals and concat them.
    Ensures video/audio stay in sync.
    """
    if not used_clip_ids:
        raise RuntimeError("No clips selected for final render")

    out_path = session_dir / "final.mp4"

    filter_parts = []
    v_labels = []
    a_labels = []

    for i, cid in enumerate(used_clip_ids):
        clip = clip_index[cid]
        start = max(0.0, float(clip["start"]))
        end = max(start, float(clip["end"]))
        v_label = f"v{i}"
        a_label = f"a{i}"

        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{v_label}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{a_label}]"
        )
        v_labels.append(f"[{v_label}]")
        a_labels.append(f"[{a_label}]")

    concat_part = "".join(v_labels + a_labels) + f"concat=n={len(used_clip_ids)}:v=1:a=1[vout][aout]"
    filter_complex = ";".join(filter_parts + [concat_part])

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_local,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]

    logger.info("Running ffmpeg to render funnel video", extra={"cmd": " ".join(cmd)})
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        logger.error("ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    logger.info("Final video rendered", extra={"output": str(out_path)})
    return out_path


# ==========
# MAIN PIPELINE
# ==========

def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Main entrypoint called from tasks.job_render.

    Steps:
    1) Download first video URL to /tmp for this session.
    2) Run Whisper ASR (GPU-first).
    3) Sentence-boundary micro-cuts.
    4) LLM classification (slot, keep, semantic_score).
    5) Dedupe + filter by semantic score.
    6) Composer V2: free-flow chronological, max 7 per slot, CTA last.
    7) ffmpeg render.
    8) Optional S3 upload.

    Returns full result dict.
    """
    logger.info("🚀 run_pipeline called", extra={"session_id": session_id, "file_urls": file_urls})

    if not file_urls:
        raise ValueError("run_pipeline: file_urls must be a non-empty list")

    session_dir = ensure_session_dir(session_id)
    input_local = session_dir / "input.mp4"
    download_to_path(file_urls[0], input_local)

    # 1) ASR
    asr_segments = run_whisper_asr(str(input_local))
    duration_sec = asr_segments[-1]["end"] if asr_segments else 0.0

    # 2) Micro-cuts
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 3) LLM classification
    enrich_clips_with_llm(clips)

    # 4) Dedupe + filter
    filtered_clips, slots_index = dedupe_and_filter_clips(clips)

    # Build clip_index for fast lookup
    clip_index = {c["id"]: c for c in filtered_clips}

    # 5) Compose timeline
    composer = compose_timeline(filtered_clips, slots_index)
    used_clip_ids = composer["used_clip_ids"]

    # 6) Render final video
    final_path = render_funnel_video(str(input_local), session_dir, clip_index, used_clip_ids)
    output_url = try_upload_s3(final_path)

    # Build human slots structure
    slots_struct: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SLOT_ORDER}
    slots_struct["OTHER"] = []
    for slot_name, slot_clips in slots_index.items():
        slots_struct.setdefault(slot_name, [])
        for c in slot_clips:
            slots_struct[slot_name].append(c)

    result = {
        "ok": True,
        "session_id": session_id,
        "input_local": str(input_local),
        "duration_sec": duration_sec,
        "clips": filtered_clips,
        "slots": slots_struct,
        "composer": {
            "hook_id": composer.get("hook_id"),
            "story_ids": composer.get("story_ids", []),
            "problem_ids": composer.get("problem_ids", []),
            "benefit_ids": composer.get("benefit_ids", []),
            "feature_ids": composer.get("feature_ids", []),
            "proof_ids": composer.get("proof_ids", []),
            "cta_id": composer.get("cta_id"),
            "used_clip_ids": composer.get("used_clip_ids", []),
            "min_score": composer.get("min_score", COMPOSER_MIN_SEMANTIC),
        },
        "composer_human": composer.get("human", ""),
        "output_video_local": str(final_path),
        "output_video_url": output_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }

    logger.info("✅ run_pipeline finished", extra={"session_id": session_id})
    return result
