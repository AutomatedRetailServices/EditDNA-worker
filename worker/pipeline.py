import os
import re
import json
import math
import uuid
import time
import logging
import string
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------
# Optional imports (safe fallbacks)
# -----------------------------
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import boto3
    from botocore.client import Config as BotoConfig
except Exception:
    boto3 = None
    BotoConfig = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# Global config & helpers
# -----------------------------

TMP_ROOT = Path(os.getenv("DOWNLOAD_ROOT", "/tmp/editdna")).expanduser()
TMP_ROOT.mkdir(parents=True, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs")
S3_ACL = os.getenv("S3_ACL", "public-read")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "86400"))

VETO_MIN_SCORE = float(os.getenv("VETO_MIN_SCORE", "0.40"))
VIZ_MERGE_SIM = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN", "200"))

EDITDNA_MIN_CLIP_SCORE = float(os.getenv("EDITDNA_MIN_CLIP_SCORE", "0.70"))
EDITDNA_HOOK_MIN_SCORE = float(os.getenv("EDITDNA_HOOK_MIN_SCORE", "0.70"))
EDITDNA_CTA_MIN_SCORE = float(os.getenv("EDITDNA_CTA_MIN_SCORE", "0.60"))

W_SEM = float(os.getenv("W_SEM", "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE", "0.5"))
W_VTX = float(os.getenv("W_VTX", "0.8"))
W_VISION = float(os.getenv("W_VISION", "0.7"))

ASR_ENABLED = os.getenv("ASR_ENABLED", "1") == "1"
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
ASR_DOWNLOAD_ROOT = os.getenv("ASR_DOWNLOAD_ROOT", "/workspace/.cache/whisper")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

BIN_SEC = float(os.getenv("BIN_SEC", "1.0"))
MIN_TAKE_SEC = float(os.getenv("MIN_TAKE_SEC", "2.0"))
MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "12.0"))

TARGET_DURATION_SEC = float(os.getenv("TARGET_DURATION_SEC", "0.0"))

OPENAI_MODEL = os.getenv("EDITDNA_LLM_MODEL", "gpt-4.1-mini")

if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai_client = None


def _normalize_text(t: str) -> str:
    t = t.strip().lower()
    t = t.replace("’", "'")
    table = str.maketrans("", "", string.punctuation)
    t = t.translate(table)
    t = re.sub(r"\s+", " ", t)
    return t


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Data model
# -----------------------------

@dataclass
class ClipSegment:
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

    def to_clip_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slot": self.slot,
            "start": float(self.start),
            "end": float(self.end),
            "score": float(self.score),
            "semantic_score": float(self.semantic_score),
            "visual_score": float(self.visual_score),
            "face_q": float(self.face_q),
            "scene_q": float(self.scene_q),
            "vtx_sim": float(self.vtx_sim),
            "chain_ids": list(self.chain_ids),
            "text": self.text,
            "llm_reason": self.llm_reason,
            "visual_flags": dict(self.visual_flags),
        }


# -----------------------------
# Download input video
# -----------------------------

def _download_first_file(session_id: str, file_urls: List[str]) -> Tuple[Path, float]:
    """
    Download the FIRST url in file_urls to a local file and return (path, duration_sec).

    Duration is best-effort (from ffprobe if available; else 0.0).
    """
    if not file_urls:
        raise ValueError("file_urls must be non-empty")

    import urllib.request

    url = file_urls[0]
    job_dir = TMP_ROOT / session_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "input.mp4"

    logger.info(f"⬇️ Downloading source video: {url} -> {out_path}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Try to get duration using ffprobe CLI (installed in the image)
    duration = 0.0
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(out_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration = float(out.decode("utf-8", "ignore").strip())
    except Exception as e:
        logger.warning(f"ffprobe failed to get duration: {e}")

    return out_path, duration


# -----------------------------
# ASR via faster-whisper (GPU when available)
# -----------------------------

def _run_whisper_asr(input_path: Path) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run local Whisper (faster-whisper) on GPU when available.
    Returns (segments, duration_sec) where segments is a list of:
      { "start": float, "end": float, "text": str }
    """
    if not ASR_ENABLED:
        raise RuntimeError("ASR is disabled (ASR_ENABLED != 1)")

    if WhisperModel is None:
        raise RuntimeError("faster_whisper is not installed in this image")

    logger.info(
        f"🎙  Running Whisper ASR model={WHISPER_MODEL} device={ASR_DEVICE} "
        f"compute={ASR_COMPUTE_TYPE}"
    )

    model = WhisperModel(
        WHISPER_MODEL,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
        download_root=ASR_DOWNLOAD_ROOT,
    )

    segments_out: List[Dict[str, Any]] = []
    t0 = time.time()
    seg_gen, info = model.transcribe(
        str(input_path),
        language=ASR_LANGUAGE,
        beam_size=5,
        word_timestamps=False,
    )

    for seg in seg_gen:
        segments_out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
            }
        )

    dt = time.time() - t0
    logger.info(f"✅ Whisper finished: {len(segments_out)} segments in {dt:.2f}s")

    duration_sec = float(getattr(info, "duration", 0.0) or 0.0)
    return segments_out, duration_sec


# -----------------------------
# Micro-cut segmentation
# -----------------------------

def _build_micro_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn raw ASR segments into micro-segments of roughly BIN_SEC buckets,
    ensuring each micro segment is within [MIN_TAKE_SEC, MAX_TAKE_SEC].
    """
    if not raw_segments:
        return []

    micro: List[Dict[str, Any]] = []
    cur_start = raw_segments[0]["start"]
    cur_end = raw_segments[0]["end"]
    cur_text_parts: List[str] = []

    def flush():
        nonlocal micro, cur_start, cur_end, cur_text_parts
        text = " ".join(p.strip() for p in cur_text_parts if p.strip()).strip()
        if not text:
            return
        seg = {
            "start": float(cur_start),
            "end": float(cur_end),
            "text": text,
        }
        micro.append(seg)
        cur_text_parts = []

    for seg in raw_segments:
        s = float(seg["start"])
        e = float(seg["end"])
        t = (seg.get("text") or "").strip()
        if not t:
            continue

        # if adding this segment makes current segment too long, flush first
        if (e - cur_start) > MAX_TAKE_SEC:
            flush()
            cur_start, cur_end = s, e
            cur_text_parts = [t]
        else:
            # extend current window
            cur_end = e
            cur_text_parts.append(t)

        # If small but we've crossed BIN_SEC and at least MIN_TAKE_SEC, flush
        if (cur_end - cur_start) >= MIN_TAKE_SEC and (cur_end - cur_start) >= BIN_SEC:
            flush()
            cur_start, cur_end = e, e
            cur_text_parts = []

    # final flush
    flush()

    # filter out too-short clips
    micro = [
        m for m in micro if (m["end"] - m["start"]) >= MIN_TAKE_SEC
    ]

    logger.info(f"✂️ Micro segments built: {len(micro)}")
    return micro
# =========================================================
# PART 2 of 4 — ASR + micro-cuts + vision + LLM scoring
# =========================================================

# -----------------------------
# Whisper (ASR)
# -----------------------------

_whisper_model_cache = None

def load_whisper():
    """
    Lazy-load Whisper model ONCE.
    GPU if ASR_DEVICE="cuda", otherwise CPU.
    """
    global _whisper_model_cache
    if _whisper_model_cache is not None:
        return _whisper_model_cache

    import whisper

    device = ASR_DEVICE if torch.cuda.is_available() else "cpu"
    logger.info(f"[ASR] Loading Whisper model '{WHISPER_MODEL}' on {device}")

    _whisper_model_cache = whisper.load_model(
        WHISPER_MODEL,
        device=device,
        download_root=ASR_DOWNLOAD_ROOT
    )
    return _whisper_model_cache


def run_asr(local_video_path: str) -> List[Dict[str, Any]]:
    """
    Returns list of segments:
        [{ "start": float, "end": float, "text": str }, ...]
    """
    if not ASR_ENABLED:
        logger.warning("ASR is disabled — returning empty segments.")
        return []

    model = load_whisper()
    logger.info(f"[ASR] Running Whisper on {local_video_path}")

    result = model.transcribe(
        local_video_path,
        language=ASR_LANGUAGE,
        verbose=False
    )

    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"].strip()
        })

    logger.info(f"[ASR] {len(segments)} segments extracted")
    return segments


# -----------------------------
# Micro filler detection
# -----------------------------

def is_filler_word(word: str) -> bool:
    w = word.lower().strip(",.!? ")
    return w in SEM_FILLER_LIST


def compute_filler_rate(text: str) -> float:
    """
    Returns fraction of words that are filler.
    """
    words = text.split()
    if not words:
        return 0.0
    fillers = sum(1 for w in words if is_filler_word(w))
    return fillers / len(words)


# -----------------------------
# Vision sampling (scene / face)
# -----------------------------

import cv2
import numpy as np

def sample_frames(local_video_path: str) -> List[Tuple[float, Any]]:
    """
    Returns list of (timestamp, frame-bgr) up to VISION_MAX_SAMPLES frames.
    Sample interval approx VISION_INTERVAL_SEC.
    """
    if not VISION_ENABLED:
        return []

    logger.info("[VISION] Sampling frames")
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = int(VISION_INTERVAL_SEC * fps)
    frames = []

    idx = 0
    while idx < total_frames and len(frames) < VISION_MAX_SAMPLES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        frames.append((t, frame))
        idx += interval

    cap.release()
    logger.info(f"[VISION] Sampled {len(frames)} frames")
    return frames


def frame_similarity(f1, f2) -> float:
    """
    Cosine similarity of downscaled grayscale histograms.
    Simple but effective.
    """
    f1g = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2g = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    f1h = cv2.calcHist([f1g], [0], None, [64], [0, 256])
    f2h = cv2.calcHist([f2g], [0], None, [64], [0, 256])

    f1h = cv2.normalize(f1h, None).flatten()
    f2h = cv2.normalize(f2h, None).flatten()

    dot = float(np.dot(f1h, f2h))
    denom = float(np.linalg.norm(f1h) * np.linalg.norm(f2h) + 1e-6)
    return dot / denom


def compute_visual_scores(segments: List[ClipSegment], frames: List[Tuple[float, Any]]):
    """
    For each segment, find nearest frame and compute visual quality metrics.
    """
    if not VISION_ENABLED or not frames:
        return

    ts_list = [t for (t, _) in frames]

    for seg in segments:
        mid = (seg.start + seg.end) / 2.0

        # nearest frame by timestamp
        diffs = [abs(t - mid) for t in ts_list]
        idx = int(np.argmin(diffs))
        frame = frames[idx][1]

        # Face detection or simple heuristics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 128.0

        seg.face_q = max(0.2, min(1.0, brightness))
        seg.scene_q = max(0.2, min(1.0, contrast))

        # store debug
        seg.visual_flags = {
            "brightness": brightness,
            "contrast": contrast
        }


# -----------------------------
# LLM scoring (OpenAI client)
# -----------------------------

def llm_score_segment(text: str) -> Dict[str, Any]:
    """
    Calls GPT to classify:
      - slot (HOOK / PROBLEM / FEATURE / PROOF / CTA)
      - semantic_score 0–1
      - explanation
      - blooper flag
    """
    if not client:
        return {
            "slot": "STORY",
            "semantic_score": 0.3,
            "reason": "LLM disabled",
            "blooper": False,
        }

    prompt = f"""
You evaluate 1 transcript line for an ad funnel.

TEXT: "{text}"

Return JSON keys only:
- slot: HOOK, PROBLEM, FEATURE, PROOF, CTA (pick best)
- semantic_score: 0 to 1
- reason: short explanation
- blooper: true/false   (true = retry, mistake, wrong wording, 'wait', 'start again', confusion)
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.2,
        )
        out = resp.choices[0].message.content
        data = json.loads(out)
        return data
    except Exception as e:
        logger.exception("[LLM] failed")
        return {
            "slot": "STORY",
            "semantic_score": 0.3,
            "reason": f"LLM error: {e}",
            "blooper": False,
        }


# -----------------------------
# Build ClipSegments from ASR segments
# -----------------------------

def build_clip_segments(raw_segments: List[Dict[str, Any]]) -> List[ClipSegment]:
    clips: List[ClipSegment] = []
    for i, s in enumerate(raw_segments):
        cid = f"ASR{i:04d}_c1"
        seg = ClipSegment(
            id=cid,
            start=float(s["start"]),
            end=float(s["end"]),
            text=s["text"],
            chain_ids=[cid]
        )
        clips.append(seg)
    return clips


# -----------------------------
# Apply semantic + visual + filler scoring
# -----------------------------

def enrich_segments(clips: List[ClipSegment]):
    """
    Adds slot, semantic_score, filler rate, total score, blooper flag.
    """
    for seg in clips:
        # LLM semantic classification
        info = llm_score_segment(seg.text)
        seg.slot = info.get("slot", "STORY")
        seg.semantic_score = float(info.get("semantic_score", 0.0))
        seg.llm_reason = info.get("reason", "")
        seg.llm_blooper = bool(info.get("blooper", False))

        # filler
        filler_rate = compute_filler_rate(seg.text)
        seg.semantic_score *= max(0.0, 1.0 - (filler_rate / SEM_FILLER_MAX_RATE))

        # combined score
        seg.score = (
            0.7 * seg.semantic_score +
            0.15 * seg.face_q +
            0.15 * seg.scene_q
        )


# ---- END OF PART 2/4 ----
# =========================================================
# PART 3 of 4 — SLOT BUILDER + COMPOSER + VIDEO RENDERING
# =========================================================

# -----------------------------
# Funnel slot grouping
# -----------------------------

def build_slots(clips: List[ClipSegment]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build slot groups:
       HOOK, PROBLEM, FEATURE, PROOF, CTA
    Each slot is a list of segment dicts (not objects).
    """
    slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}

    for c in clips:
        key = None
        if c.slot == "HOOK":
            key = "HOOK"
        elif c.slot == "PROBLEM":
            key = "PROBLEM"
        elif c.slot == "FEATURE":
            key = "FEATURE"
        elif c.slot == "PROOF":
            key = "PROOF"
        elif c.slot == "CTA":
            key = "CTA"

        if key:
            slots[key].append(c.to_dict())

    # sort each slot by score descending
    for k in slots:
        slots[k] = sorted(slots[k], key=lambda x: x["score"], reverse=True)

    return slots


# -----------------------------
# Composer: decide final funnel clips
# -----------------------------

def choose_best_hook(slots: Dict[str, List[Dict[str, Any]]], min_score=0.6):
    hooks = slots["HOOK"]
    if not hooks:
        return None
    best = hooks[0]
    return best if best["score"] >= min_score else None


def choose_features(slots: Dict[str, List[Dict[str, Any]]], min_score=0.6, max_features=8):
    feats = slots["FEATURE"]
    good = [f for f in feats if f["score"] >= min_score]
    return good[:max_features]


def choose_cta(slots: Dict[str, List[Dict[str, Any]]], min_score=0.6):
    ctas = slots["CTA"]
    if not ctas:
        return None
    best = ctas[0]
    return best if best["score"] >= min_score else None


def build_composer(slots: Dict[str, List[Dict[str, Any]]], min_score=0.6):
    hook = choose_best_hook(slots, min_score)
    features = choose_features(slots, min_score)
    cta = choose_cta(slots, min_score)

    used = []
    if hook:
        used.append(hook["id"])
    for f in features:
        used.append(f["id"])
    if cta:
        used.append(cta["id"])

    return {
        "hook_id": hook["id"] if hook else None,
        "feature_ids": [f["id"] for f in features],
        "cta_id": cta["id"] if cta else None,
        "used_clip_ids": used,
        "min_score": min_score,
    }


def human_readable_composer(slots, composer):
    """
    Pretty text block for debugging in returned JSON.
    """
    lines = ["===== EDITDNA FUNNEL COMPOSER =====\n"]

    def find_clip(cid):
        for group in slots.values():
            for c in group:
                if c["id"] == cid:
                    return c
        return None

    # HOOK
    if composer["hook_id"]:
        c = find_clip(composer["hook_id"])
        lines.append(f'HOOK ({composer["hook_id"]}, score={c["score"]:.2f}):')
        lines.append(f'  "{c["text"]}"\n')
    else:
        lines.append("HOOK: NONE FOUND\n")

    # FEATURES
    lines.append("FEATURES (kept):")
    for fid in composer["feature_ids"]:
        c = find_clip(fid)
        lines.append(f'  - [{fid}] score={c["score"]:.2f} → "{c["text"]}"')
    lines.append("")

    # CTA
    if composer["cta_id"]:
        c = find_clip(composer["cta_id"])
        lines.append(f'CTA ({composer["cta_id"]}, score={c["score"]:.2f}):')
        lines.append(f'  "{c["text"]}"\n')

    # timeline
    lines.append("FINAL ORDER TIMELINE:")
    idx = 1
    for cid in composer["used_clip_ids"]:
        c = find_clip(cid)
        lines.append(f"{idx}) {cid} → \"{c['text']}\"")
        idx += 1

    lines.append("\n=====================================\n")
    return "\n".join(lines)


# -----------------------------
# Render final mp4
# -----------------------------

def render_final_video(local_video_path: str, clips: List[ClipSegment], composer: Dict[str, Any], output_path: str):
    """
    Splice selected used_clip_ids in order into final.mp4.
    """
    used = composer["used_clip_ids"]
    if not used:
        raise ValueError("Composer returned no used_clip_ids — nothing to render.")

    logger.info(f"[RENDER] rendering {len(used)} segments")

    subclips = []
    for cid in used:
        seg = next((c for c in clips if c.id == cid), None)
        if not seg:
            continue
        subclips.append(
            VideoFileClip(local_video_path).subclip(seg.start, seg.end)
        )

    if not subclips:
        raise ValueError("No subclips found to render.")

    final = concatenate_videoclips(subclips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    logger.info(f"[RENDER] Saved: {output_path}")
    return output_path


# -----------------------------
# Clean temporary files (optional)
# -----------------------------

def safe_unlink(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---- END OF PART 3/4 ----
# =========================================================
# PART 4 of 4 — MAIN PIPELINE ENTRY + IMPLEMENTATION
# =========================================================

def upload_to_s3_if_configured(local_path: str, session_id: str) -> Optional[str]:
    """
    Upload final.mp4 to S3 ONLY if:
        EDITDNA_OUTPUT_BUCKET + EDITDNA_OUTPUT_PREFIX are set.

    If S3 is not configured, return None (worker still OK).
    """
    bucket = os.getenv("EDITDNA_OUTPUT_BUCKET", "").strip()
    prefix = os.getenv("EDITDNA_OUTPUT_PREFIX", "").strip()

    if not bucket or not prefix:
        logger.warning("[S3] Skipping upload (bucket/prefix not configured)")
        return None

    key = f"{prefix}/{session_id}/final.mp4"

    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key,
                       ExtraArgs={"ContentType": "video/mp4"})
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        logger.info(f"[S3] Uploaded → {url}")
        return url

    except Exception as e:
        logger.exception(f"[S3] Upload failed: {e}")
        return None


# ---------------------------------------------------------
# RUN PIPELINE INTERNAL IMPLEMENTATION
# ---------------------------------------------------------

def _run_pipeline_impl(job: Dict[str, Any]) -> Dict[str, Any]:

    # ------------------------------------------
    # validate
    # ------------------------------------------
    if "session_id" not in job:
        raise ValueError("job missing session_id")

    if not job.get("file_urls"):
        raise ValueError("job missing file_urls[]")

    session_id = job["session_id"]
    file_urls = job["file_urls"]

    # ------------------------------------------
    # 1. Download first video to /tmp/editdna/<session_id>/input.mp4
    # ------------------------------------------
    input_url = file_urls[0]
    base = f"/tmp/editdna/{session_id}"
    os.makedirs(base, exist_ok=True)
    local_path = os.path.join(base, "input.mp4")

    logger.info(f"[DL] downloading: {input_url}")
    with requests.get(input_url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # ------------------------------------------
    # 2. Transcribe with Whisper (GPU if available)
    # ------------------------------------------
    logger.info("[ASR] running whisper…")
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    asr = model.transcribe(local_path, fp16=torch.cuda.is_available())

    segments_raw = asr.get("segments", [])
    logger.info(f"[ASR] {len(segments_raw)} segments")

    # ------------------------------------------
    # 3. Convert whisper segments → ClipSegment objects & score
    # ------------------------------------------
    clips: List[ClipSegment] = []

    for idx, seg in enumerate(segments_raw):
        s = float(seg["start"])
        e = float(seg["end"])
        text = seg["text"].strip()
        cid = f"ASR{idx:04d}_c1"

        clip = ClipSegment(
            id=cid,
            start=s,
            end=e,
            text=text
        )

        # semantic scoring (LLM)
        try:
            sem = llm_score_segment(text)
            clip.semantic_score = float(sem["semantic_score"])
            clip.slot = sem["slot"]
            clip.llm_reason = sem["reason"]
        except Exception as e:
            logger.exception(f"[LLM] scoring failed: {e}")

        # visual scoring
        try:
            vs = score_clip_for_funnel(seg)
            clip.visual_score = float(vs["visual_score"])
        except Exception as e:
            logger.exception(f"[VIZ] scoring failed: {e}")

        # combine weighted
        clip.score = (
            SEMANTIC_WEIGHT * clip.semantic_score +
            VISUAL_WEIGHT * clip.visual_score
        )
        clip.score = max(0, min(1, clip.score))
        clip.vtx_sim = clip.score

        clips.append(clip)

    # ------------------------------------------
    # 4. Slots + Composer
    # ------------------------------------------
    slots = build_slots(clips)
    composer = build_composer(slots, min_score=0.7)
    composer_human = human_readable_composer(slots, composer)

    # ------------------------------------------
    # 5. Render final video
    # ------------------------------------------
    output_local = os.path.join(base, "final.mp4")

    try:
        render_final_video(local_path, clips, composer, output_local)
    except Exception as e:
        logger.exception("[RENDER] failed")
        raise

    # ------------------------------------------
    # 6. Upload (optional)
    # ------------------------------------------
    s3_url = upload_to_s3_if_configured(output_local, session_id)

    # ------------------------------------------
    # DONE
    # ------------------------------------------
    return {
        "session_id": session_id,
        "input_local": local_path,
        "duration_sec": float(asr.get("duration", 0)),
        "clips": [c.to_dict() for c in clips],
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_local,
        "output_video_url": s3_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }


# ---------------------------------------------------------
# PUBLIC ENTRYPOINT — used by tasks.job_render()
# ---------------------------------------------------------

def run_pipeline(*, session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    This is exactly what tasks.py calls:

       pipeline.run_pipeline(session_id=..., file_urls=[...])
    """
    job = {
        "session_id": session_id,
        "file_urls": file_urls
    }
    return _run_pipeline_impl(job)

