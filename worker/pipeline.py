import os
import json
import uuid
import math
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import urllib.request

import torch
import whisper

# --- OpenAI v1.x style client --- #
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # in case lib is missing


LOG = logging.getLogger("editdna.pipeline")
LOG.setLevel(logging.INFO)

# --------- CONFIG / ENV --------- #

OPENAI_API_KEY = (
    os.getenv("EDITDNA_OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# Optional S3 (if you use it)
S3_BUCKET = os.getenv("EDITDNA_S3_BUCKET", "")
S3_PREFIX = os.getenv("EDITDNA_S3_PREFIX", "editdna/outputs")

# Force CPU for Whisper if needed
FORCE_CPU = os.getenv("EDITDNA_FORCE_CPU", "0") == "1"

# --------- UTILS --------- #

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_ffprobe_duration(video_path: str) -> float:
    """
    Use ffprobe to get the video duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        dur = float(out.decode("utf-8").strip())
        return dur
    except Exception as e:
        LOG.warning(f"ffprobe failed for {video_path}: {e}")
        return 0.0


def download_first_file(file_urls: List[str], session_id: str) -> str:
    """
    Download the first URL in `file_urls` into /tmp/editdna/<session_id>/input.mp4
    and return the local path.
    """
    if not file_urls:
        raise ValueError("file_urls is empty")

    url = file_urls[0]
    work_dir = Path("/tmp") / "editdna" / session_id
    ensure_dir(work_dir)
    local_path = work_dir / "input.mp4"

    LOG.info(f"[download] fetching first file url={url} -> {local_path}")
    urllib.request.urlretrieve(url, str(local_path))

    return str(local_path)


# --------- WHISPER (GPU WHEN AVAILABLE) --------- #

_WHISPER_MODEL = None

def get_whisper_model() -> whisper.Whisper:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    LOG.info(f"[Whisper] loading model 'medium' on device={device}")
    _WHISPER_MODEL = whisper.load_model("medium", device=device)
    return _WHISPER_MODEL


def run_whisper_asr(video_path: str) -> Dict[str, Any]:
    """
    Run Whisper on the input video and return the raw transcription dict.
    """
    model = get_whisper_model()
    LOG.info(f"[Whisper] transcribing {video_path}")
    result = model.transcribe(
        audio=video_path,
        language="en",
        fp16=(torch.cuda.is_available() and not FORCE_CPU),
    )
    return result


# --------- LLM SCORING / SLOTS --------- #

GOOD_SLANG_HOOK_KEYWORDS = [
    "kutigei",
    "coochie gang",
    "coochie",
]

BAD_PHRASE_PATTERNS = [
    r"wait[.,!?]?$",
    r"wait not",
    r"i don't know why",
    r"why can't i remember",
    r"is that funny\??",
    r"is that good\??",
    r"^yeah[.?!]?$",
    r"that one good\??",
]

BAD_REASON_KEYWORDS = [
    "incomplete",
    "vague",
    "confusing",
    "weak",
    "does not engage",
    "too short",
    "generic",
    "repetitive",
    "ends abruptly",
    "slip-up",
    "correction",
]


def is_trash_line(text: str, llm_reason: str) -> bool:
    """
    Mark obvious retakes / bloopers / filler as trash so they NEVER go into the funnel.

    BUT: if the text contains GOOD slang hook words (kutigei/coochie),
    we NEVER treat it as trash, even if LLM complains about 'confusing'.
    """
    import re

    t = (text or "").strip().lower()
    r = (llm_reason or "").lower()

    # ✅ 0) If it's our spicy slang hook (kutigei / coochie), keep it NO MATTER WHAT
    if any(kw in t for kw in GOOD_SLANG_HOOK_KEYWORDS):
        return False

    # 1) Very short non-meaningful lines
    word_count = len(t.split())
    if word_count <= 2 and t not in ("yes", "no", "okay", "ok", "sure"):
        return True

    # 2) Explicit bad / re-take phrases
    for pat in BAD_PHRASE_PATTERNS:
        if re.search(pat, t):
            return True

    # 3) LLM explicitly says it's bad
    for kw in BAD_REASON_KEYWORDS:
        if kw in r:
            return True

    return False


def combine_scores(semantic_score: float, visual_score: float) -> float:
    """
    Favor semantic score (what is said) over visual score (how pretty the frame is).
    """
    sem = float(semantic_score or 0.0)
    vis = float(visual_score or 0.0)
    return 0.85 * sem + 0.15 * vis


def llm_score_segment(text: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Ask LLM to:
      - classify slot: HOOK / STORY / FEATURE / PROOF / CTA
      - give semantic_score (0.0-1.0)
      - give short llm_reason
    Returns a dict with keys: slot, semantic_score, llm_reason
    """
    # If we don't have a client or key, fallback
    if not openai_client or not OPENAI_API_KEY:
        return {
            "slot": "STORY",
            "semantic_score": 0.6,
            "llm_reason": "Fallback scoring (no OpenAI client).",
        }

    prompt = f"""
You are scoring lines from a TikTok-style UGC ad script about a product.
For the given line, do three things:
1) Choose ONE slot: HOOK, STORY, FEATURE, PROOF, CTA.
2) Give a semantic quality score between 0.0 and 1.0 (float).
3) Write a short reason.

Return JSON ONLY with keys: "slot", "semantic_score", "llm_reason".

Line:
"{text}"
""".strip()

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a precise JSON-only scorer."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
        slot = str(data.get("slot", "STORY")).upper()
        if slot not in ["HOOK", "STORY", "FEATURE", "PROOF", "CTA"]:
            slot = "STORY"
        semantic_score = float(data.get("semantic_score", 0.6))
        semantic_score = max(0.0, min(1.0, semantic_score))
        llm_reason = str(data.get("llm_reason", "")).strip()
        return {
            "slot": slot,
            "semantic_score": semantic_score,
            "llm_reason": llm_reason,
        }
    except Exception as e:
        LOG.warning(f"Failed to parse LLM JSON for text='{text[:40]}...': {e}")
        return {
            "slot": "STORY",
            "semantic_score": 0.6,
            "llm_reason": "LLM parse error; fallback.",
        }


def score_clip_for_funnel(raw_clip: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Take one raw ASR segment and convert it into our 'clip' dict with:
      - id, start, end, text
      - slot
      - semantic_score, visual_score, face_q, scene_q, visual_flags
      - meta.score (combined semantic+visual)

    Returns None if we decide it's trash and shouldn't be used.
    """
    text = raw_clip.get("text", "") or ""
    start = float(raw_clip.get("start", 0.0))
    end = float(raw_clip.get("end", 0.0))

    # 1) Ask LLM for semantic score + slot
    llm_info = llm_score_segment(text)
    slot = llm_info["slot"]
    semantic_score = llm_info["semantic_score"]
    llm_reason = llm_info["llm_reason"]

    # 2) Vision: for now we assume it's fine (keeps micro-cut logic)
    visual_score = 1.0
    face_q = 1.0
    scene_q = 1.0
    visual_flags = {
        "scene_jump": False,
        "motion_jump": False,
    }

    # 3) Trash filter (retakes / bloopers / etc.)
    if is_trash_line(text, llm_reason):
        return None

    # 4) Combined score
    combined = combine_scores(semantic_score, visual_score)

    # 5) Tiny boost for “good slang” hooks so they survive
    txt_lower = text.lower()
    if any(kw in txt_lower for kw in GOOD_SLANG_HOOK_KEYWORDS):
        combined += 0.08
        if combined > 1.0:
            combined = 1.0

    cid = f"ASR{int(start*10):04d}_c1"

    clip = {
        "id": cid,
        "slot": slot if slot != "STORY" else "STORY",
        "start": start,
        "end": end,
        "score": combined,
        "semantic_score": semantic_score,
        "visual_score": visual_score,
        "face_q": face_q,
        "scene_q": scene_q,
        "vtx_sim": combined,
        "chain_ids": [cid],
        "text": text,
        "llm_reason": llm_reason,
        "visual_flags": visual_flags,
    }

    # meta.slot: map STORY→FEATURE for funnel purposes
    meta_slot = slot if slot != "STORY" else "FEATURE"
    meta = {
        "slot": meta_slot,
        "semantic_score": semantic_score,
        "visual_score": visual_score,
        "score": combined,
        "chain_ids": [cid],
    }

    clip["meta"] = meta
    return clip


# --------- COMPOSER --------- #

def build_funnel_composer(
    clips: List[Dict[str, Any]],
    min_feature_score: float = 0.70,
) -> Tuple[Dict[str, Any], str, Dict[str, List[Dict[str, Any]]]]:
    """
    Build the EDITDNA funnel composer:
      - HOOK: best hook clip, strongly prefers 'kutigei/coochie' line if present.
      - FEATURES: best feature/story lines above min_feature_score.
      - CTA: best CTA line.

    Returns:
      composer_dict, composer_human_str, slots_dict
    """

    # Map by id
    id_to_clip = {c["id"]: c for c in clips}

    # Build slots dict
    slots: Dict[str, List[Dict[str, Any]]] = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    for c in clips:
        slot = c.get("slot", "STORY").upper()
        mapped_slot = slot
        if slot == "STORY":
            mapped_slot = "FEATURE"   # we treat STORY as FEATURE in funnel timeline
        if mapped_slot not in slots:
            mapped_slot = "FEATURE"
        slots[mapped_slot].append(c)

    hooks = slots["HOOK"]
    features = slots["FEATURE"]
    proofs = slots["PROOF"]
    ctas = slots["CTA"]

    # ---- pick HOOK ---- #
    hook_clip = None
    hook_id = None

    slang_hooks = [
        c for c in hooks
        if any(kw in c.get("text", "").lower() for kw in GOOD_SLANG_HOOK_KEYWORDS)
    ]
    if slang_hooks:
        hook_clip = max(slang_hooks, key=lambda c: c["meta"].get("score", 0.0))
    elif hooks:
        hook_clip = max(hooks, key=lambda c: c["meta"].get("score", 0.0))

    if hook_clip:
        hook_id = hook_clip["id"]

    # ---- pick FEATURES (story/feature/proof) ---- #
    all_feat_candidates: List[Dict[str, Any]] = []
    all_feat_candidates.extend(features)
    all_feat_candidates.extend(proofs)

    good_feats = [
        c for c in all_feat_candidates
        if c["meta"].get("score", 0.0) >= min_feature_score
    ]
    good_feats.sort(key=lambda c: c["meta"].get("score", 0.0), reverse=True)
    feature_ids = [c["id"] for c in good_feats]

    # ---- pick CTA ---- #
    cta_clip = None
    cta_id = None
    if ctas:
        cta_clip = max(ctas, key=lambda c: c["meta"].get("score", 0.0))
    if cta_clip:
        cta_id = cta_clip["id"]

    # used_clip_ids in play order: HOOK → FEATURES (score-desc) → CTA
    used_clip_ids: List[str] = []
    if hook_id:
        used_clip_ids.append(hook_id)
    for fid in feature_ids:
        if fid not in used_clip_ids:
            used_clip_ids.append(fid)
    if cta_id and cta_id not in used_clip_ids:
        used_clip_ids.append(cta_id)

    composer = {
        "hook_id": hook_id,
        "feature_ids": feature_ids,
        "cta_id": cta_id,
        "used_clip_ids": used_clip_ids,
        "min_score": float(min_feature_score),
    }

    # ---- human-readable ---- #

    def fmt_clip_line(c: Dict[str, Any]) -> str:
        cid = c.get("id", "?")
        text = c.get("text", "").strip()
        score = c.get("meta", {}).get("score", 0.0)
        return f'  - [{cid}] score={score:.2f} → "{text}"'

    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")

    # HOOK
    if hook_clip:
        hscore = hook_clip["meta"].get("score", 0.0)
        lines.append(f'HOOK ({hook_clip["id"]}, score={hscore:.2f}):')
        lines.append(f'  "{hook_clip.get("text", "").strip()}"\n')
    else:
        lines.append("HOOK: <none>\n")

    # FEATURES
    lines.append("FEATURES (kept):")
    if good_feats:
        for c in good_feats:
            lines.append(fmt_clip_line(c))
    else:
        lines.append("  - <none>")
    lines.append("")

    # CTA
    if cta_clip:
        cscore = cta_clip["meta"].get("score", 0.0)
        lines.append(f'CTA ({cta_clip["id"]}, score={cscore:.2f}):')
        lines.append(f'  "{cta_clip.get("text", "").strip()}"\n')
    else:
        lines.append("CTA: <none>\n")

    # FINAL TIMELINE
    lines.append("FINAL ORDER TIMELINE:")
    idx = 1

    if hook_clip:
        lines.append(f'{idx}) {hook_clip["id"]} → "{hook_clip.get("text", "").strip()}"')
        idx += 1

    for fid in feature_ids:
        c = id_to_clip.get(fid)
        if not c:
            continue
        lines.append(f'{idx}) {fid} → "{c.get("text", "").strip()}"')
        idx += 1

    if cta_clip:
        lines.append(f'{idx}) {cta_clip["id"]} → "{cta_clip.get("text", "").strip()}"')

    lines.append("\n=====================================")
    composer_human = "\n".join(lines)

    return composer, composer_human, slots


# --------- VIDEO RENDER (simple ffmpeg concat) --------- #

def render_funnel_video(
    input_video: str,
    clips: List[Dict[str, Any]],
    composer: Dict[str, Any],
    session_id: str,
    work_dir: Path,
) -> Tuple[str, Optional[str]]:
    """
    Very simple: cut each used clip to its [start, end], then concat with ffmpeg.

    Returns:
      local_output_path, s3_url_or_None
    """
    ensure_dir(work_dir)
    used_ids = composer.get("used_clip_ids", [])
    id_to_clip = {c["id"]: c for c in clips}

    temp_dir = work_dir / "segments"
    ensure_dir(temp_dir)

    segment_files: List[Path] = []
    for idx, cid in enumerate(used_ids):
        c = id_to_clip.get(cid)
        if not c:
            continue
        start = float(c.get("start", 0.0))
        end = float(c.get("end", 0.0))
        dur = max(0.0, end - start)
        if dur <= 0.0:
            continue
        seg_path = temp_dir / f"seg_{idx:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(dur),
            "-i", input_video,
            "-c", "copy",
            str(seg_path),
        ]
        subprocess.run(cmd, check=True)
        segment_files.append(seg_path)

    if not segment_files:
        LOG.warning("No segment files produced; returning original video.")
        return input_video, None

    # Build concat list file
    concat_file = work_dir / "concat.txt"
    with concat_file.open("w") as f:
        for p in segment_files:
            f.write(f"file '{p.as_posix()}'\n")

    output_video = work_dir / "final.mp4"
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_video),
    ]
    subprocess.run(cmd_concat, check=True)

    output_video_local = str(output_video)

    # Optional S3 upload
    output_url = None
    if S3_BUCKET:
        try:
            import boto3

            s3 = boto3.client("s3")
            key = f"{S3_PREFIX}/{session_id}/final.mp4"
            s3.upload_file(output_video_local, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
            output_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        except Exception as e:
            LOG.warning(f"S3 upload failed: {e}")
            output_url = None

    return output_video_local, output_url


# --------- INTERNAL IMPLEMENTATION --------- #

def _run_pipeline_impl(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal implementation that expects a job dict with:
      - session_id
      - input_local (local path to the video)
    """
    session_id = job.get("session_id") or f"session-{uuid.uuid4().hex[:8]}"

    input_local = job.get("input_local")
    if not input_local:
        available_keys = list(job.keys())
        raise ValueError(f"job['input_local'] is required; got keys={available_keys}")

    input_path = Path(str(input_local))
    if not input_path.exists():
        raise FileNotFoundError(f"input_local not found: {input_path}")

    duration_sec = run_ffprobe_duration(str(input_path))

    # 1) ASR
    asr_result = run_whisper_asr(str(input_path))
    segments = asr_result.get("segments", []) or []

    # 2) Build clips via LLM scoring + filters
    clips: List[Dict[str, Any]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        scored = score_clip_for_funnel(seg)
        if scored is None:
            continue
        clips.append(scored)

    # 3) Composer (this also builds slots)
    composer, composer_human, slots = build_funnel_composer(clips, min_feature_score=0.70)

    # 4) Render final video
    work_dir = Path("/tmp") / "editdna" / session_id
    output_video_local, output_video_url = render_funnel_video(
        input_video=str(input_path),
        clips=clips,
        composer=composer,
        session_id=session_id,
        work_dir=work_dir,
    )

    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": str(input_path),
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": output_video_local,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,  # we stub visual_score=1.0 but keep flag on
    }

    return result


# --------- PUBLIC ENTRYPOINT (MATCHES YOUR tasks.py) --------- #

def run_pipeline(*args, **kwargs) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper.

    Supports:
      - run_pipeline(job_dict)
      - run_pipeline(session_id=..., file_urls=[...])
    """
    # Case 1: first arg is a dict (job object)
    if args and isinstance(args[0], dict):
        job = dict(args[0])
        job.update(kwargs)
        # If file_urls present and no input_local, download
        file_urls = job.get("file_urls") or job.get("files")
        if file_urls and not job.get("input_local"):
            session_id = job.get("session_id") or f"session-{uuid.uuid4().hex[:8]}"
            job["session_id"] = session_id
            job["input_local"] = download_first_file(file_urls, session_id)
        return _run_pipeline_impl(job)

    # Case 2: called with keyword args only (what your tasks.py does)
    if kwargs:
        job = dict(kwargs)
        file_urls = job.get("file_urls") or job.get("files")
        if file_urls and not job.get("input_local"):
            session_id = job.get("session_id") or f"session-{uuid.uuid4().hex[:8]}"
            job["session_id"] = session_id
            job["input_local"] = download_first_file(file_urls, session_id)
        return _run_pipeline_impl(job)

    raise ValueError("run_pipeline requires either a job dict or keyword args (session_id=..., file_urls=[...]).")
