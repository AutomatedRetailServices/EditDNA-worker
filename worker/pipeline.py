import os
import re
import json
import uuid
import math
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

import boto3
from faster_whisper import WhisperModel
from openai import OpenAI

# -------------------------------------------------------------------
#  CONFIG & GLOBALS
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("editdna.pipeline")

# === ENV VARS ===
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

S3_BUCKET = os.environ.get("S3_BUCKET", "script2clipshop-video-automatedretailservices")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

TMP_ROOT = Path(os.environ.get("TMP_ROOT", "/tmp/TMP/editdna"))

# Composer config
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.80"))

# Slot limits:
# 0 = SIN LÍMITE (solo se filtra por calidad, no por cantidad)
SLOT_LIMITS: Dict[str, int] = {
    "HOOK": 1,        # 1 hook fuerte
    "STORY": 0,       # sin límite (no lo usamos mucho todavía)
    "PROBLEM": 1,     # 1 problema máximo
    "BENEFITS": 0,    # 0 = sin límite → no pierdes beneficios importantes
    "FEATURES": 0,    # 0 = sin límite → no pierdes descripciones ricas
    "PROOF": 1,       # 1 prueba/testimonio
    "CTA": 1,         # 1 CTA
}

# Veto layer (por ahora simple)
VETO_MIN_SCORE = float(os.environ.get("VETO_MIN_SCORE", "0.4"))

# -------------------------------------------------------------------
#  CLIENTS
# -------------------------------------------------------------------

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
s3_client = boto3.client("s3")

# Creamos el modelo de Whisper global para no cargarlo en cada job
logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})")
whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

# -------------------------------------------------------------------
#  HELPERS
# -------------------------------------------------------------------

def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _jaccard(a: str, b: str) -> float:
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_duplicate(text: str, used_norm_texts: List[str], threshold: float) -> bool:
    """Very simple near-duplicate check using Jaccard on tokens."""
    if not text:
        return False
    for prev in used_norm_texts:
        if _jaccard(text, prev) >= threshold:
            return True
    return False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _ensure_tmp_dir(session_id: str) -> Path:
    session_dir = TMP_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _upload_to_s3(local_path: Path, session_id: str) -> str:
    key = f"{S3_PREFIX.rstrip('/')}/{session_id}-final.mp4"
    logger.info(f"Uploading {local_path} to s3://{S3_BUCKET}/{key}")
    s3_client.upload_file(str(local_path), S3_BUCKET, key)

    url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=60 * 60 * 24 * 30,  # 30 días
    )
    return url


# -------------------------------------------------------------------
#  ASR (Whisper)
# -------------------------------------------------------------------

def run_asr(input_local: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Ejecuta Whisper y devuelve:
    - lista de segmentos [{start, end, text}, ...]
    - duración total del audio (segundos)
    """
    logger.info(f"Running Whisper ASR on {input_local}")
    segments_iter, info = whisper_model.transcribe(
        input_local,
        beam_size=5,
        word_timestamps=False,
    )

    segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments_iter):
        segments.append(
            {
                "index": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
            }
        )

    duration_sec = float(getattr(info, "duration", 0.0)) or (
        segments[-1]["end"] if segments else 0.0
    )

    logger.info(f"ASR produced {len(segments)} segments, duration ~{duration_sec:.2f}s")
    return segments, duration_sec


# -------------------------------------------------------------------
#  LLM CLASSIFIER (slot + keep + semantic_score)
# -------------------------------------------------------------------

CLASS_SYSTEM_PROMPT = """You are a performance ad editor assistant.

Task:
- You receive ONE spoken segment from a UGC-style ad.
- Classify it into ONE of these funnel slots:
  - HOOK
  - STORY
  - PROBLEM
  - BENEFITS
  - FEATURES
  - PROOF
  - CTA

- Decide:
  - semantic_score: number between 0.0 and 1.0 (how useful for the ad?)
  - keep: true/false
    - false for:
      - mistakes, "wait", "is that good?", "thanks", random laughter, meta comments about recording.
      - incomplete phrases that don't add value.
    - true for:
      - clear benefits, features, hook, CTA, testimonial, problem, story.

Return STRICTLY JSON like:
{
  "slot": "HOOK",
  "semantic_score": 0.92,
  "keep": true,
  "reason": "short explanation"
}"""

def classify_segment(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "slot": "STORY",
            "semantic_score": 0.0,
            "keep": False,
            "reason": "Empty text",
        }

    user_prompt = f"""Segment:
\"\"\"{text.strip()}\"\"\"

Classify this segment for a TikTok Shop performance ad.
Remember to ONLY output JSON as specified."""
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLASS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        slot = data.get("slot", "STORY").strip().upper()
        if slot not in ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]:
            slot = "STORY"
        sem = _safe_float(data.get("semantic_score", 0.0), 0.0)
        keep = bool(data.get("keep", False))
        reason = data.get("reason", "")

        return {
            "slot": slot,
            "semantic_score": sem,
            "keep": keep,
            "reason": reason,
        }
    except Exception as e:
        logger.warning(f"LLM classify failed, falling back: {e}")
        # fallback: treat as STORY but keep it with low score
        return {
            "slot": "STORY",
            "semantic_score": 0.3,
            "keep": True,
            "reason": "Fallback classification",
        }


# -------------------------------------------------------------------
#  BUILD CLIPS FROM ASR + LLM
# -------------------------------------------------------------------

def build_clips_from_asr(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    for seg in asr_segments:
        idx = seg["index"]
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]

        cls = classify_segment(text)
        slot = cls["slot"]
        semantic_score = float(cls["semantic_score"])
        keep = bool(cls["keep"])
        llm_reason = cls["reason"]

        # basic score = semantic_score (puedes mezclar visual luego)
        score = semantic_score

        clip_id = f"ASR{idx:04d}_c1"

        clip = {
            "id": clip_id,
            "slot": slot,
            "start": float(start),
            "end": float(end),
            "score": score,
            "semantic_score": semantic_score,
            # Visual metrics "fake" 1.0 por ahora, para no romper nada:
            "visual_score": 1.0,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": semantic_score,
            "chain_ids": [clip_id],
            "text": text,
            "llm_reason": llm_reason,
            "visual_flags": {
                "scene_jump": False,
                "motion_jump": False,
            },
            "meta": {
                "slot": slot,
                "semantic_score": semantic_score,
                "visual_score": 1.0,
                "score": score,
                "chain_ids": [],
                "keep": keep,
            },
        }
        clips.append(clip)

    return clips


# -------------------------------------------------------------------
#  GROUP CLIPS BY SLOT
# -------------------------------------------------------------------

def group_by_slot(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURES": [],
        "PROOF": [],
        "CTA": [],
    }
    for c in clips:
        slot = c.get("slot", "STORY").upper()
        if slot not in slots:
            slot = "STORY"
        slots[slot].append(c)
    return slots


# -------------------------------------------------------------------
#  FUNNEL COMPOSER (CON NUEVOS LÍMITES)
# -------------------------------------------------------------------

def compose_funnel(slots: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Devuelve:
    - composer (IDs por slot)
    - used_clip_ids (en orden de timeline final)
    """

    used_ids: List[str] = []
    used_norm_texts: List[str] = []

    def pick_from_slot(slot_name: str) -> List[str]:
        clips = slots.get(slot_name, []) or []
        limit = SLOT_LIMITS.get(slot_name, 0)

        if not clips:
            return []

        chosen_ids: List[str] = []

        ordered = sorted(
            clips,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )

        for c in ordered:
            # si hay límite (>0) y ya llegamos, paramos
            if limit > 0 and len(chosen_ids) >= limit:
                break

            meta = c.get("meta", {}) or {}
            if not meta.get("keep", True):
                continue

            sem = c.get("semantic_score")
            if sem is None:
                sem = meta.get("semantic_score", 0.0)
            sem = float(sem or 0.0)

            # filtro por semantic_score mínimo
            if sem < COMPOSER_MIN_SEMANTIC:
                continue

            text = c.get("text", "") or ""
            if _is_duplicate(text, used_norm_texts, COMPOSER_DUP_SIM_THRESHOLD):
                continue

            used_norm_texts.append(_normalize_text(text))
            cid = c["id"]
            chosen_ids.append(cid)
            used_ids.append(cid)

        return chosen_ids

    # HOOK (1)
    hook_ids = pick_from_slot("HOOK")
    hook_id = hook_ids[0] if hook_ids else None

    # STORY (puede ser 0 = sin límite, pero por ahora no usamos mucho)
    story_ids = pick_from_slot("STORY")

    # PROBLEM (1 máx)
    problem_ids = pick_from_slot("PROBLEM")

    # BENEFITS (0 = sin límite)
    benefit_ids = pick_from_slot("BENEFITS")

    # FEATURES (0 = sin límite)
    feature_ids = pick_from_slot("FEATURES")

    # PROOF (1 máx)
    proof_ids = pick_from_slot("PROOF")

    # CTA (1)
    cta_ids = pick_from_slot("CTA")
    cta_id = cta_ids[0] if cta_ids else None

    # Timeline final (orden funnel)
    final_order: List[str] = []
    if hook_id:
        final_order.append(hook_id)
    final_order.extend(story_ids)
    final_order.extend(problem_ids)
    final_order.extend(benefit_ids)
    final_order.extend(feature_ids)
    final_order.extend(proof_ids)
    if cta_id:
        final_order.append(cta_id)

    composer: Dict[str, Any] = {
        "hook_id": hook_id,
        "story_ids": story_ids,
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": feature_ids,
        "proof_ids": proof_ids,
        "cta_id": cta_id,
        "used_clip_ids": final_order,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }
    return composer


def build_composer_human(composer: Dict[str, Any], clip_index: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====")

    def line_for_id(cid: str) -> str:
        c = clip_index.get(cid)
        if not c:
            return f"  [{cid}] (missing)"
        score = float(c.get("score", 0.0))
        text = c.get("text", "")
        return f"  [{cid}] score={score:.2f} → \"{text}\""

    # HOOK
    lines.append("HOOK:")
    if composer.get("hook_id"):
        lines.append(line_for_id(composer["hook_id"]))
    else:
        lines.append("  (none)")

    # STORY
    lines.append("STORY:")
    if composer.get("story_ids"):
        for cid in composer["story_ids"]:
            lines.append(line_for_id(cid))
    else:
        lines.append("  (none)")

    # PROBLEM
    lines.append("PROBLEM:")
    if composer.get("problem_ids"):
        for cid in composer["problem_ids"]:
            lines.append(line_for_id(cid))
    else:
        lines.append("  (none)")

    # BENEFITS
    lines.append("BENEFITS:")
    if composer.get("benefit_ids"):
        for cid in composer["benefit_ids"]:
            lines.append(line_for_id(cid))
    else:
        lines.append("  (none)")

    # FEATURES
    lines.append("FEATURES:")
    if composer.get("feature_ids"):
        for cid in composer["feature_ids"]:
            lines.append(line_for_id(cid))
    else:
        lines.append("  (none)")

    # PROOF
    lines.append("PROOF:")
    if composer.get("proof_ids"):
        for cid in composer["proof_ids"]:
            lines.append(line_for_id(cid))
    else:
        lines.append("  (none)")

    # CTA
    lines.append("CTA:")
    if composer.get("cta_id"):
        lines.append(line_for_id(composer["cta_id"]))
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("FINAL ORDER TIMELINE:")
    for i, cid in enumerate(composer.get("used_clip_ids", []), start=1):
        c = clip_index.get(cid)
        if not c:
            lines.append(f"{i}) {cid}")
        else:
            text = c.get("text", "")
            lines.append(f"{i}) {cid} → \"{text}\"")

    lines.append("")
    lines.append("=====================================")
    return "\n".join(lines)


# -------------------------------------------------------------------
#  RENDER (FFMPEG) — CON FIX DE AUDIO/VIDEO
# -------------------------------------------------------------------

def render_funnel_video(
    input_local: str,
    session_dir: Path,
    clip_index: Dict[str, Dict[str, Any]],
    used_clip_ids: List[str],
) -> Path:
    """
    Corta del video original los segmentos usados y los concatena,
    con filtros para mantener audio y video sincronizados.
    """
    if not used_clip_ids:
        raise ValueError("No clips selected for final video")

    input_path = Path(input_local)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_local}")

    output_path = session_dir / "final.mp4"

    # Construimos filter_complex
    filter_parts: List[str] = []
    concat_video_inputs: List[str] = []
    concat_audio_inputs: List[str] = []

    for i, cid in enumerate(used_clip_ids):
        c = clip_index[cid]
        start = float(c["start"])
        end = float(c["end"])

        vname = f"v{i}"
        aname = f"a{i}"

        # trim de video y audio, reseteando PTS
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{vname}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{aname}]"
        )

        concat_video_inputs.append(f"[{vname}]")
        concat_audio_inputs.append(f"[{aname}]")

    n = len(used_clip_ids)
    # concat y luego resample de audio para sync
    filter_parts.append(
        f"{''.join(concat_video_inputs)}{''.join(concat_audio_inputs)}"
        f"concat=n={n}:v=1:a=1[vf][af]"
    )
    # audio sync fix
    filter_parts.append("[af]aresample=async=1:first_pts=0[af_sync]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vf]",
        "-map",
        "[af_sync]",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    logger.info("Running ffmpeg to render funnel video")
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    logger.info(f"Final video rendered at {output_path}")
    return output_path


# -------------------------------------------------------------------
#  VETO LAYER (simple)
# -------------------------------------------------------------------

def apply_veto(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Por ahora, solo filtra clips con score muy bajo."""
    result = []
    for c in clips:
        score = _safe_float(c.get("score", 0.0), 0.0)
        if score < VETO_MIN_SCORE:
            # podríamos marcar meta.keep=False, pero el composer ya usa eso
            meta = c.get("meta", {}) or {}
            meta["keep"] = False
            c["meta"] = meta
        result.append(c)
    return result


# -------------------------------------------------------------------
#  MAIN PIPELINE
# -------------------------------------------------------------------

def run_pipeline(session_id: str, input_local: str) -> Dict[str, Any]:
    """
    Main entry point desde tasks.py

    Args:
        session_id: ID de sesión (ej: "funnel-test-1")
        input_local: ruta local al video de entrada

    Returns:
        dict con:
          - ok
          - session_id
          - input_local
          - duration_sec
          - clips
          - slots
          - composer
          - composer_human
          - output_video_local
          - output_video_url
          - asr / semantic / vision flags
    """
    session_dir = _ensure_tmp_dir(session_id)

    # 1) ASR
    asr_segments, duration_sec = run_asr(input_local)

    # 2) LLM classification → clips
    clips = build_clips_from_asr(asr_segments)

    # 3) Veto/cleanup
    clips = apply_veto(clips)

    # 4) Slots
    slots = group_by_slot(clips)

    # 5) Composer
    composer = compose_funnel(slots)

    # index por ID
    clip_index = {c["id"]: c for c in clips}

    composer_human = build_composer_human(composer, clip_index)

    used_clip_ids = composer.get("used_clip_ids", [])
    if not used_clip_ids:
        # si no hay nada seleccionado, devolvemos solo JSON sin video final
        return {
            "ok": False,
            "session_id": session_id,
            "input_local": input_local,
            "duration_sec": duration_sec,
            "clips": clips,
            "slots": slots,
            "composer": composer,
            "composer_human": composer_human,
            "output_video_local": None,
            "output_video_url": None,
            "asr": True,
            "semantic": True,
            "vision": True,
        }

    # 6) Render video final (con fix de audio/video)
    final_path = render_funnel_video(input_local, session_dir, clip_index, used_clip_ids)

    # 7) Upload to S3
    output_video_url = _upload_to_s3(final_path, session_id)

    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": str(final_path),
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,  # por ahora lo dejamos en True para no romper UI
    }
    return result
