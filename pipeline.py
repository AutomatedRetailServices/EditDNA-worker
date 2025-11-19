import os
import re
import json
import math
import uuid
import logging
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

import requests
from faster_whisper import WhisperModel
from openai import OpenAI

try:
    import boto3
except ImportError:
    boto3 = None

logger = logging.getLogger("editdna.pipeline")

# ==========
# CONFIG
# ==========

TMP_ROOT = os.environ.get("TMP_ROOT", "/tmp/TMP/editdna")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "medium")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

# Composer tuning – SUAVE para no matar frases buenas
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.30"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.98"))

SLOT_LIMITS: Dict[str, int] = {
    "HOOK": 7,
    "STORY": 7,
    "PROBLEM": 7,
    "BENEFITS": 7,
    "FEATURES": 7,
    "PROOF": 7,
    "CTA": 3,  # pocas CTAs máximo
}

# ==========
# GLOBAL CLIENTS
# ==========

_openai_client: Optional[OpenAI] = None
_whisper_model: Optional[WhisperModel] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_whisper_model() -> WhisperModel:
    """
    Intenta cargar Whisper en GPU (cuda, float16).
    Si falla, cae a CPU (int8).
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        logger.info("Loading Whisper model on CUDA (float16)")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cuda",
            compute_type="float16",
        )
    except Exception as e:
        logger.warning(f"Falling back to CPU for Whisper: {e}")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


# ==========
# HELPERS
# ==========

def ensure_session_dir(session_id: str) -> str:
    session_dir = os.path.join(TMP_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def download_to_local(url: str, session_dir: str) -> str:
    """
    Descarga un solo video a input.mp4 dentro del session_dir.
    """
    local_path = os.path.join(session_dir, "input.mp4")
    logger.info(f"Downloading input video from {url} -> {local_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_path


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_similarity(a: str, b: str) -> float:
    a_n = norm_text(a)
    b_n = norm_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


# ==========
# ASR (WHISPER)
# ==========

def run_asr(input_local: str) -> List[Dict[str, Any]]:
    """
    Ejecuta faster-whisper con word timestamps.
    Devuelve lista de segments:
    [
      {
        "id": "SEG0000",
        "start": 0.0,
        "end": 3.2,
        "text": "...",
        "words": [{"start":..., "end":..., "word": "Hello"}, ...]
      },
      ...
    ]
    """
    model = get_whisper_model()
    logger.info("Running Whisper ASR with word timestamps")
    segments_gen, info = model.transcribe(
        input_local,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
    )

    segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments_gen):
        words = []
        if getattr(seg, "words", None):
            for w in seg.words:
                if w is None:
                    continue
                words.append({
                    "start": safe_float(getattr(w, "start", None)),
                    "end": safe_float(getattr(w, "end", None)),
                    "word": str(getattr(w, "word", "")).strip(),
                })
        segments.append(
            {
                "id": f"SEG{i:04d}",
                "start": safe_float(getattr(seg, "start", None)),
                "end": safe_float(getattr(seg, "end", None)),
                "text": str(getattr(seg, "text", "")).strip(),
                "words": words,
            }
        )

    if segments:
        logger.info(
            f"ASR produced {len(segments)} segments, duration ~{segments[-1]['end']:.2f}s"
        )
    else:
        logger.warning("ASR produced 0 segments")

    return segments


# ==========
# MICRO-CUTS (SENTENCE BOUNDARY)
# ==========

def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Toma segments con words y los parte en oraciones más cortas:
    - Se corta por puntuación (. ! ?)
    - También por silencios > ~0.6s
    - También cuando una oración se alarga > 10s aprox.

    Devuelve lista de "clips" base (sin aún slot ni scores LLM).
    """
    clips: List[Dict[str, Any]] = []
    global_idx = 0

    PUNCT = {".", "!", "?"}
    MAX_SENT_DURATION = 10.0
    GAP_THRESHOLD = 0.6

    for seg in asr_segments:
        words = seg.get("words") or []
        if not words:
            # sin words: usar el segmento como uno solo
            text = seg.get("text", "").strip()
            if not text:
                continue
            clip_id = f"ASR{global_idx:04d}_c0"
            clip = {
                "id": clip_id,
                "slot": "UNLABELED",
                "start": safe_float(seg.get("start", 0.0)),
                "end": safe_float(seg.get("end", 0.0)),
                "score": 0.0,
                "semantic_score": 0.0,
                "visual_score": 1.0,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [clip_id],
                "text": text,
                "llm_reason": "",
                "visual_flags": {
                    "scene_jump": False,
                    "motion_jump": False,
                },
                "meta": {
                    "slot": "UNLABELED",
                    "semantic_score": 0.0,
                    "visual_score": 1.0,
                    "score": 0.0,
                    "chain_ids": [],
                    "keep": True,
                },
            }
            clips.append(clip)
            global_idx += 1
            continue

        buffer: List[Dict[str, Any]] = []
        for i, w in enumerate(words):
            w_text = str(w.get("word", "")).strip()
            w_start = safe_float(w.get("start", None))
            w_end = safe_float(w.get("end", None))

            if w_start is None or w_end is None:
                continue

            if buffer:
                last_end = safe_float(buffer[-1].get("end", 0.0))
                gap = max(0.0, w_start - last_end)
            else:
                gap = 0.0

            buffer.append({"start": w_start, "end": w_end, "word": w_text})

            # Condiciones para cortar:
            is_punct = len(w_text) > 0 and w_text[-1] in PUNCT
            duration = buffer[-1]["end"] - buffer[0]["start"]
            too_long = duration >= MAX_SENT_DURATION
            big_gap = gap >= GAP_THRESHOLD

            if is_punct or too_long or big_gap:
                # flush
                sent_start = buffer[0]["start"]
                sent_end = buffer[-1]["end"]
                sent_text = " ".join(b["word"] for b in buffer).strip()
                if sent_text:
                    clip_id = f"ASR{global_idx:04d}_c0"
                    clip = {
                        "id": clip_id,
                        "slot": "UNLABELED",
                        "start": sent_start,
                        "end": sent_end,
                        "score": 0.0,
                        "semantic_score": 0.0,
                        "visual_score": 1.0,
                        "face_q": 1.0,
                        "scene_q": 1.0,
                        "vtx_sim": 0.0,
                        "chain_ids": [clip_id],
                        "text": sent_text,
                        "llm_reason": "",
                        "visual_flags": {
                            "scene_jump": False,
                            "motion_jump": False,
                        },
                        "meta": {
                            "slot": "UNLABELED",
                            "semantic_score": 0.0,
                            "visual_score": 1.0,
                            "score": 0.0,
                            "chain_ids": [],
                            "keep": True,
                        },
                    }
                    clips.append(clip)
                    global_idx += 1
                buffer = []

        # flush final
        if buffer:
            sent_start = buffer[0]["start"]
            sent_end = buffer[-1]["end"]
            sent_text = " ".join(b["word"] for b in buffer).strip()
            if sent_text:
                clip_id = f"ASR{global_idx:04d}_c0"
                clip = {
                    "id": clip_id,
                    "slot": "UNLABELED",
                    "start": sent_start,
                    "end": sent_end,
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "visual_score": 1.0,
                    "face_q": 1.0,
                    "scene_q": 1.0,
                    "vtx_sim": 0.0,
                    "chain_ids": [clip_id],
                    "text": sent_text,
                    "llm_reason": "",
                    "visual_flags": {
                        "scene_jump": False,
                        "motion_jump": False,
                    },
                    "meta": {
                        "slot": "UNLABELED",
                        "semantic_score": 0.0,
                        "visual_score": 1.0,
                        "score": 0.0,
                        "chain_ids": [],
                        "keep": True,
                    },
                }
                clips.append(clip)
                global_idx += 1

    logger.info(f"sentence_boundary_micro_cuts produced {len(clips)} micro-clips")
    return clips


# ==========
# LLM CLASSIFIER
# ==========

CLASSIFIER_SYSTEM_PROMPT = """
You are an assistant that labels very short spoken sentences from UGC ads for TikTok, Reels, etc.

For each sentence you MUST return a JSON object with:
- slot: one of ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]
- keep: true or false
  - keep = false ONLY if the sentence is clearly:
    * a redo (e.g. "wait, let me start again", "no, that's not right"),
    * filler/meta (e.g. "is this good?", "can we cut this?", "I messed up"),
    * or non-ad trash (unrelated chatter).
- reason: short explanation in one sentence.
- semantic_score: float between 0 and 1 (0.0=useless, 1.0=amazing for the ad).
- score: float between 0 and 1 (normally same as semantic_score).

Return ONLY valid JSON. No extra text.
"""


def classify_clip_with_llm(clip: Dict[str, Any]) -> None:
    """
    Modifica el clip IN-PLACE:
    - slot
    - llm_reason
    - semantic_score
    - score
    - meta.slot
    - meta.semantic_score
    - meta.score
    - meta.keep
    """
    client = get_openai_client()
    text = clip.get("text", "").strip()

    if not text:
        clip["slot"] = "STORY"
        clip["semantic_score"] = 0.0
        clip["score"] = 0.0
        clip["llm_reason"] = "Empty text."
        clip["meta"]["slot"] = "STORY"
        clip["meta"]["semantic_score"] = 0.0
        clip["meta"]["score"] = 0.0
        clip["meta"]["keep"] = False
        return

    user_prompt = {
        "role": "user",
        "content": json.dumps(
            {
                "sentence": text,
            },
            ensure_ascii=False,
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                user_prompt,
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM classification failed, using fallback: {e}")
        data = {
            "slot": "BENEFITS",
            "keep": True,
            "reason": "Fallback: sentence kept by default as a benefit.",
            "semantic_score": 0.6,
            "score": 0.6,
        }

    slot = str(data.get("slot", "BENEFITS")).upper()
    if slot not in ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]:
        slot = "BENEFITS"

    keep = bool(data.get("keep", True))
    reason = str(data.get("reason", "")).strip()
    semantic_score = float(data.get("semantic_score", 0.6))
    score = float(data.get("score", semantic_score))

    clip["slot"] = slot
    clip["llm_reason"] = reason
    clip["semantic_score"] = semantic_score
    clip["score"] = score
    clip["meta"]["slot"] = slot
    clip["meta"]["semantic_score"] = semantic_score
    clip["meta"]["score"] = score
    clip["meta"]["keep"] = keep


# ==========
# COMPOSER V2 – FREE-FLOW + DEDUPE SUAVE + ORDEN
# ==========

def funnel_composer(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    - NO reorden loco por slots: respetamos orden cronológico.
    - Eliminamos:
        * clips con keep=False (clear redo/filler)
        * clips con semantic_score < COMPOSER_MIN_SEMANTIC
        * duplicados MUY parecidos (similarity >= COMPOSER_DUP_SIM_THRESHOLD)
        * exceso de clips por slot según SLOT_LIMITS
    - CTA(s) se mueven al final, nada más.
    - Marcamos en meta.keep = True/False pero NO borramos nada de `clips`.

    Devuelve:
    {
      "hook_id": ...,
      "story_ids": [...],
      "problem_ids": [...],
      "benefit_ids": [...],
      "feature_ids": [...],
      "proof_ids": [...],
      "cta_id": ...,
      "used_clip_ids": [...],
      "min_score": COMPOSER_MIN_SEMANTIC,
    }
    """
    # Trabajamos sobre una copia para lógica, pero actualizamos meta.keep en clips originales
    by_id = {c["id"]: c for c in clips}

    # Orden cronológico
    sorted_clips = sorted(clips, key=lambda c: safe_float(c.get("start", 0.0)))

    # 1) Filtro suave por keep + semantic_score
    filtered: List[Dict[str, Any]] = []
    for c in sorted_clips:
        keep = bool(c.get("meta", {}).get("keep", True))
        s_score = float(c.get("semantic_score", 0.0))
        if not keep:
            continue
        if s_score < COMPOSER_MIN_SEMANTIC:
            # se queda en clips pero se marca como no usado
            c["meta"]["keep"] = False
            continue
        filtered.append(c)

    # 2) DEDUPE SUAVE (texto muy parecido)
    deduped: List[Dict[str, Any]] = []
    chosen_norms: List[str] = []

    for c in filtered:
        t = c.get("text", "")
        norm = norm_text(t)
        if not norm:
            c["meta"]["keep"] = False
            continue

        is_dup = False
        best_sim = 0.0
        best_idx = -1

        for idx, existing_norm in enumerate(chosen_norms):
            sim = text_similarity(norm, existing_norm)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= COMPOSER_DUP_SIM_THRESHOLD:
            # Sólo reemplazar si este nuevo es claramente mejor (score mayor)
            existing_clip = deduped[best_idx]
            if float(c.get("score", 0.0)) > float(existing_clip.get("score", 0.0)):
                # Nuevo gana → viejo pasa a no-keep
                existing_clip["meta"]["keep"] = False
                c["meta"]["keep"] = True
                deduped[best_idx] = c
                chosen_norms[best_idx] = norm
            else:
                # El viejo se queda, este no
                c["meta"]["keep"] = False
            is_dup = True
        else:
            deduped.append(c)
            chosen_norms.append(norm)
            c["meta"]["keep"] = True

        if is_dup:
            continue

    # 3) Caps por slot (pero seguimos orden cronológico)
    slot_counts: Dict[str, int] = {}
    capped: List[Dict[str, Any]] = []
    for c in deduped:
        slot = c.get("slot", "STORY")
        limit = SLOT_LIMITS.get(slot, 999)
        current = slot_counts.get(slot, 0)
        if current >= limit:
            c["meta"]["keep"] = False
            continue
        slot_counts[slot] = current + 1
        c["meta"]["keep"] = True
        capped.append(c)

    # 4) CTA(s) al final
    non_cta = [c for c in capped if c.get("slot") != "CTA"]
    ctas = [c for c in capped if c.get("slot") == "CTA"]
    timeline = non_cta + ctas

    used_clip_ids = [c["id"] for c in timeline]

    # 5) Construir listas por slot para el JSON
    slots_map: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURES": [],
        "PROOF": [],
        "CTA": [],
    }
    for c in capped:
        s = c.get("slot", "STORY")
        if s in slots_map:
            slots_map[s].append(c)

    hook_id = slots_map["HOOK"][0]["id"] if slots_map["HOOK"] else None
    story_ids = [c["id"] for c in slots_map["STORY"]]
    problem_ids = [c["id"] for c in slots_map["PROBLEM"]]
    benefit_ids = [c["id"] for c in slots_map["BENEFITS"]]
    feature_ids = [c["id"] for c in slots_map["FEATURES"]]
    proof_ids = [c["id"] for c in slots_map["PROOF"]]
    cta_id = slots_map["CTA"][-1]["id"] if slots_map["CTA"] else None

    # 6) composer_human bonito
    def _fmt_clip(c: Dict[str, Any]) -> str:
        return f"[{c['id']}] score={c.get('score', 0.0):.2f} → \"{c.get('text', '').strip()}\""

    lines = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====")
    lines.append("HOOK:")
    if slots_map["HOOK"]:
        for c in slots_map["HOOK"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("STORY:")
    if slots_map["STORY"]:
        for c in slots_map["STORY"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("PROBLEM:")
    if slots_map["PROBLEM"]:
        for c in slots_map["PROBLEM"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("BENEFITS:")
    if slots_map["BENEFITS"]:
        for c in slots_map["BENEFITS"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("FEATURES:")
    if slots_map["FEATURES"]:
        for c in slots_map["FEATURES"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("PROOF:")
    if slots_map["PROOF"]:
        for c in slots_map["PROOF"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("CTA:")
    if slots_map["CTA"]:
        for c in slots_map["CTA"]:
            lines.append(f"  {_fmt_clip(c)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("FINAL ORDER TIMELINE:")
    for idx, cid in enumerate(used_clip_ids, start=1):
        c = by_id[cid]
        lines.append(f"{idx}) {cid} → \"{c.get('text', '').strip()}\"")

    lines.append("")
    lines.append("=====================================")
    composer_human = "\n".join(lines)

    composer = {
        "hook_id": hook_id,
        "story_ids": story_ids,
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": feature_ids,
        "proof_ids": proof_ids,
        "cta_id": cta_id,
        "used_clip_ids": used_clip_ids,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }

    return composer, slots_map, composer_human


# ==========
# RENDER – FFMPEG A/V TRIM + CONCAT
# ==========

def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    Corta [0:v] y [0:a] en paralelo usando trim/atrim,
    resetea PTS con setpts/asetpts,
    concatena con concat=n=N:v=1:a=1[vout][aout].
    """
    if not used_clip_ids:
        raise RuntimeError("render_funnel_video: no used_clip_ids provided")

    out_path = os.path.join(session_dir, "final.mp4")

    filter_parts: List[str] = []
    v_labels: List[str] = []
    a_labels: List[str] = []

    idx = 0
    lookup = {c["id"]: c for c in clips}

    for cid in used_clip_ids:
        c = lookup.get(cid)
        if not c:
            continue
        start = safe_float(c.get("start", 0.0))
        end = safe_float(c.get("end", 0.0))
        if end <= start:
            continue

        v_label = f"v{idx}"
        a_label = f"a{idx}"

        # video chain
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{v_label}]"
        )
        # audio chain
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{a_label}]"
        )

        v_labels.append(f"[{v_label}]")
        a_labels.append(f"[{a_label}]")
        idx += 1

    n = idx
    if n == 0:
        raise RuntimeError("render_funnel_video: no valid clips after trimming")

    # concat
    filter_parts.append(
        f"{''.join(v_labels)}{''.join(a_labels)}concat=n={n}:v=1:a=1[vout][aout]"
    )

    filter_complex = "; ".join(filter_parts)

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
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        out_path,
    ]

    logger.info("Running ffmpeg to render funnel video")
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    return out_path


# ==========
# S3 UPLOAD
# ==========

def upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    if not S3_BUCKET or boto3 is None:
        return None
    key = f"{S3_PREFIX.rstrip('/')}/{session_id}-final.mp4"
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=604800,  # 7 días
        )
        return url
    except Exception as e:
        logger.warning(f"S3 upload failed: {e}")
        return None


# ==========
# MAIN PIPELINE
# ==========

def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Entrada principal que usa tasks.job_render.

    - session_id
    - file_urls: lista de URLs (por ahora usamos sólo el primero).
    """
    logger.info(f"run_pipeline(session_id={session_id}, file_urls={file_urls})")
    session_dir = ensure_session_dir(session_id)

    if not file_urls:
        raise ValueError("run_pipeline: file_urls is empty")

    # 1) Download
    input_local = download_to_local(file_urls[0], session_dir)

    # 2) ASR
    asr_segments = run_asr(input_local)
    duration_sec = asr_segments[-1]["end"] if asr_segments else 0.0

    # 3) Micro-cuts
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 4) LLM classify cada clip
    for c in clips:
        classify_clip_with_llm(c)
        # mantener placeholders visuales (por ahora todos buenos)
        c["visual_score"] = 1.0
        c["face_q"] = 1.0
        c["scene_q"] = 1.0
        # vtx_sim por ahora 0 (si luego metemos embeddings lo usamos)
        c["vtx_sim"] = float(c.get("vtx_sim", 0.0))
        c.setdefault("visual_flags", {"scene_jump": False, "motion_jump": False})

    # 5) Composer V2 (free-flow + dedupe suave + orden)
    composer, slots, composer_human = funnel_composer(clips)

    used_clip_ids = composer["used_clip_ids"]

    # 6) Render
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)
    output_video_url = upload_to_s3(final_path, session_id)

    result: Dict[str, Any] = {
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": {
            k: v for k, v in slots.items()
        },
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": final_path,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }

    return result
