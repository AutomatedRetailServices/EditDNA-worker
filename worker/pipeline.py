import os
import re
import json
import math
import uuid
import logging
import subprocess
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

# Para esta versión: prioridad = mantener discurso natural
# Ignoramos env alto y usamos un valor fijo más suave
MIN_SEMANTIC_KEEP = 0.30
DUP_SIM_THRESHOLD = 0.90  # más agresivo que antes

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


def looks_like_meta_filler(text: str) -> bool:
    """
    Heurística para redos / meta / basura que el LLM a veces marca mal.
    """
    t = norm_text(text)
    # patrones obvios de "redo / meta"
    bad_fragments = [
        "is that good",
        "am i saying it right",
        "say it right",
        "start again",
        "let me start again",
        "try again",
        "that one good",
        "why cant i remember",
        "can we cut this",
        "wait you",
        "wait no",
        "no thats not right",
        "thanks",
        "thank you guys",  # típico cierre random
    ]
    for frag in bad_fragments:
        if frag in t:
            return True

    # preguntas muy cortas con pocas palabras → casi siempre meta
    if t.endswith("?") and len(t.split()) <= 4:
        return True

    return False


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
# MICRO-CUTS (SENTENCE BOUNDARY + PADDING SUAVE)
# ==========

def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Partimos por oraciones, con un poquito de aire antes/después.
    """
    clips: List[Dict[str, Any]] = []
    global_idx = 0

    PUNCT = {".", "!", "?"}
    MAX_SENT_DURATION = 10.0
    GAP_THRESHOLD = 0.6
    PADDING = 0.10  # 100ms antes/después

    for seg in asr_segments:
        words = seg.get("words") or []
        if not words:
            text = seg.get("text", "").strip()
            if not text:
                continue
            # padding al nivel de segmento
            s = max(0.0, safe_float(seg.get("start", 0.0)) - PADDING)
            e = safe_float(seg.get("end", 0.0)) + PADDING
            clip_id = f"ASR{global_idx:04d}_c0"
            clip = {
                "id": clip_id,
                "slot": "UNLABELED",
                "start": s,
                "end": e,
                "score": 0.0,
                "semantic_score": 0.0,
                "visual_score": 1.0,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [clip_id],
                "text": text,
                "llm_reason": "",
                "visual_flags": {"scene_jump": False, "motion_jump": False},
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

            is_punct = len(w_text) > 0 and w_text[-1] in PUNCT
            duration = buffer[-1]["end"] - buffer[0]["start"]
            too_long = duration >= MAX_SENT_DURATION
            big_gap = gap >= GAP_THRESHOLD

            if is_punct or too_long or big_gap:
                sent_start = max(0.0, buffer[0]["start"] - PADDING)
                sent_end = buffer[-1]["end"] + PADDING
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
                        "visual_flags": {"scene_jump": False, "motion_jump": False},
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

        if buffer:
            sent_start = max(0.0, buffer[0]["start"] - PADDING)
            sent_end = buffer[-1]["end"] + PADDING
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
                    "visual_flags": {"scene_jump": False, "motion_jump": False},
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
# LLM CLASSIFIER (solo para slot + keep + scores)
# ==========

CLASSIFIER_SYSTEM_PROMPT = """
You label very short spoken sentences from UGC ads.

For each sentence return JSON:
- slot: one of ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]
- keep: true/false (false ONLY if it's clearly redo/meta/filler)
- reason: one short sentence.
- semantic_score: float [0,1]
- score: float [0,1] (usually same as semantic_score)
"""


def classify_clip_with_llm(clip: Dict[str, Any]) -> None:
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
        "content": json.dumps({"sentence": text}, ensure_ascii=False),
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
            "reason": "Fallback keep.",
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

    # override manual para meta/filler obvio
    if looks_like_meta_filler(text):
        keep = False
        semantic_score = 0.0
        score = 0.0
        if not reason:
            reason = "Overridden as meta/filler by heuristic."

    clip["slot"] = slot
    clip["llm_reason"] = reason
    clip["semantic_score"] = semantic_score
    clip["score"] = score
    clip["meta"]["slot"] = slot
    clip["meta"]["semantic_score"] = semantic_score
    clip["meta"]["score"] = score
    clip["meta"]["keep"] = keep


# ==========
# LIMPIEZA DE TIMELINE (SIN REORDENAR HISTORIA)
# ==========

def build_cleansed_timeline(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    - Mantiene ORDEN CRONOLÓGICO.
    - Saca redos/filler (keep=False o semantic < MIN_SEMANTIC_KEEP).
    - Deduplica frases casi iguales (nos quedamos con la mejor versión).
    - NO fuerza estructura de funnel todavía (solo metadata).
    """
    by_id = {c["id"]: c for c in clips}
    sorted_clips = sorted(clips, key=lambda c: safe_float(c.get("start", 0.0)))

    # 1) Filtro por keep + semantic_score
    candidates: List[Dict[str, Any]] = []
    for c in sorted_clips:
        keep = bool(c.get("meta", {}).get("keep", True))
        if not keep:
            c["meta"]["keep"] = False
            continue
        sem = float(c.get("semantic_score", 0.0))
        if sem < MIN_SEMANTIC_KEEP:
            c["meta"]["keep"] = False
            continue
        c["meta"]["keep"] = True
        candidates.append(c)

    # 2) Dedup (texto casi igual o uno es subcadena del otro)
    final: List[Dict[str, Any]] = []
    chosen_norms: List[str] = []

    for c in candidates:
        t = c.get("text", "")
        norm = norm_text(t)
        if not norm:
            c["meta"]["keep"] = False
            continue

        best_sim = 0.0
        best_idx = -1
        for idx, existing_norm in enumerate(chosen_norms):
            sim = text_similarity(norm, existing_norm)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        def is_substring_relation(a: str, b: str) -> bool:
            # a y/o b son subcadena con suficiente overlap
            if not a or not b:
                return False
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if shorter in longer and len(shorter) / len(longer) >= 0.7:
                return True
            return False

        dup_idx = None
        if best_sim >= DUP_SIM_THRESHOLD or (
            best_idx >= 0 and is_substring_relation(norm, chosen_norms[best_idx])
        ):
            dup_idx = best_idx

        if dup_idx is not None and dup_idx >= 0:
            existing_clip = final[dup_idx]
            # elegir el de mayor score/semantic_score
            if float(c.get("score", 0.0)) > float(existing_clip.get("score", 0.0)):
                existing_clip["meta"]["keep"] = False
                c["meta"]["keep"] = True
                final[dup_idx] = c
                chosen_norms[dup_idx] = norm
            else:
                c["meta"]["keep"] = False
        else:
            c["meta"]["keep"] = True
            final.append(c)
            chosen_norms.append(norm)

    used_clip_ids = [c["id"] for c in final]

    # Construimos listas por slot solo para stats/metadatos
    slots_map: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURES": [],
        "PROOF": [],
        "CTA": [],
    }
    for c in final:
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

    lines = []
    lines.append("===== EDITDNA CLEAN TIMELINE (NO REORDER) =====")
    for idx, cid in enumerate(used_clip_ids, start=1):
        c = by_id[cid]
        lines.append(f"{idx}) [{c.get('slot','?')}] {cid} → \"{c.get('text','').strip()}\"")
    lines.append("")
    lines.append("===============================================")
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
        "min_score": MIN_SEMANTIC_KEEP,
    }

    return composer, slots_map, composer_human


# ==========
# RENDER – FFMPEG (SIN ALTERAR VELOCIDAD)
# ==========

def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
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

        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{v_label}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{a_label}]"
        )

        v_labels.append(f"[{v_label}]")
        a_labels.append(f"[{a_label}]")
        idx += 1

    n = idx
    if n == 0:
        raise RuntimeError("render_funnel_video: no valid clips after trimming")

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

    logger.info("Running ffmpeg to render (clean-timeline) video")
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
            ExpiresIn=604800,
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
    Versión V1: limpieza de redos/filler + dedupe + orden original.
    NO reordenamos por funnel todavía.
    """
    logger.info(f"run_pipeline(session_id={session_id}, file_urls={file_urls})")
    session_dir = ensure_session_dir(session_id)

    if not file_urls:
        raise ValueError("run_pipeline: file_urls is empty")

    input_local = download_to_local(file_urls[0], session_dir)

    # 1) ASR
    asr_segments = run_asr(input_local)
    duration_sec = asr_segments[-1]["end"] if asr_segments else 0.0

    # 2) Micro-cuts
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 3) LLM classify
    for c in clips:
        classify_clip_with_llm(c)
        c["visual_score"] = 1.0
        c["face_q"] = 1.0
        c["scene_q"] = 1.0
        c.setdefault("visual_flags", {"scene_jump": False, "motion_jump": False})

    # 4) Limpieza de timeline (sin reorden)
    composer, slots, composer_human = build_cleansed_timeline(clips)
    used_clip_ids = composer["used_clip_ids"]

    # 5) Render
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)
    output_video_url = upload_to_s3(final_path, session_id)

    result: Dict[str, Any] = {
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": {k: v for k, v in slots.items()},
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": final_path,
        "output_video_url": output_video_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }

    return result
