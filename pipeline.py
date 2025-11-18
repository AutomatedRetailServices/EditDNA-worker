import os
import io
import re
import math
import json
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import requests
import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util as st_util

# Opcional / LLM
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============
# ENV CONFIG
# ============

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")  # "cuda" o "cpu"

S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# 🎯 Free-flow composer (no límites duros, solo filtros suaves)
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.70"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.80"))
COMPOSER_MAX_TOTAL_CLIPS = int(os.environ.get("COMPOSER_MAX_TOTAL_CLIPS", "16"))

# Veto para basura extrema
VETO_MIN_SCORE = float(os.environ.get("VETO_MIN_SCORE", "0.30"))

# Micro-cuts
SILENCE_GAP_SEC = float(os.environ.get("SILENCE_GAP_SEC", "0.50"))  # pausa entre palabras para cortar
MIN_SENTENCE_DURATION = float(os.environ.get("MIN_SENTENCE_DURATION", "0.80"))  # no micro fragmentos ridículos
MAX_SENTENCE_DURATION = float(os.environ.get("MAX_SENTENCE_DURATION", "8.0"))   # evita bloques demasiado largos

# ============
# MODELOS GLOBALES (cargan una sola vez)
# ============

_whisper_model: Optional[WhisperModel] = None
_embed_model: Optional[SentenceTransformer] = None
_openai_client: Optional["OpenAI"] = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"🧠 Loading Whisper model: {WHISPER_MODEL_NAME} on {WHISPER_DEVICE}")
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8")
    return _whisper_model


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("🧠 Loading sentence-transformers model: all-MiniLM-L6-v2")
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


def get_openai_client() -> Optional["OpenAI"]:
    global _openai_client
    if not HAS_OPENAI or not OPENAI_API_KEY:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ============
# HELPERS I/O
# ============

def download_video_to_temp(url: str, session_id: str) -> str:
    """
    Descarga un MP4 desde URL a /tmp/TMP/editdna/{session_id}/input.mp4
    """
    tmp_dir = os.path.join("/tmp", "TMP", "editdna", session_id)
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = os.path.join(tmp_dir, "input.mp4")

    logger.info(f"⬇️ Downloading video from {url} to {local_path}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


def upload_to_s3(local_path: str, session_id: str) -> Tuple[str, str]:
    """
    Sube el archivo final a S3 (si S3_BUCKET está configurado) y devuelve:
    - s3_uri
    - presigned_url
    Si no hay bucket, devuelve (local_path, "").
    """
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not set, skipping upload.")
        return local_path, ""

    key = f"{S3_PREFIX}/{session_id}-final.mp4"
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info(f"☁️ Uploading final video to s3://{S3_BUCKET}/{key}")
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=7 * 24 * 3600,  # 7 días
        )
    except Exception:
        logger.exception("Failed to generate presigned URL")
        url = ""

    return f"s3://{S3_BUCKET}/{key}", url


# ============
# ASR + MICRO-CUTS
# ============

def run_whisper_sentence_pass(video_path: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Corre Whisper con word_timestamps y devuelve:
    - sentences: lista de dicts {id, start, end, text}
    - duration_sec del audio/video
    Hace micro-cortes por oración/pausa (micro-cut intelligence).
    """
    model = get_whisper_model()
    logger.info(f"🗣️ Running Whisper on {video_path}")

    # MoviePy solo para duración / FPS
    with VideoFileClip(video_path) as base_clip:
        duration_sec = float(base_clip.duration or 0.0)
        fps = float(base_clip.fps or 30.0)

    # Faster-Whisper con timestamps de palabras
    segments, _info = model.transcribe(
        audio=video_path,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    sentences: List[Dict[str, Any]] = []
    sent_idx = 0

    for seg in segments:
        words = seg.words or []
        if not words:
            continue

        current_start = words[0].start
        current_words = []

        def flush_sentence(end_time: float):
            nonlocal sentences, sent_idx, current_start, current_words
            if not current_words:
                return
            text = " ".join(w.word for w in current_words).strip()
            if not text:
                current_words = []
                return
            start = float(current_start)
            end = float(end_time)

            # Duración mínima / máxima para evitar basura
            if (end - start) < MIN_SENTENCE_DURATION:
                current_words = []
                return
            if (end - start) > MAX_SENTENCE_DURATION:
                # igual la agregamos, pero podría ser cortada más adelante si quieres
                pass

            sent_id = f"SENT_{sent_idx:04d}"
            sentences.append(
                {
                    "id": sent_id,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
            sent_idx += 1
            current_words = []

        prev_word_end = words[0].start
        for w in words:
            # corte por signo de puntuación final o silencio largo
            gap = (w.start - prev_word_end) if prev_word_end is not None else 0.0
            if current_words:
                last_text = current_words[-1].word
                if last_text and last_text[-1] in ".?!":
                    # cierre por puntuación
                    flush_sentence(prev_word_end)
                    current_start = w.start
            if gap > SILENCE_GAP_SEC and current_words:
                # corte por silencio
                flush_sentence(prev_word_end)
                current_start = w.start

            current_words.append(w)
            prev_word_end = w.end

        # flush al final de este segmento
        if current_words:
            flush_sentence(current_words[-1].end)

    # Si por alguna razón no salió nada, fallback a segmentos completos
    if not sentences:
        logger.warning("No micro-sentences produced, falling back to raw segments.")
        idx = 0
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            sentences.append(
                {
                    "id": f"SENT_{idx:04d}",
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )
            idx += 1

    logger.info(f"🎯 Whisper sentence-pass produced {len(sentences)} sentences.")
    return sentences, duration_sec


# ============
# LLM CLASSIFICATION (SLOTS + KEEP + REASON)
# ============

FILLER_PATTERNS = [
    r"\bwait\b",
    r"\bhold on\b",
    r"\blet me start again\b",
    r"\bstart again\b",
    r"\bdo it again\b",
    r"\bi don't know\b",
    r"\bthat was bad\b",
    r"\bis that good\?\b",
    r"\bam i saying it right\?\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bokay,? okay\b",
    r"\bis that funny\?\b",
]

_filler_regex = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


def looks_like_filler(text: str) -> bool:
    """Detecta redos, relleno, dudas, risas, etc."""
    if not text:
        return True
    if len(text.strip()) < 4:
        return True
    if _filler_regex.search(text):
        return True
    return False


def llm_classify_segments(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Usa OpenAI (si está disponible) para asignar:
      - slot (HOOK/STORY/PROBLEM/BENEFITS/FEATURES/PROOF/CTA)
      - keep (True/False)
      - reason
      - semantic_score (0–1)

    Devuelve un dict: {segment_id: {...}}
    Si no hay OpenAI, hace una clasificación de fallback por reglas.
    """
    client = get_openai_client()
    out: Dict[str, Dict[str, Any]] = {}

    # fallback RULE-BASED si no hay OpenAI
    if client is None:
        logger.warning("OpenAI client not available, using rule-based classification.")
        for seg in segments:
            text = seg["text"]
            seg_id = seg["id"]
            if looks_like_filler(text):
                out[seg_id] = {
                    "slot": "HOOK",
                    "keep": False,
                    "semantic_score": 0.2,
                    "reason": "Filler / redo / meta-comment, not useful for the ad.",
                }
                continue

            lower = text.lower()
            slot = "FEATURES"
            if "click the link" in lower or "get yours" in lower or "check them out" in lower:
                slot = "CTA"
            elif "support" in lower or "you only need to take" in lower or "feel fresh" in lower:
                slot = "BENEFITS"
            elif "because i found" in lower or "is your" in lower or "utis" in lower or "odor" in lower:
                slot = "HOOK"
            elif "review" in lower or "i think" in lower or "every time i" in lower:
                slot = "PROOF"

            # Baseline semantic score
            semantic_score = 0.85
            out[seg_id] = {
                "slot": slot,
                "keep": True,
                "semantic_score": semantic_score,
                "reason": f"Rule-based classification as {slot}.",
            }
        return out

    # Con OpenAI
    logger.info(f"🤖 Classifying {len(segments)} segments via OpenAI: {OPENAI_MODEL}")

    # construimos una lista compacta para el prompt
    items_for_prompt = [
        {"id": seg["id"], "text": seg["text"]} for seg in segments
    ]

    system_msg = (
        "You are an expert UGC ad editor for TikTok Shop.\n"
        "Given short spoken segments (1–8 seconds), decide:\n"
        "- slot: one of [HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA]\n"
        "- keep: true if the segment should be used in the final ad; false if it is a redo, filler, 'wait', 'I don't know', meta-talk, or mistake.\n"
        "- semantic_score: 0.0–1.0 measuring how strong/clear/useful the line is.\n"
        "- reason: very brief explanation.\n\n"
        "Rules:\n"
        "- HOOK: bold opening, questions, strong pattern interrupts, 'Is your X doing Y?'.\n"
        "- PROBLEM: describing pain, issues, complaints.\n"
        "- BENEFITS: outcomes/transformations ('you'll feel', 'you can', 'you'll get').\n"
        "- FEATURES: ingredients, specs, how it works, 'each gummy has', 'made for women'.\n"
        "- PROOF: testimonials, social proof, 'I get so many compliments', 'I think these are amazing'.\n"
        "- CTA: instructions to click, buy, check the link.\n"
        "- If the line is clearly a redo ('wait', 'am I saying it right', 'start again'), set keep=false and semantic_score<=0.3.\n"
        "- Output a JSON object mapping id -> {slot, keep, semantic_score, reason}.\n"
    )

    user_msg = (
        "Classify the following segments:\n\n"
        + json.dumps(items_for_prompt, ensure_ascii=False, indent=2)
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        # esperamos un dict {id: {slot, keep, semantic_score, reason}}
        for seg in segments:
            sid = seg["id"]
            item = parsed.get(sid)
            if not item:
                # fallback local si ese id no viene
                text = seg["text"]
                if looks_like_filler(text):
                    out[sid] = {
                        "slot": "HOOK",
                        "keep": False,
                        "semantic_score": 0.2,
                        "reason": "Filler / redo / meta-comment detected locally.",
                    }
                else:
                    out[sid] = {
                        "slot": "FEATURES",
                        "keep": True,
                        "semantic_score": 0.8,
                        "reason": "Fallback local classification as FEATURES.",
                    }
                continue

            slot = item.get("slot", "FEATURES").upper()
            if slot not in ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]:
                slot = "FEATURES"
            keep = bool(item.get("keep", True))
            try:
                semantic_score = float(item.get("semantic_score", 0.8))
            except Exception:
                semantic_score = 0.8
            reason = item.get("reason", "")
            out[sid] = {
                "slot": slot,
                "keep": keep,
                "semantic_score": semantic_score,
                "reason": reason,
            }

    except Exception:
        logger.exception("OpenAI classification failed, falling back to rule-based.")
        return llm_classify_segments_fallback(segments)

    return out


def llm_classify_segments_fallback(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Rule-based fallback si falla OpenAI.
    Separado para poder llamar desde arriba.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        text = seg["text"]
        seg_id = seg["id"]
        if looks_like_filler(text):
            out[seg_id] = {
                "slot": "HOOK",
                "keep": False,
                "semantic_score": 0.2,
                "reason": "Filler / redo / meta-comment, not useful for the ad.",
            }
            continue

        lower = text.lower()
        slot = "FEATURES"
        if "click the link" in lower or "get yours" in lower or "check them out" in lower:
            slot = "CTA"
        elif "support" in lower or "you only need to take" in lower or "feel fresh" in lower:
            slot = "BENEFITS"
        elif "because i found" in lower or "is your" in lower or "utis" in lower or "odor" in lower:
            slot = "HOOK"
        elif "review" in lower or "i think" in lower or "every time i" in lower:
            slot = "PROOF"

        semantic_score = 0.85
        out[seg_id] = {
            "slot": slot,
            "keep": True,
            "semantic_score": semantic_score,
            "reason": f"Rule-based classification as {slot}.",
        }
    return out


# ============
# DEDUPE & COMPOSER FREE-FLOW
# ============

def dedupe_segments_by_text(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Elimina duplicados muy similares usando embeddings (sentence-transformers)
    según COMPOSER_DUP_SIM_THRESHOLD.
    Respeta el orden original y se queda con el primero de cada grupo similar.
    """
    if len(segments) <= 1:
        return segments

    texts = [s["text"] for s in segments]
    model = get_embed_model()
    embs = model.encode(texts, convert_to_tensor=True)

    keep_flags = [True] * len(segments)

    for i in range(len(segments)):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(segments)):
            if not keep_flags[j]:
                continue
            sim = float(st_util.pytorch_cos_sim(embs[i], embs[j]).item())
            if sim >= COMPOSER_DUP_SIM_THRESHOLD:
                # marcamos j como duplicado
                keep_flags[j] = False

    deduped = [seg for seg, k in zip(segments, keep_flags) if k]
    return deduped


def build_clips_and_composer(
    sentences: List[Dict[str, Any]],
    cls_info: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any], str]:
    """
    Construye:
      - clips: lista plana de todos los segmentos con metadata
      - slots: dict slot -> lista de clips
      - composer: dict con ids elegidos por funnel
      - composer_human: string legible
    """
    clips: List[Dict[str, Any]] = []
    slots_map: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURES": [],
        "PROOF": [],
        "CTA": [],
    }

    # 1) Construir clips básicos a partir de sentences + cls_info
    for sent in sentences:
        sid = sent["id"]
        start = float(sent["start"])
        end = float(sent["end"])
        text = sent["text"].strip()

        info = cls_info.get(sid, {})
        slot = info.get("slot", "FEATURES")
        keep_flag = bool(info.get("keep", True))
        semantic_score = float(info.get("semantic_score", 0.8))
        reason = info.get("reason", "")

        # Veto global
        if semantic_score < VETO_MIN_SCORE:
            keep_flag = False

        # score final
        score = semantic_score

        clip = {
            "id": sid,
            "slot": slot,
            "start": start,
            "end": end,
            "score": score,
            "semantic_score": semantic_score,
            "visual_score": 1.0,   # placeholder (ya tienes visión integrada arriba con 'vision': true)
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [sid],
            "text": text,
            "llm_reason": reason,
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
                "keep": False,  # lo ajustamos luego para los que se usen
            },
        }

        clips.append(clip)
        if slot in slots_map:
            slots_map[slot].append(clip)
        else:
            slots_map[slot] = [clip]

    # 2) Ordenar internamente por score descendente
    for slot, lst in slots_map.items():
        lst.sort(key=lambda c: c["score"], reverse=True)

    # 3) Dedup dentro de cada slot (por texto)
    for slot in list(slots_map.keys()):
        slots_map[slot] = dedupe_segments_by_text(slots_map[slot])

    # 4) Free-flow COMPOSER (sin límites duros)
    used_clip_ids: List[str] = []

    # HOOK → el mejor HOOK que pase threshold
    hook_id = None
    for c in slots_map.get("HOOK", []):
        if c["semantic_score"] >= COMPOSER_MIN_SEMANTIC and c["id"] not in used_clip_ids:
            hook_id = c["id"]
            used_clip_ids.append(c["id"])
            break

    def gather(slot_name: str) -> List[str]:
        ids: List[str] = []
        for c in slots_map.get(slot_name, []):
            if c["semantic_score"] < COMPOSER_MIN_SEMANTIC:
                continue
            if c["id"] in used_clip_ids:
                continue
            ids.append(c["id"])
        return ids

    benefit_ids = gather("BENEFITS")
    feature_ids = gather("FEATURES")
    proof_ids = gather("PROOF")
    problem_ids = gather("PROBLEM")
    story_ids = gather("STORY")

    # CTA → mejor CTA
    cta_id = None
    for c in slots_map.get("CTA", []):
        if c["semantic_score"] >= COMPOSER_MIN_SEMANTIC and c["id"] not in used_clip_ids:
            cta_id = c["id"]
            break

    # Orden de funnel: HOOK → PROBLEM → STORY → BENEFITS → FEATURES → PROOF → CTA
    funnel_order: List[str] = []
    if hook_id:
        funnel_order.append(hook_id)

    funnel_order.extend(problem_ids)
    funnel_order.extend(story_ids)
    funnel_order.extend(benefit_ids)
    funnel_order.extend(feature_ids)
    funnel_order.extend(proof_ids)
    if cta_id:
        funnel_order.append(cta_id)

    # Si excede máximo total, recortamos al final
    if len(funnel_order) > COMPOSER_MAX_TOTAL_CLIPS:
        funnel_order = funnel_order[:COMPOSER_MAX_TOTAL_CLIPS]

    # marcar meta.keep True para los usados
    for clip in clips:
        if clip["id"] in funnel_order:
            clip["meta"]["keep"] = True

    composer = {
        "hook_id": hook_id,
        "story_ids": story_ids,
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": feature_ids,
        "proof_ids": proof_ids,
        "cta_id": cta_id,
        "used_clip_ids": funnel_order,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }

    # composer_human bonito
    def line(clip_id: str) -> str:
        clip = next((c for c in clips if c["id"] == clip_id), None)
        if not clip:
            return f"{clip_id} → (missing)"
        return f"{clip_id} → \"{clip['text']}\""

    buf = []
    buf.append("===== EDITDNA FUNNEL COMPOSER =====")

    buf.append("HOOK:")
    if hook_id:
        hclip = next(c for c in clips if c["id"] == hook_id)
        buf.append(f"  [{hook_id}] score={hclip['score']:.2f} → \"{hclip['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("STORY:")
    if story_ids:
        for cid in story_ids:
            c = next(cc for cc in clips if cc["id"] == cid)
            buf.append(f"  [{cid}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("PROBLEM:")
    if problem_ids:
        for cid in problem_ids:
            c = next(cc for cc in clips if cc["id"] == cid)
            buf.append(f"  [{cid}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("BENEFITS:")
    if benefit_ids:
        for cid in benefit_ids:
            c = next(cc for cc in clips if cc["id"] == cid)
            buf.append(f"  [{cid}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("FEATURES:")
    if feature_ids:
        for cid in feature_ids:
            c = next(cc for cc in clips if cc["id"] == cid)
            buf.append(f"  [{cid}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("PROOF:")
    if proof_ids:
        for cid in proof_ids:
            c = next(cc for cc in clips if cc["id"] == cid)
            buf.append(f"  [{cid}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("CTA:")
    if cta_id:
        c = next(cc for cc in clips if cc["id"] == cta_id)
        buf.append(f"  [{cta_id}] score={c['score']:.2f} → \"{c['text']}\"")
    else:
        buf.append("  (none)")

    buf.append("")
    buf.append("FINAL ORDER TIMELINE:")
    for idx, cid in enumerate(funnel_order, start=1):
        buf.append(f"{idx}) {line(cid)}")

    buf.append("")
    buf.append("=====================================")
    composer_human = "\n".join(buf)

    return clips, slots_map, composer, composer_human


# ============
# VIDEO CUTTING (SYNC AUDIO/VIDEO)
# ============

def cut_and_export_video(
    video_path: str,
    session_id: str,
    used_clip_ids: List[str],
    all_clips: List[Dict[str, Any]],
) -> Tuple[str, float]:
    """
    Corta el video original según used_clip_ids en orden funnel
    y exporta final.mp4 con audio sync usando el mismo FPS del original.
    """
    tmp_dir = os.path.join("/tmp", "TMP", "editdna", session_id)
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, "final.mp4")

    if not used_clip_ids:
        # Si no hay nada que usar, devolvemos el original
        logger.warning("No used_clip_ids, returning original video as final.")
        return video_path, float(VideoFileClip(video_path).duration or 0.0)

    id_to_clip = {c["id"]: c for c in all_clips}

    clips = []
    with VideoFileClip(video_path) as base:
        fps = float(base.fps or 30.0)

        for cid in used_clip_ids:
            clip_meta = id_to_clip.get(cid)
            if not clip_meta:
                continue
            start = max(0.0, float(clip_meta["start"]))
            end = max(start + 0.05, float(clip_meta["end"]))
            if end > base.duration:
                end = base.duration

            sub = base.subclip(start, end)
            clips.append(sub)

        if not clips:
            logger.warning("No valid subclips, returning original video.")
            return video_path, float(base.duration or 0.0)

        final_clip = concatenate_videoclips(clips, method="compose")

        logger.info(f"💾 Writing final video to {out_path} (fps={fps})")
        # clave: usar fps del original + temp audiofile para sync
        final_clip.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(tmp_dir, "temp-audio.m4a"),
            remove_temp=True,
            fps=fps,
            verbose=False,
            logger=None,
        )

        final_duration = float(final_clip.duration or 0.0)
        final_clip.close()

    return out_path, final_duration


# ============
# ENTRYPOINT
# ============

def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Punto principal llamado desde tasks.job_render.

    - Descarga el primer video
    - ASR + sentence-boundary (micro-cuts)
    - LLM classification (slots + keep + reason)
    - Free-flow composer (sin límites duros, dedupe texto)
    - Corte de video respetando sync audio/video
    - Upload a S3 (si S3_BUCKET está set)
    """
    logger.info(f"🚀 run_pipeline(session_id={session_id}, file_urls={file_urls})")

    if not file_urls:
        raise ValueError("run_pipeline requires at least one file URL")

    # 1) Download
    input_url = file_urls[0]
    input_local = download_video_to_temp(input_url, session_id)

    # 2) ASR + micro-sentences
    sentences, duration_sec = run_whisper_sentence_pass(input_local)

    # 3) LLM classify
    cls_info = llm_classify_segments(sentences)

    # 4) Build clips + composer (free-flow)
    clips, slots_map, composer, composer_human = build_clips_and_composer(sentences, cls_info)

    # 5) Cut + export video
    final_local_path, final_duration = cut_and_export_video(
        input_local,
        session_id,
        composer["used_clip_ids"],
        clips,
    )

    # 6) Upload S3 (opcional)
    s3_uri, presigned_url = upload_to_s3(final_local_path, session_id)

    # 7) Armar respuesta
    result = {
        "ok": True,
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots_map,
        "composer": composer,
        "composer_human": composer_human,
        "output_video_local": final_local_path,
        "output_video_url": presigned_url or final_local_path,
        "asr": True,
        "semantic": True,
        "vision": True,
        "s3_uri": s3_uri,
    }

    logger.info("✅ run_pipeline finished successfully")
    return result
