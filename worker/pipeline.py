import os
import io
import json
import math
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
import boto3
from botocore.client import Config
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, concatenate_videoclips
from openai import OpenAI

# ---------------------------------------------------------
# 🔧 Logging
# ---------------------------------------------------------
logger = logging.getLogger("editdna.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# 🔧 Environment / Config
# ---------------------------------------------------------
TMP_ROOT = os.environ.get("EDITDNA_TMP_ROOT", "/tmp/TMP/editdna")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))

ASR_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # e.g. tiny, base, small, medium

# Composer / dedup
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.70"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.82"))

# Visual / veto (stubbed, pero dejamos la env por compatibilidad)
VETO_MIN_SCORE = float(os.environ.get("VETO_MIN_SCORE", "0.40"))

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
S3_BASE_PREFIX = os.environ.get("S3_BASE_PREFIX", "editdna/outputs")

# ---------------------------------------------------------
# 🔧 OpenAI client
# ---------------------------------------------------------
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# ---------------------------------------------------------
# 🔧 S3 client
# ---------------------------------------------------------
_s3_client: Optional[Any] = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            config=Config(signature_version="s3v4"),
        )
    return _s3_client


def upload_to_s3(local_path: str, session_id: str) -> Optional[str]:
    """
    Sube el archivo final a S3 si S3_BUCKET está definido.
    Retorna URL firmado (si posible) o None.
    """
    if not S3_BUCKET:
        logger.info("S3_BUCKET not set, skipping upload")
        return None

    s3 = get_s3_client()
    filename = os.path.basename(local_path)
    key = f"{S3_BASE_PREFIX}/{session_id}-{filename}"

    logger.info("☁️ Uploading to S3", extra={"bucket": S3_BUCKET, "key": key})
    s3.upload_file(local_path, S3_BUCKET, key)

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=7 * 24 * 3600,  # 7 días
        )
        logger.info("✅ S3 presigned URL created", extra={"url": url})
        return url
    except Exception:
        logger.exception("Failed to create presigned URL")
        return None


# ---------------------------------------------------------
# 🔧 Utilidades de texto / similitud
# ---------------------------------------------------------
def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())


def simple_similarity(a: str, b: str) -> float:
    """
    Similitud muy simple basada en tokens (no embeddings).
    Suficiente para detectar duplicados texto casi igual.
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    set_a = set(a_norm.split())
    set_b = set(b_norm.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b) or 1
    return inter / union


# ---------------------------------------------------------
# 🔧 Descarga de video
# ---------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_to_local(url: str, dest_path: str) -> str:
    ensure_dir(os.path.dirname(dest_path))
    logger.info("⬇️ Downloading video", extra={"url": url, "dest": dest_path})
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest_path


# ---------------------------------------------------------
# 🔧 ASR con faster-whisper
# ---------------------------------------------------------
_asr_model: Optional[WhisperModel] = None


def get_asr_model() -> WhisperModel:
    global _asr_model
    if _asr_model is None:
        logger.info("🧠 Loading faster-whisper model", extra={"size": ASR_MODEL_SIZE})
        _asr_model = WhisperModel(
            ASR_MODEL_SIZE,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            compute_type="float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "int8",
        )
    return _asr_model


def run_asr(input_local: str) -> Tuple[List[Dict[str, Any]], float, str]:
    """
    Corre ASR sobre el video y devuelve:
      - segments: lista de dicts {start, end, text}
      - duration_sec: duración aproximada
      - language: código de idioma
    """
    model = get_asr_model()
    logger.info("🎙️ Running ASR", extra={"input": input_local})

    segments_iter, info = model.transcribe(
        input_local,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=False,
    )

    segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments_iter):
        segments.append(
            {
                "id": f"ASR{i:04d}",
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )

    logger.info(
        "ASR produced %d segments, duration ~%.2fs",
        len(segments),
        info.duration,
    )

    language = info.language or "en"
    return segments, float(info.duration or 0.0), language


# ---------------------------------------------------------
# 🔧 LLM: clasificación de segmento (slot + keep + reason)
# ---------------------------------------------------------
def classify_segment_with_llm(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Pide a OpenAI que:
      - Asigne slot (HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA, FILLER)
      - Diga si keep = true/false
      - Dé una razón
      - Dé un score 0–1 (importance)
    """
    client = get_openai_client()

    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that labels short ad video segments "
                        "for a funnel editor (TikTok/UGC ads).\n"
                        "You MUST respond strictly as a JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Language: {language}\n\n"
                        f"Transcript segment:\n\"{text}\"\n\n"
                        "Classify this segment for a direct-response ad funnel.\n"
                        "Slots are:\n"
                        "- HOOK: strong opening question/claim, grabs attention.\n"
                        "- STORY: personal story or context.\n"
                        "- PROBLEM: describes pain, frustration, or issue.\n"
                        "- BENEFITS: positive outcomes or transformation.\n"
                        "- FEATURES: concrete details, ingredients, specs, how it works.\n"
                        "- PROOF: social proof, testimonials, results, credibility.\n"
                        "- CTA: call-to-action, telling the viewer what to do.\n"
                        "- FILLER: bloopers, redos, meta talk, confusion, camera mistakes.\n\n"
                        "Decide if this segment should be kept in the final ad. "
                        "FILLER almost always has keep=false.\n\n"
                        "Return JSON with keys:\n"
                        "{\n"
                        "  \"slot\": \"HOOK\" | \"STORY\" | \"PROBLEM\" | \"BENEFITS\" | \"FEATURES\" | \"PROOF\" | \"CTA\" | \"FILLER\",\n"
                        "  \"keep\": true/false,\n"
                        "  \"reason\": \"short explanation\",\n"
                        "  \"importance\": float between 0 and 1 (how strong/useful this is)\n"
                        "}\n"
                    ),
                },
            ]

            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            slot = str(data.get("slot", "FILLER")).upper().strip()
            if slot not in {
                "HOOK",
                "STORY",
                "PROBLEM",
                "BENEFITS",
                "FEATURES",
                "PROOF",
                "CTA",
                "FILLER",
            }:
                slot = "FILLER"

            keep = bool(data.get("keep", slot != "FILLER"))
            importance = float(data.get("importance", 0.7))
            importance = max(0.0, min(1.0, importance))
            reason = str(data.get("reason", "")).strip()

            return {
                "slot": slot,
                "keep": keep,
                "importance": importance,
                "reason": reason,
            }

        except Exception as e:
            logger.warning(
                "LLM classify attempt %d failed: %s", attempt + 1, str(e)
            )
            time.sleep(1.0)

    # Fallback
    return {
        "slot": "FILLER",
        "keep": False,
        "importance": 0.4,
        "reason": "LLM classification failed, treated as filler.",
    }


# ---------------------------------------------------------
# 🔧 Construcción de clips (micro-cuts)
# ---------------------------------------------------------
def build_clips_from_asr(
    segments: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """
    Toma segmentos de ASR (ya son micro-cuts) y los pasa por el LLM
    para slot + keep + reason. No limita slots: Free-Flow Mode.
    """
    clips: List[Dict[str, Any]] = []

    for seg in segments:
        seg_id = seg["id"]
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"]

        if not text.strip():
            continue

        llm = classify_segment_with_llm(text, language)
        slot = llm["slot"]

        # semantic_score = importance del LLM
        semantic_score = float(llm["importance"])
        # visual_score/face_q/scene_q stub en 1.0 (no visión aún)
        visual_score = 1.0
        face_q = 1.0
        scene_q = 1.0

        # score combinado (simple promedio)
        score = (semantic_score + visual_score) / 2.0

        # Heurística extra para redos/meta muy obvios
        txt_norm = normalize_text(text)
        obvious_filler_patterns = [
            "is that good",
            "was that good",
            "let me start again",
            "wait",
            "do it again",
            "i don't know how to do it",
            "okay okay",
            "am i saying it right",
            "is that funny",
        ]
        is_obvious_filler = any(p in txt_norm for p in obvious_filler_patterns)

        keep = bool(llm["keep"])
        if slot == "FILLER" or is_obvious_filler:
            keep = False

        clip = {
            "id": f"{seg_id}_c1",
            "slot": slot,
            "start": start,
            "end": end,
            "score": round(score, 2),
            "semantic_score": round(semantic_score, 2),
            "visual_score": round(visual_score, 2),
            "face_q": face_q,
            "scene_q": scene_q,
            "vtx_sim": 1.0,  # se ajusta luego para dedup si quieres
            "chain_ids": [f"{seg_id}_c1"],
            "text": text,
            "llm_reason": llm["reason"],
            "visual_flags": {
                "scene_jump": False,
                "motion_jump": False,
            },
            "meta": {
                "slot": slot,
                "semantic_score": round(semantic_score, 2),
                "visual_score": round(visual_score, 2),
                "score": round(score, 2),
                "chain_ids": [],
                "keep": keep,
            },
        }
        clips.append(clip)

    return clips


# ---------------------------------------------------------
# 🔧 Agrupar por slots + Free-Flow Composer (sin límites)
# ---------------------------------------------------------
def group_clips_by_slot(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
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
        slot = c.get("slot", "FILLER")
        if slot in slots:
            slots[slot].append(c)
    # Ordenar cada slot por tiempo
    for slot, arr in slots.items():
        arr.sort(key=lambda x: x["start"])
    return slots


def dedup_within_slot(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup muy simple basado en similitud de texto. No quita info dura,
    sólo elimina clones obvios (sim > COMPOSER_DUP_SIM_THRESHOLD).
    """
    kept: List[Dict[str, Any]] = []
    for c in clips:
        text = c.get("text", "")
        is_dup = False
        for kc in kept:
            sim = simple_similarity(text, kc.get("text", ""))
            if sim >= COMPOSER_DUP_SIM_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept


def compose_funnel(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Free-Flow Composer:
      - Respeta meta.keep (no limita # por slot).
      - Orden:
          1) HOOK
          2) STORY
          3) PROBLEM
          4) BENEFITS
          5) FEATURES
          6) PROOF
          7) CTA
    """
    kept_clips = [c for c in clips if c["meta"].get("keep", True)]

    # Filtro mínimo por semantic_score
    kept_clips = [
        c for c in kept_clips
        if c.get("semantic_score", 0.0) >= COMPOSER_MIN_SEMANTIC
    ]

    # Agrupar
    slots = group_clips_by_slot(kept_clips)

    # Dedup dentro de cada slot
    for slot, arr in slots.items():
        slots[slot] = dedup_within_slot(arr)

    order_slots = ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]

    ordered: List[Dict[str, Any]] = []
    for slot in order_slots:
        ordered.extend(slots[slot])

    used_clip_ids = [c["id"] for c in ordered]

    # armar ids por slot para metadata
    hook_id = slots["HOOK"][0]["id"] if slots["HOOK"] else None
    story_ids = [c["id"] for c in slots["STORY"]]
    problem_ids = [c["id"] for c in slots["PROBLEM"]]
    benefit_ids = [c["id"] for c in slots["BENEFITS"]]
    feature_ids = [c["id"] for c in slots["FEATURES"]]
    proof_ids = [c["id"] for c in slots["PROOF"]]
    cta_id = slots["CTA"][0]["id"] if slots["CTA"] else None

    # Texto humano para debug
    def line(c):
        return f"[{c['id']}] score={c['score']} → \"{c['text']}\""

    parts = ["===== EDITDNA FUNNEL COMPOSER ====="]
    parts.append("HOOK:")
    if hook_id:
        h = next(c for c in slots["HOOK"] if c["id"] == hook_id)
        parts.append(f"  {line(h)}")
    else:
        parts.append("  (none)")

    parts.append("STORY:")
    if story_ids:
        for sid in story_ids:
            c = next(x for x in slots["STORY"] if x["id"] == sid)
            parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("PROBLEM:")
    if problem_ids:
        for pid in problem_ids:
            c = next(x for x in slots["PROBLEM"] if x["id"] == pid)
            parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("BENEFITS:")
    if benefit_ids:
        for bid in benefit_ids:
            c = next(x for x in slots["BENEFITS"] if x["id"] == bid)
            parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("FEATURES:")
    if feature_ids:
        for fid in feature_ids:
            c = next(x for x in slots["FEATURES"] if x["id"] == fid)
            parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("PROOF:")
    if proof_ids:
        for pid in proof_ids:
            c = next(x for x in slots["PROOF"] if x["id"] == pid)
            parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("CTA:")
    if cta_id:
        c = next(x for x in slots["CTA"] if x["id"] == cta_id)
        parts.append(f"  {line(c)}")
    else:
        parts.append("  (none)")

    parts.append("")
    parts.append("FINAL ORDER TIMELINE:")
    for i, c in enumerate(ordered, start=1):
        parts.append(f"{i}) {c['id']} → \"{c['text']}\"")

    parts.append("")
    parts.append("=====================================")

    composer_human = "\n".join(parts)

    composer_meta = {
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

    return {
        "composer": composer_meta,
        "composer_human": composer_human,
        "slots": slots,
        "used_clip_ids": used_clip_ids,
    }


# ---------------------------------------------------------
# 🔧 Render final con MoviePy (sin ffmpeg filter_complex)
# ---------------------------------------------------------
def render_funnel_video(
    input_local: str,
    session_dir: str,
    clip_index: Dict[str, Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    Render final funnel video usando MoviePy.

    - Respeta el audio original (sin desync).
    - Corta por tiempos start/end de cada clip elegido.
    - Mantiene el orden de used_clip_ids.
    """
    logger.info(
        "🎞️ render_funnel_video (MoviePy) starting",
        extra={
            "input_local": input_local,
            "session_dir": session_dir,
            "used_clip_ids": used_clip_ids,
        },
    )

    if not used_clip_ids:
        raise ValueError("render_funnel_video: no used_clip_ids provided")

    ensure_dir(session_dir)
    final_path = os.path.join(session_dir, "final.mp4")

    # 1) Abrimos el video maestro
    base_clip = VideoFileClip(input_local)

    # FPS: si no viene definido, usamos 30
    fps = getattr(base_clip, "fps", None) or 30

    # 2) Crear subclips
    subclips = []
    for cid in used_clip_ids:
        c = clip_index.get(cid)
        if not c:
            logger.warning("render_funnel_video: clip_id not in index", extra={"cid": cid})
            continue

        start = float(c.get("start", 0.0))
        end = float(c.get("end", start + 0.1))

        if end <= start:
            end = start + 0.1

        logger.info(
            "✂️ Subclip",
            extra={"cid": cid, "start": start, "end": end},
        )

        sub = base_clip.subclip(start, end)
        subclips.append(sub)

    if not subclips:
        base_clip.close()
        raise ValueError("render_funnel_video: no valid subclips to concatenate")

    final_clip = concatenate_videoclips(subclips, method="compose")

    logger.info(
        "💾 Writing final video with MoviePy",
        extra={"final_path": final_path, "fps": fps},
    )

    final_clip.write_videofile(
        final_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        temp_audiofile=os.path.join(session_dir, "temp-audio.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None,
    )

    final_clip.close()
    base_clip.close()

    logger.info("✅ render_funnel_video done", extra={"final_path": final_path})
    return final_path


# ---------------------------------------------------------
# 🔧 run_pipeline (entrypoint desde tasks.job_render)
# ---------------------------------------------------------
def run_pipeline(session_id: str, files: List[str]) -> Dict[str, Any]:
    """
    Entry principal del worker.

    tasks.job_render lo llama así:

        result = pipeline.run_pipeline(
            session_id=session_id,
            files=files,
        )

    Donde `files` es una lista de URLs (usamos la primera).
    """
    if not files:
        raise ValueError("run_pipeline: files list is empty")

    video_url = files[0]

    session_dir = os.path.join(TMP_ROOT, session_id)
    ensure_dir(session_dir)

    input_local = os.path.join(session_dir, "input.mp4")
    download_to_local(video_url, input_local)

    # 1) ASR
    segments, duration_sec, language = run_asr(input_local)

    # 2) Clips + LLM
    clips = build_clips_from_asr(segments, language)

    # 3) Composer (Free-Flow, sin límites de slots)
    comp = compose_funnel(clips)
    composer_meta = comp["composer"]
    composer_human = comp["composer_human"]
    slots_map = comp["slots"]
    used_clip_ids = comp["used_clip_ids"]

    # índice por id para el render
    clip_index: Dict[str, Dict[str, Any]] = {c["id"]: c for c in clips}

    # 4) Render final
    final_path = render_funnel_video(input_local, session_dir, clip_index, used_clip_ids)

    # 5) Upload opcional a S3
    output_url = upload_to_s3(final_path, session_id)

    result = {
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": {
            k: v for k, v in slots_map.items()
        },
        "composer": composer_meta,
        "composer_human": composer_human,
        "output_video_local": final_path,
        "output_video_url": output_url,
        "asr": True,
        "semantic": True,
        "vision": True,  # aún stub, pero mantenemos flag como en tus logs
    }

    return result
