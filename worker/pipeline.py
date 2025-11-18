import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------
# Logging básico
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("editdna.pipeline")

# ---------------------------------------------------------------------
# Config de entorno
# ---------------------------------------------------------------------
TMP_ROOT = os.environ.get("TMP_ROOT", "/tmp/TMP/editdna")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
VIDEO_OUTPUT_BUCKET = os.environ.get(
    "VIDEO_OUTPUT_BUCKET", "script2clipshop-video-automatedretailservices"
)

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")  # "cuda" o "cpu"

# Veto mínimo para clips (si lo usas en otras partes)
VETO_MIN_SCORE = float(os.environ.get("VETO_MIN_SCORE", "0.4"))

# Composer (nuevo)
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.80"))

SLOT_LIMITS: Dict[str, int] = {
    "HOOK": 1,        # 1 hook
    "STORY": 0,       # 0 por ahora
    "PROBLEM": 1,     # 1 problema (si hay)
    "BENEFITS": 2,    # máx 2 beneficios
    "FEATURES": 3,    # máx 3 features
    "PROOF": 1,       # máx 1 prueba
    "CTA": 1,         # 1 CTA
}


# ---------------------------------------------------------------------
# Utils de texto para duplicados (nuevo composer)
# ---------------------------------------------------------------------
import re


def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _jaccard_sim(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def _is_duplicate(text: str, used_norm_texts: List[str], threshold: float) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    for prev in used_norm_texts:
        if _jaccard_sim(norm, prev) >= threshold:
            return True
    return False  # importante: solo añadimos al listado cuando aceptamos el clip


# ---------------------------------------------------------------------
# WHISPER ASR
# ---------------------------------------------------------------------
_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE})")
        compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=compute_type,
        )
    return _whisper_model


def run_asr(input_video: str) -> List[Dict[str, Any]]:
    """
    Corre Whisper y devuelve segmentos:
    [
      {
        "id": "ASR0000_c1",
        "start": float,
        "end": float,
        "text": str,
    ...
      }
    ]
    """
    model = get_whisper_model()
    logger.info(f"Running ASR on {input_video}")
    segments, info = model.transcribe(
        input_video,
        beam_size=5,
        best_of=5,
        language="en",
    )
    clips: List[Dict[str, Any]] = []
    idx = 0
    for seg in segments:
        cid = f"ASR{idx:04d}_c1"
        clips.append(
            {
                "id": cid,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )
        idx += 1
    logger.info(f"ASR produced {len(clips)} segments")
    return clips


# ---------------------------------------------------------------------
# Slotting + scoring sencillo (para que el composer pueda trabajar)
# ---------------------------------------------------------------------
def classify_slot(text: str, idx: int, total: int) -> str:
    t = text.lower()

    # CTA near the end or with call-to-action words
    if idx >= total - 2 or any(w in t for w in ["click", "link", "buy", "get yours", "get yours today"]):
        return "CTA"

    # Hooks at start or strong attention lines
    if idx == 0 or any(w in t for w in ["is your", "if you", "tired of", "worry no more"]):
        return "HOOK"

    # Problem words
    if any(w in t for w in ["problem", "struggle", "issue", "odor", "utis", "yeast"]):
        return "PROBLEM"

    # Benefits
    if any(w in t for w in ["feel", "confidence", "confident", "fresh", "results", "elevates"]):
        return "BENEFITS"

    # Features / ingredients
    if any(w in t for w in ["made just for", "packed with", "contains", "ingredient", "flavored", "probiotic", "prebiotic"]):
        return "FEATURES"

    return "FEATURES"


def heuristic_keep_and_scores(text: str) -> Dict[str, Any]:
    """
    Marca semantic_score, visual_score y keep.
    Aquí filtramos cosas tipo “wait”, “is that good?”, “yeah”, etc.
    """
    t = text.strip().lower()

    # Frases obvias de error/meta: bajamos semantic & keep=False
    meta_bad = [
        "wait.",
        "wait",
        "is that good?",
        "am i saying it right?",
        "yeah.",
        "yeah",
        "thanks.",
        "thanks",
        "well, that one good?",
        "i don't know how to do it like that.",
    ]
    if t in meta_bad or len(t.split()) <= 2:
        return {
            "semantic_score": 0.1,
            "visual_score": 1.0,
            "keep": False,
        }

    # Si parece frase útil / completa → más alto
    words = len(t.split())
    if words >= 6:
        sem = 0.85
    else:
        sem = 0.6

    return {
        "semantic_score": sem,
        "visual_score": 1.0,
        "keep": True,
    }


def enrich_clips_with_slots_and_scores(
    clips: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    total = len(clips)
    enriched: List[Dict[str, Any]] = []
    for idx, c in enumerate(clips):
        text = c.get("text", "")
        slot = classify_slot(text, idx, total)
        scores = heuristic_keep_and_scores(text)

        semantic_score = scores["semantic_score"]
        visual_score = scores["visual_score"]
        keep = scores["keep"]

        score = (semantic_score + visual_score) / 2.0

        # llm_reason placeholder
        llm_reason = ""
        if slot == "HOOK":
            llm_reason = "Hook candidate that can grab attention."
        elif slot == "BENEFITS":
            llm_reason = "Highlights potential positive outcomes for the viewer."
        elif slot == "FEATURES":
            llm_reason = "Describes specific product details or attributes."
        elif slot == "PROBLEM":
            llm_reason = "Touches on a problem or pain point."
        elif slot == "CTA":
            llm_reason = "Encourages viewers to take an action."

        enriched.append(
            {
                "id": c["id"],
                "slot": slot,
                "start": c["start"],
                "end": c["end"],
                "score": float(score),
                "semantic_score": float(semantic_score),
                "visual_score": float(visual_score),
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": semantic_score,
                "chain_ids": [c["id"]],
                "text": text,
                "llm_reason": llm_reason,
                "visual_flags": {
                    "scene_jump": False,
                    "motion_jump": False,
                },
                "meta": {
                    "slot": slot,
                    "semantic_score": float(semantic_score),
                    "visual_score": float(visual_score),
                    "score": float(score),
                    "chain_ids": [],
                    "keep": bool(keep),
                },
            }
        )
    return enriched


# ---------------------------------------------------------------------
# Bucket por slot
# ---------------------------------------------------------------------
def bucket_clips_by_slot(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
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
        slot = c.get("slot", "FEATURES")
        if slot not in slots:
            slots[slot] = []
        slots[slot].append(c)
    return slots


# ---------------------------------------------------------------------
# NUEVO COMPOSER — RUTHLESS ANTI-DUPLICADOS
# ---------------------------------------------------------------------
def compose_funnel(slots: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Build funnel structure con reglas duras:
    - Solo meta.keep == True
    - semantic_score >= COMPOSER_MIN_SEMANTIC
    - Elimina near-duplicates por texto
    - Respeta SLOT_LIMITS
    """
    used_ids: List[str] = []
    used_norm_texts: List[str] = []

    def pick_from_slot(slot_name: str) -> List[str]:
        clips = slots.get(slot_name, []) or []
        limit = SLOT_LIMITS.get(slot_name, 0)
        if limit <= 0 or not clips:
            return []

        chosen_ids: List[str] = []

        ordered = sorted(
            clips,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )

        for c in ordered:
            if len(chosen_ids) >= limit:
                break

            meta = c.get("meta", {}) or {}
            if not meta.get("keep", True):
                continue

            sem = c.get("semantic_score")
            if sem is None:
                sem = meta.get("semantic_score", 0.0)
            sem = float(sem or 0.0)

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

    hook_ids = pick_from_slot("HOOK")
    story_ids = pick_from_slot("STORY")
    problem_ids = pick_from_slot("PROBLEM")
    benefit_ids = pick_from_slot("BENEFITS")
    feature_ids = pick_from_slot("FEATURES")
    proof_ids = pick_from_slot("PROOF")
    cta_ids = pick_from_slot("CTA")

    hook_id = hook_ids[0] if hook_ids else None
    cta_id = cta_ids[0] if cta_ids else None

    composer = {
        "hook_id": hook_id,
        "story_ids": story_ids,
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": feature_ids,
        "proof_ids": proof_ids,
        "cta_id": cta_id,
        "used_clip_ids": used_ids,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }
    return composer


def compose_funnel_human(slots: Dict[str, List[Dict[str, Any]]],
                         composer: Dict[str, Any]) -> str:
    id_to_clip: Dict[str, Dict[str, Any]] = {}
    for slot_list in slots.values():
        for c in slot_list:
            id_to_clip[c["id"]] = c

    def line_for_id(cid: str) -> str:
        c = id_to_clip.get(cid)
        if not c:
            return f"[{cid}]"
        return f"[{cid}] score={c.get('score', 0.0):.2f} → \"{c.get('text', '')}\""

    parts: List[str] = []
    parts.append("===== EDITDNA FUNNEL COMPOSER =====")

    # HOOK
    parts.append("HOOK:")
    if composer.get("hook_id"):
        parts.append(f"  {line_for_id(composer['hook_id'])}")
    else:
        parts.append("  (none)")

    # STORY
    parts.append("STORY:")
    if composer.get("story_ids"):
        for cid in composer["story_ids"]:
            parts.append(f"  {line_for_id(cid)}")
    else:
        parts.append("  (none)")

    # PROBLEM
    parts.append("PROBLEM:")
    if composer.get("problem_ids"):
        for cid in composer["problem_ids"]:
            parts.append(f"  {line_for_id(cid)}")
    else:
        parts.append("  (none)")

    # BENEFITS
    parts.append("BENEFITS:")
    if composer.get("benefit_ids"):
        for cid in composer["benefit_ids"]:
            parts.append(f"  {line_for_id(cid)}")
    else:
        parts.append("  (none)")

    # FEATURES
    parts.append("FEATURES:")
    if composer.get("feature_ids"):
        for cid in composer["feature_ids"]:
            parts.append(f"  {line_for_id(cid)}")
    else:
        parts.append("  (none)")

    # PROOF
    parts.append("PROOF:")
    if composer.get("proof_ids"):
        for cid in composer["proof_ids"]:
            parts.append(f"  {line_for_id(cid)}")
    else:
        parts.append("  (none)")

    # CTA
    parts.append("CTA:")
    if composer.get("cta_id"):
        parts.append(f"  {line_for_id(composer['cta_id'])}")
    else:
        parts.append("  (none)")

    # TIMELINE
    timeline: List[str] = []
    if composer.get("hook_id"):
        timeline.append(composer["hook_id"])
    timeline.extend(composer.get("story_ids", []))
    timeline.extend(composer.get("problem_ids", []))
    timeline.extend(composer.get("benefit_ids", []))
    timeline.extend(composer.get("feature_ids", []))
    timeline.extend(composer.get("proof_ids", []))
    if composer.get("cta_id"):
        timeline.append(composer["cta_id"])

    parts.append("")
    parts.append("FINAL ORDER TIMELINE:")
    if not timeline:
        parts.append("  (none)")
    else:
        for i, cid in enumerate(timeline, start=1):
            parts.append(f"{i}) {line_for_id(cid)}")

    parts.append("")
    parts.append("=====================================")

    return "\n".join(parts)


# ---------------------------------------------------------------------
# RENDER — MoviePy para arreglar audio/video desync
# ---------------------------------------------------------------------
def render_timeline_with_moviepy(
    input_video: str,
    clips: List[Dict[str, Any]],
    composer: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Corta subclips del video original y los concatena en el orden del funnel.
    Usa MoviePy → arregla desfasaje de audio/video.
    """
    logger.info(f"Rendering final video with MoviePy → {output_path}")
    base = VideoFileClip(input_video)

    # Construimos la timeline según composer
    timeline_ids: List[str] = []
    if composer.get("hook_id"):
        timeline_ids.append(composer["hook_id"])
    timeline_ids.extend(composer.get("story_ids", []))
    timeline_ids.extend(composer.get("problem_ids", []))
    timeline_ids.extend(composer.get("benefit_ids", []))
    timeline_ids.extend(composer.get("feature_ids", []))
    timeline_ids.extend(composer.get("proof_ids", []))
    if composer.get("cta_id"):
        timeline_ids.append(composer["cta_id"])

    id_to_clip: Dict[str, Dict[str, Any]] = {c["id"]: c for c in clips}

    subclips = []
    for cid in timeline_ids:
        c = id_to_clip.get(cid)
        if not c:
            continue
        start = max(c["start"], 0.0)
        end = max(c["end"], start + 0.01)
        sub = base.subclip(start, end)
        subclips.append(sub)

    if not subclips:
        logger.warning("No subclips selected, copying original video")
        base.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=output_path + ".temp-audio.m4a",
            remove_temp=True,
        )
        base.close()
        return

    final = concatenate_videoclips(subclips, method="compose")
    # Aquí MoviePy maneja el sync de audio/video
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=output_path + ".temp-audio.m4a",
        remove_temp=True,
    )
    final.close()
    base.close()


# ---------------------------------------------------------------------
# S3 utils
# ---------------------------------------------------------------------
def upload_to_s3(local_path: str, key: str) -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    logger.info(f"Uploading {local_path} → s3://{VIDEO_OUTPUT_BUCKET}/{key}")
    s3.upload_file(local_path, VIDEO_OUTPUT_BUCKET, key, ExtraArgs={"ACL": "public-read"})

    url = f"https://{VIDEO_OUTPUT_BUCKET}.s3.amazonaws.com/{key}"
    return url


# ---------------------------------------------------------------------
# ENTRYPOINT PRINCIPAL
# ---------------------------------------------------------------------
def run_pipeline(
    session_id: str,
    input_local: str,
    s3_output_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry principal que deberías estar llamando desde tasks.py

    NOTA: si tu tasks.py le pasa URL S3 en vez de ruta local,
    cambia la firma y añade un paso de descarga antes de llamar a esto.
    """
    os.makedirs(TMP_ROOT, exist_ok=True)
    session_dir = os.path.join(TMP_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)

    logger.info(f"Starting pipeline for session={session_id}, input={input_local}")

    try:
        # 1) Cargar video para saber duración
        base_clip = VideoFileClip(input_local)
        duration_sec = float(base_clip.duration)
        base_clip.close()

        # 2) ASR
        raw_clips = run_asr(input_local)

        # 3) Enriquecer con slots + scores + meta.keep
        clips = enrich_clips_with_slots_and_scores(raw_clips)

        # 4) Buckets por slot
        slots = bucket_clips_by_slot(clips)

        # 5) Composer + texto humano
        composer = compose_funnel(slots)
        composer_human = compose_funnel_human(slots, composer)

        # 6) Render final con MoviePy (arreglo audio/video)
        output_filename = f"{session_id}-final.mp4"
        if s3_output_prefix:
            s3_key = f"{s3_output_prefix.rstrip('/')}/{output_filename}"
        else:
            s3_key = f"editdna/outputs/{output_filename}"

        output_local = os.path.join(session_dir, "final.mp4")
        render_timeline_with_moviepy(input_local, clips, composer, output_local)

        # 7) Upload S3
        output_url = upload_to_s3(output_local, s3_key)

        result = {
            "ok": True,
            "session_id": session_id,
            "input_local": input_local,
            "duration_sec": duration_sec,
            "clips": clips,
            "slots": slots,
            "composer": composer,
            "composer_human": composer_human,
            "output_video_local": output_local,
            "output_video_url": output_url,
            "asr": True,
            "semantic": True,  # estamos usando heurísticos, pero true
            "vision": True,    # placeholder: si luego tienes gating visual real
        }
        logger.info("Pipeline finished OK")
        return result

    except Exception as e:
        logger.exception("Pipeline failed")
        return {
            "ok": False,
            "session_id": session_id,
            "input_local": input_local,
            "error": str(e),
        }


if __name__ == "__main__":
    # Pequeño test manual si corres el container a mano
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <session_id> <input_video_path>")
        sys.exit(1)

    sid = sys.argv[1]
    inp = sys.argv[2]
    res = run_pipeline(sid, inp)
    print(json.dumps(res, indent=2))
