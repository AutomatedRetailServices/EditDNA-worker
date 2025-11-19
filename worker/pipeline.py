import os
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

import requests
from faster_whisper import WhisperModel
import boto3
import openai

logger = logging.getLogger("editdna.pipeline")
logger.setLevel(logging.INFO)

# ==== CONFIG ====

TMP_DIR = os.environ.get("TMP_DIR", "/tmp/TMP/editdna")

# Whisper config (acepta WHISPER_MODEL_NAME o WHISPER_MODEL del ENV)
WHISPER_MODEL_NAME = (
    os.environ.get("WHISPER_MODEL_NAME")
    or os.environ.get("WHISPER_MODEL")
    or "medium"
)

# Device puede venir como WHISPER_DEVICE o ASR_DEVICE
WHISPER_DEVICE = (
    os.environ.get("WHISPER_DEVICE")
    or os.environ.get("ASR_DEVICE")
    or "auto"
)  # auto / cuda / cpu

COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.55"))
COMPOSER_MAX_PER_SLOT = int(os.environ.get("COMPOSER_MAX_PER_SLOT", "7"))
MICRO_SENTENCE_MAX_SECONDS = float(os.environ.get("MICRO_SENTENCE_MAX_SECONDS", "8.0"))

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

# LLM + Vision
EDITDNA_LLM_MODEL = os.environ.get("EDITDNA_LLM_MODEL", "gpt-5.1")
EDITDNA_USE_LLM = int(os.environ.get("EDITDNA_USE_LLM", "0"))
VISION_ENABLED = int(os.environ.get("VISION_ENABLED", "0"))

openai.api_key = os.environ.get("OPENAI_API_KEY")

# ==== HELPERS BÁSICOS ====


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def ensure_session_dir(session_id: str) -> str:
    base = TMP_DIR
    session_dir = os.path.join(base, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def download_to_local(url: str, dst_path: str) -> None:
    logger.info(f"Descargando input: {url} -> {dst_path}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def ffprobe_json(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.warning(f"ffprobe fallo con code={proc.returncode}: {proc.stderr}")
        return {}
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {}


def probe_duration(path: str) -> float:
    info = ffprobe_json(path)
    if not info:
        return 0.0
    fmt = info.get("format", {})
    if "duration" in fmt:
        return safe_float(fmt["duration"])
    return 0.0


def has_audio_stream(path: str) -> bool:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            return True
    return False


def upload_to_s3(local_path: str, bucket: str, key: str) -> Optional[str]:
    try:
        s3 = boto3.client("s3")
        logger.info(f"Subiendo a S3 s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except Exception as e:
        logger.exception(f"Error subiendo a S3: {e}")
        return None


# ==== WHISPER ASR ====


_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    device = "cpu"
    compute_type = "int8"
    if WHISPER_DEVICE in ("cuda", "gpu", "auto"):
        try:
            device = "cuda"
            compute_type = "float16"
        except Exception:
            device = "cpu"
            compute_type = "int8"

    logger.info(
        f"Cargando Whisper model={WHISPER_MODEL_NAME} device={device} compute_type={compute_type}"
    )
    _WHISPER_MODEL = WhisperModel(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type=compute_type,
    )
    return _WHISPER_MODEL


def run_asr(input_local: str) -> List[Dict[str, Any]]:
    """
    Corre faster-whisper con timestamps por palabra.
    Devuelve segmentos con: start, end, text, words[{start,end,word}]
    """
    model = get_whisper_model()
    logger.info(f"Corriendo ASR sobre {input_local}")
    segments_iter, info = model.transcribe(
        input_local,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    out: List[Dict[str, Any]] = []
    idx = 0
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "start": safe_float(w.start),
                        "end": safe_float(w.end),
                        "word": w.word,
                    }
                )
        out.append(
            {
                "id": f"S{idx:04d}",
                "start": safe_float(seg.start),
                "end": safe_float(seg.end),
                "text": seg.text.strip(),
                "words": words,
            }
        )
        idx += 1

    logger.info(
        f"ASR produjo {len(out)} segmentos, duración ~{probe_duration(input_local):.2f}s"
    )
    return out


# ==== SENTENCE-BOUNDARY MICRO-CUTS ====


def make_base_clip(cid: str, start: float, end: float, text: str) -> Dict[str, Any]:
    # Valores por defecto para visión (placeholder 1.0 / 0.0)
    clip = {
        "id": cid,
        "slot": "STORY",  # se corregirá luego
        "start": start,
        "end": end,
        "score": 0.0,
        "semantic_score": 0.0,
        "visual_score": 1.0,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [cid],
        "text": text,
        "llm_reason": "",
        "visual_flags": {
            "scene_jump": False,
            "motion_jump": False,
        },
        "meta": {
            "slot": "STORY",
            "semantic_score": 0.0,
            "visual_score": 1.0,
            "score": 0.0,
            "chain_ids": [],
            "keep": True,
        },
    }
    return clip


def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convierte segmentos de Whisper en micro-oraciones:
    - Split por puntuación (. ? !) o duración > MICRO_SENTENCE_MAX_SECONDS
    - Mantiene timestamps precisos (start/end de primera/última palabra)
    Devuelve estructura "clips" con IDs tipo ASR0000_c0.
    """
    clips: List[Dict[str, Any]] = []
    clip_index = 0

    for seg_idx, seg in enumerate(asr_segments):
        words = seg.get("words") or []
        if not words:
            text = seg.get("text", "").strip()
            if not text:
                continue
            cid = f"ASR{seg_idx:04d}_c0"
            start = safe_float(seg.get("start", 0.0))
            end = safe_float(seg.get("end", start))
            if end <= start:
                continue
            clip = make_base_clip(
                cid=cid,
                start=start,
                end=end,
                text=text,
            )
            clips.append(clip)
            clip_index += 1
            continue

        buffer_words: List[Dict[str, Any]] = []
        sent_start: Optional[float] = None

        def flush_sentence():
            nonlocal clip_index, buffer_words, sent_start
            if not buffer_words:
                return
            s = buffer_words[0]
            e = buffer_words[-1]
            start_ts = safe_float(s.get("start", 0.0))
            end_ts = safe_float(e.get("end", start_ts))
            text = "".join([bw.get("word", "") for bw in buffer_words]).strip()
            if not text or end_ts <= start_ts:
                buffer_words = []
                sent_start = None
                return
            cid = f"ASR{seg_idx:04d}_c{clip_index}"
            clip = make_base_clip(
                cid=cid,
                start=start_ts,
                end=end_ts,
                text=text,
            )
            clips.append(clip)
            clip_index += 1
            buffer_words = []
            sent_start = None

        for w in words:
            w_start = safe_float(w.get("start", 0.0))
            w_end = safe_float(w.get("end", w_start))
            token = str(w.get("word", ""))

            if sent_start is None:
                sent_start = w_start
            buffer_words.append(w)

            duration = w_end - sent_start
            punct = token.strip().endswith((".", "?", "!"))
            if punct or duration >= MICRO_SENTENCE_MAX_SECONDS:
                flush_sentence()

        flush_sentence()

    return clips


# ==== CLASIFICACIÓN HEURÍSTICA (BACKUP) ====


FILLER_PATTERNS = [
    "is that good",
    "is that funny",
    "am i saying it right",
    "let me start again",
    "start again",
    "wait",
    "redo",
    "again?",
    "cut that",
    "thanks.",
    "thank you guys",
    "that one good",
    "why can't i remember",
]


def looks_like_filler(text: str) -> bool:
    t = text.lower().strip()
    for pat in FILLER_PATTERNS:
        if pat in t:
            return True
    if len(t.split()) <= 1 and t in {"and", "uh", "um", "hmm", "like"}:
        return True
    return False


def classify_slot(text: str) -> str:
    t = text.lower()

    # CTA
    if any(
        p in t
        for p in [
            "click the link",
            "tap the link",
            "shop now",
            "get yours",
            "grab one",
            "link below",
            "i left it for you",
        ]
    ):
        return "CTA"

    # Hook
    if "?" in t or t.startswith(("if ", "hey ", "listen", "stop scrolling", "ladies", "guys")):
        return "HOOK"

    # Problem
    if any(
        p in t
        for p in [
            "tired of",
            "sick of",
            "problem",
            "problems",
            "struggle",
            "does your",
            "is your",
            "keep giving you",
        ]
    ):
        return "PROBLEM"

    # Proof
    if any(
        p in t
        for p in [
            "i've been using",
            "i've tried",
            "i think they're really good",
            "i get so many compliments",
            "honestly",
            "for me",
        ]
    ):
        return "PROOF"

    # Benefits
    if any(
        p in t
        for p in [
            "so you can",
            "you can",
            "you'll",
            "you will",
            "feel",
            "helps you",
            "so freaking",
            "elevates any outfit",
            "feel fresh",
            "confident",
        ]
    ):
        return "BENEFITS"

    # Features
    if any(
        p in t
        for p in [
            "each gummy",
            "packed with",
            "ingredients",
            "it has",
            "it comes with",
            "it's actually",
            "it's a",
            "this bag",
            "these probiotics",
            "slippery m",
            "prebiotic",
            "probiotic",
            "flavored",
        ]
    ):
        return "FEATURES"

    # Story
    if any(
        p in t
        for p in ["because i found", "let me tell you", "when i", "the first time", "my experience"]
    ):
        return "STORY"

    return "STORY"


def tag_clips_heuristic(clips: List[Dict[str, Any]]) -> None:
    """
    Rellena:
      - slot
      - keep
      - semantic_score (0-1)
      - score
      - llm_reason (explicación breve)
      - meta.*
    Sirve como BACKUP si el LLM falla.
    """
    for c in clips:
        text = c.get("text", "")
        t = text.strip()
        slot = classify_slot(t)
        keep = not looks_like_filler(t)

        if not t:
            sem = 0.0
        elif keep:
            length = len(t.split())
            sem = min(0.95, 0.4 + 0.03 * length)
        else:
            sem = 0.0

        reason = ""
        if not keep:
            reason = "Marcado como filler / meta (redo, wait, duda, etc.)."
        else:
            if slot == "HOOK":
                reason = "Frase que llama la atención o hace una pregunta."
            elif slot == "PROBLEM":
                reason = "Describe un problema o situación dolorosa."
            elif slot == "BENEFITS":
                reason = "Enfatiza resultados o cambios positivos para la usuaria."
            elif slot == "FEATURES":
                reason = "Describe características específicas del producto."
            elif slot == "PROOF":
                reason = "Actúa como testimonio u opinión personal."
            elif slot == "CTA":
                reason = "Llama directamente a tomar acción (click / compra)."
            else:
                reason = "Conecta la historia o contexto del mensaje."

        c["slot"] = slot
        c["semantic_score"] = sem
        c["score"] = sem
        c["llm_reason"] = reason
        c["meta"]["slot"] = slot
        c["meta"]["semantic_score"] = sem
        c["meta"]["score"] = sem
        c["meta"]["keep"] = keep
def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())


def dedupe_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup simple: se queda con la primera aparición de cada texto normalizado.
    """
    seen = set()
    out = []
    for c in clips:
        if not c["meta"].get("keep", True):
            out.append(c)
            continue
        norm = normalize_text(c.get("text", ""))
        if not norm:
            out.append(c)
            continue
        if norm in seen:
            c["meta"]["keep"] = False
        else:
            seen.add(norm)
        out.append(c)
    return out


def build_slots_dict(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
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
        slot = c.get("slot", "STORY")
        if slot not in slots:
            slots[slot] = []
        slots[slot].append(c)
    return slots


def build_composer(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Free-Flow Composer:
      - Mantiene el orden cronológico.
      - Sólo conserva clips keep=True & semantic_score >= COMPOSER_MIN_SEMANTIC.
      - CTA (si existe) se mueve al final.
    """
    usable = [
        c
        for c in clips
        if c["meta"].get("keep", True)
        and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
    ]

    usable.sort(key=lambda c: safe_float(c.get("start", 0.0)))

    ctas = [c for c in usable if c.get("slot") == "CTA"]
    cta_clip = None
    if ctas:
        ctas.sort(key=lambda c: safe_float(c.get("semantic_score", 0.0)), reverse=True)
        cta_clip = ctas[0]

    timeline: List[Dict[str, Any]] = []
    used_ids: List[str] = []

    for c in usable:
        if cta_clip is not None and c["id"] == cta_clip["id"]:
            continue
        timeline.append(c)
        used_ids.append(c["id"])

    if cta_clip is not None:
        timeline.append(cta_clip)
        used_ids.append(cta_clip["id"])

    def cap_ids(slot_name: str) -> List[str]:
        ids = [c["id"] for c in timeline if c.get("slot") == slot_name]
        return ids[:COMPOSER_MAX_PER_SLOT]

    composer = {
        "hook_id": next((c["id"] for c in timeline if c.get("slot") == "HOOK"), None),
        "story_ids": cap_ids("STORY"),
        "problem_ids": cap_ids("PROBLEM"),
        "benefit_ids": cap_ids("BENEFITS"),
        "feature_ids": cap_ids("FEATURES"),
        "proof_ids": cap_ids("PROOF"),
        "cta_id": cta_clip["id"] if cta_clip else None,
        "used_clip_ids": used_ids,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }
    return composer


def pretty_print_composer(clips: List[Dict[str, Any]], composer: Dict[str, Any]) -> str:
    lookup = {c["id"]: c for c in clips}

    def line_for(cid: str) -> str:
        c = lookup.get(cid)
        if not c:
            return f"[{cid}] (no encontrado)"
        return f"[{cid}] score={c.get('score', 0.0):.2f} → \"{c.get('text', '').strip()}\""

    parts = ["===== EDITDNA FUNNEL COMPOSER ====="]

    slots_order = [
        ("HOOK", [composer.get("hook_id")] if composer.get("hook_id") else []),
        ("STORY", composer.get("story_ids", [])),
        ("PROBLEM", composer.get("problem_ids", [])),
        ("BENEFITS", composer.get("benefit_ids", [])),
        ("FEATURES", composer.get("feature_ids", [])),
        ("PROOF", composer.get("proof_ids", [])),
        ("CTA", [composer.get("cta_id")] if composer.get("cta_id") else []),
    ]

    for slot_name, ids in slots_order:
        parts.append(f"{slot_name}:")
        if not ids:
            parts.append("  (none)")
        else:
            for cid in ids:
                parts.append(f"  {line_for(cid)}")

    parts.append("\nFINAL ORDER TIMELINE:")
    for i, cid in enumerate(composer.get("used_clip_ids", []), start=1):
        c = lookup.get(cid)
        if not c:
            parts.append(f"{i}) {cid} (no encontrado)")
        else:
            parts.append(f"{i}) {cid} → \"{c.get('text', '').strip()}\"")

    parts.append("\n=====================================")
    return "\n".join(parts)


# ==== LLM SEMANTIC BRAIN ====


def run_llm_semantic_pass(clips: List[Dict[str, Any]]) -> bool:
    """
    Pasa todos los clips por GPT (EDITDNA_LLM_MODEL) para:
      - decidir slot (HOOK / PROBLEM / FEATURES / etc.)
      - decidir keep True/False (quita fillers, errores, redos)
      - ajustar semantic_score 0-1
      - explicar en llm_reason

    Si algo falla, devuelve False y se mantienen solo las heurísticas.
    """
    if not EDITDNA_USE_LLM:
        return False
    if not openai.api_key:
        logger.warning("EDITDNA_USE_LLM=1 pero falta OPENAI_API_KEY")
        return False

    items = [
        {"id": c["id"], "text": c.get("text", "").strip()}
        for c in clips
        if c.get("text", "").strip()
    ]
    if not items:
        return False

    system_msg = (
        "You are EditDNA Semantic Brain, a funnel-aware classifier for TikTok Shop ads. "
        "Your job is to label each spoken clip with:\n"
        "- slot: one of [HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA]\n"
        "- keep: true if the line should appear in the final ad; false if it is a filler, mistake, redo, doubt, or off-topic.\n"
        "- semantic_score: 0.0 to 1.0 representing how strong and useful the line is for conversion.\n"
        "- reason: a short explanation in Spanish of why you chose that slot/decision.\n\n"
        "You understand slang, Spanglish, and casual UGC talking head style. "
        "You also understand funnel logic: HOOK → PROBLEM → FEATURES → BENEFITS → PROOF → CTA."
    )

    user_payload = {
        "clips": items,
        "required_slots": ["HOOK", "BENEFITS", "FEATURES", "CTA"],
    }

    try:
        resp = openai.ChatCompletion.create(
            model=EDITDNA_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Clasifica cada clip y responde SOLO con un JSON de la forma:\n"
                        "[{\"id\":\"ASR0000_c0\",\"slot\":\"HOOK\",\"keep\":true,"
                        "\"semantic_score\":0.85,\"reason\":\"...\"}, ...]\n\n"
                        f"Datos:\n{json.dumps(user_payload, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
    except Exception as e:
        logger.exception(f"run_llm_semantic_pass fallo: {e}")
        return False

    if not isinstance(data, list):
        logger.warning("run_llm_semantic_pass: respuesta LLM no es lista")
        return False

    mapping = {}
    for item in data:
        cid = item.get("id")
        if not cid:
            continue
        mapping[cid] = item

    for c in clips:
        upd = mapping.get(c["id"])
        if not upd:
            continue

        slot = upd.get("slot") or c.get("slot", "STORY")
        keep = bool(upd.get("keep", True))
        sem = safe_float(upd.get("semantic_score", c.get("semantic_score", 0.0)))
        sem = max(0.0, min(1.0, sem))
        reason = upd.get("reason") or c.get("llm_reason", "")

        c["slot"] = slot
        c["semantic_score"] = sem
        c["score"] = sem
        c["llm_reason"] = reason
        c["meta"]["slot"] = slot
        c["meta"]["semantic_score"] = sem
        c["meta"]["score"] = sem
        c["meta"]["keep"] = keep

    logger.info("run_llm_semantic_pass: LLM aplicado correctamente sobre clips.")
    return True


# ==== VISUAL PASS (placeholder, para activar vision=True) ====


def run_visual_pass(input_local: str, clips: List[Dict[str, Any]]) -> None:
    """
    Visual pass placeholder.

    Ahora mismo NO usamos un modelo visual pesado. Solo dejamos el hook listo
    y marcamos que la fase visual se ejecutó. Más adelante aquí se enchufa
    detección de cara, gaze, scene cuts, etc. usando GPU.
    """
    logger.info("run_visual_pass: placeholder – sin modelo visual todavía.")
    # En el futuro podremos ajustar:
    #   - c['visual_score']
    #   - c['face_q']
    #   - c['scene_q']
    #   - c['visual_flags']
    return


# ==== FFMPEG RENDER (CONCAT VIDEO + AUDIO SEPARADOS) ====


def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    Render final:
      - Respeta 'used_clip_ids' (orden ya decidido por composer).
      - Corta [0:v] y [0:a] en paralelo con trim/atrim + setpts/asetpts.
      - Usa concat separado:
          * concat video: v=1:a=0 → [vout]
          * concat audio: v=0:a=1 → [aout]
      - Si algo raro pasa con audio (conteo desincronizado), desactiva audio
        en vez de lanzar el error 'media type mismatch'.
    """
    if not used_clip_ids:
        raise RuntimeError("render_funnel_video: no used_clip_ids provided")

    out_path = os.path.join(session_dir, "final.mp4")

    has_audio = has_audio_stream(input_local)
    logger.info(f"render_funnel_video: has_audio={has_audio}")

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
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{v_label}]"
        )
        v_labels.append(f"[{v_label}]")

        if has_audio:
            a_label = f"a{idx}"
            filter_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{a_label}]"
            )
            a_labels.append(f"[{a_label}]")

        idx += 1

    n = len(v_labels)
    if n == 0:
        raise RuntimeError("render_funnel_video: no valid clips after trimming")

    if has_audio and len(a_labels) != n:
        logger.warning(
            f"render_funnel_video: has_audio=True pero len(a_labels)={len(a_labels)} != len(v_labels)={n}, "
            f"desactivando audio para evitar media type mismatch."
        )
        has_audio = False

    filter_complex_parts: List[str] = list(filter_parts)

    filter_complex_parts.append(f"{''.join(v_labels)}concat=n={n}:v=1:a=0[vout]")

    if has_audio:
        filter_complex_parts.append(
            f"{''.join(a_labels)}concat=n={n}:v=0:a=1[aout]"
        )

    filter_complex = "; ".join(filter_complex_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_local,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
    ]

    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac"]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v",
        "libx264",
        "-movflags",
        "+faststart",
        out_path,
    ]

    logger.info("Running ffmpeg to render (separate concat for v/a)")
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
# ==== ENTRYPOINT PRINCIPAL ====


def run_pipeline(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Firma compatible con tasks.py, tanto si llama con:
        run_pipeline(session_id=session_id, files=files)
    como si llama con:
        run_pipeline(session_id=session_id, file_urls=files)
    """
    logger.info(f"run_pipeline session_id={session_id} files={files} file_urls={file_urls}")

    # Normalizamos: preferimos 'files', pero si viene 'file_urls', lo usamos
    effective_files: Optional[List[str]] = None
    if files and isinstance(files, list):
        effective_files = files
    elif file_urls and isinstance(file_urls, list):
        effective_files = file_urls

    if not effective_files:
        raise ValueError(
            "run_pipeline: se requiere 'files' o 'file_urls' como lista con al menos 1 URL"
        )

    session_dir = ensure_session_dir(session_id)
    input_local = os.path.join(session_dir, "input.mp4")

    # Tomamos sólo el primer archivo por ahora
    download_to_local(effective_files[0], input_local)

    duration = probe_duration(input_local)

    # 1) ASR
    asr_segments = run_asr(input_local)

    # 2) Micro-cuts por oración
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 3) Heurísticas (backup base)
    tag_clips_heuristic(clips)

    # 4) LLM semantic brain (si está activado)
    llm_used = False
    if EDITDNA_USE_LLM:
        llm_used = run_llm_semantic_pass(clips)

    # 5) Dedupe por texto
    clips = dedupe_clips(clips)

    # 6) Slots agrupados (para el JSON final)
    slots = build_slots_dict(clips)

    # 7) Composer free-flow
    composer = build_composer(clips)
    used_clip_ids = composer.get("used_clip_ids", [])

    # 8) Visual pass (si está activado)
    vision_used = False
    if VISION_ENABLED:
        try:
            run_visual_pass(input_local, clips)
            vision_used = True
        except Exception as e:
            logger.exception(f"run_visual_pass fallo: {e}")
            vision_used = False

    # 9) Render video recortando sólo used_clip_ids
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)

    # 10) S3 (opcional)
    output_url = None
    if S3_BUCKET:
        key = f"{S3_PREFIX}/{session_id}-final.mp4"
        output_url = upload_to_s3(final_path, S3_BUCKET, key)

    result = {
        "ok": True,
        "session_id": session_id,
        "input_local": input_local,
        "duration_sec": duration,
        "clips": clips,
        "slots": slots,
        "composer": composer,
        "composer_human": pretty_print_composer(clips, composer),
        "output_video_local": final_path,
        "output_video_url": output_url,
        "asr": True,
        "semantic": True,
        "vision": vision_used,
        "llm_used": bool(llm_used),
    }
    return result
