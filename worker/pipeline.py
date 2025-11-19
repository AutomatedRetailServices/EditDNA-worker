import os
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

import requests
from faster_whisper import WhisperModel
import boto3

logger = logging.getLogger("editdna.pipeline")
logger.setLevel(logging.INFO)

# ==== CONFIG ====
TMP_DIR = os.environ.get("TMP_DIR", "/tmp/TMP/editdna")

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "medium")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # auto / cuda / cpu

COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.55"))
COMPOSER_MAX_PER_SLOT = int(os.environ.get("COMPOSER_MAX_PER_SLOT", "7"))
MICRO_SENTENCE_MAX_SECONDS = float(os.environ.get("MICRO_SENTENCE_MAX_SECONDS", "8.0"))
COMPOSER_TARGET_DURATION = float(os.environ.get("COMPOSER_TARGET_DURATION", "35.0"))
COMPOSER_EARLY_WINDOW = float(os.environ.get("COMPOSER_EARLY_WINDOW", "8.0"))

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")


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

    logger.info(f"ASR produjo {len(out)} segmentos, duración ~{probe_duration(input_local):.2f}s")
    return out


# ==== SENTENCE-BOUNDARY MICRO-CUTS ====


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
            # Sin palabras → usamos el segmento completo (si no está vacío)
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
            # fin de oración por puntuación o duración
            punct = token.strip().endswith((".", "?", "!"))
            if punct or duration >= MICRO_SENTENCE_MAX_SECONDS:
                flush_sentence()

        # última oración del segmento
        flush_sentence()

    return clips


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


# ==== CLASIFICACIÓN HEURÍSTICA (SIN LLM) ====


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
    # frases súper cortas de una sola palabra tipo "and", "uh", "okay?"
    if len(t.split()) <= 1 and t in {"and", "uh", "um", "hmm", "like"}:
        return True
    return False


def classify_slot(text: str) -> str:
    t = text.lower()

    # CTA
    if any(p in t for p in ["click the link", "tap the link", "shop now", "get yours", "grab one", "link below", "i left it for you"]):
        return "CTA"

    # Hook
    if "?" in t or t.startswith(("if ", "hey ", "listen", "stop scrolling", "ladies", "guys")):
        return "HOOK"

    # Problem
    if any(p in t for p in ["tired of", "sick of", "problem", "problems", "struggle", "does your", "is your", "keep giving you"]):
        return "PROBLEM"

    # Proof
    if any(p in t for p in ["i've been using", "i've tried", "i think they're really good", "i get so many compliments", "honestly", "for me"]):
        return "PROOF"

    # Benefits
    if any(p in t for p in ["so you can", "you can", "you'll", "you will", "feel", "helps you", "so freaking", "elevates any outfit", "feel fresh", "confident"]):
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
    if any(p in t for p in ["because i found", "let me tell you", "when i", "the first time", "my experience"]):
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
    """
    for c in clips:
        text = c.get("text", "")
        t = text.strip()
        slot = classify_slot(t)
        keep = not looks_like_filler(t)

        # puntuación semántica simple:
        if not t:
            sem = 0.0
        elif keep:
            # más largo = más info → más score (hasta 0.95)
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


# ==== DEDUPE & SLOTS & COMPOSER ====


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
            # vacío → lo dejamos, pero normalmente su sem score será 0
            out.append(c)
            continue
        if norm in seen:
            # duplicado → descartamos
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
    Funnel Composer V3 FULL:
      - Mantiene bloque inicial HOOK/PROBLEM dentro de COMPOSER_EARLY_WINDOW.
      - Respeta orden cronológico.
      - Intenta construir HOOK → PROBLEM/STORY → FEATURES → BENEFITS → PROOF → CTA.
      - Limita duración a COMPOSER_TARGET_DURATION (CTA puede extender un poco si no entra).
    """
    # 1) Filtrar clips keep=True
    kept = [c for c in clips if c.get("meta", {}).get("keep", True)]
    if not kept:
        # fallback: usar todos
        kept = clips[:]

    # 2) Orden cronológico
    kept.sort(key=lambda c: safe_float(c.get("start", 0.0)))

    timeline: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    def add_clip(c: Dict[str, Any]) -> None:
        cid = c.get("id")
        if not cid or cid in used_ids:
            return
        timeline.append(c)
        used_ids.add(cid)

    # 3) Bloque inicial HOOK/PROBLEM dentro de early window (ej. primeras 8s)
    early_block = [
        c
        for c in kept
        if safe_float(c.get("start", 0.0)) < COMPOSER_EARLY_WINDOW
        and c.get("slot") in ("HOOK", "PROBLEM")
        and safe_float(c.get("score", 0.0)) >= 0.25
    ]
    for c in early_block:
        add_clip(c)

    # 4) Si no hay HOOK aún, elegimos mejor HOOK global
    if not any(c.get("slot") == "HOOK" for c in timeline):
        hook_candidates = [
            c
            for c in kept
            if c.get("slot") == "HOOK"
            and c.get("id") not in used_ids
            and safe_float(c.get("score", 0.0)) >= COMPOSER_MIN_SEMANTIC
        ]
        if hook_candidates:
            hook_candidates.sort(
                key=lambda c: (-safe_float(c.get("score", 0.0)), safe_float(c.get("start", 0.0)))
            )
            add_clip(hook_candidates[0])

    # 5) Helper para elegir mejores clips por slot
    def add_best_by_slot(slot_name: str, max_count: int) -> None:
        candidates = [
            c
            for c in kept
            if c.get("slot") == slot_name
            and c.get("id") not in used_ids
            and safe_float(c.get("score", 0.0)) >= COMPOSER_MIN_SEMANTIC
        ]
        if not candidates:
            return
        # ordenar por score (desc) luego por tiempo
        candidates.sort(
            key=lambda c: (-safe_float(c.get("score", 0.0)), safe_float(c.get("start", 0.0)))
        )
        chosen = candidates[:max_count]
        # Añadimos en orden cronológico
        for c in sorted(chosen, key=lambda c: safe_float(c.get("start", 0.0))):
            add_clip(c)

    # Orden lógico de slots (sin CTA todavía)
    add_best_by_slot("PROBLEM", max_count=1)
    add_best_by_slot("STORY", max_count=2)
    add_best_by_slot("FEATURES", max_count=3)
    add_best_by_slot("BENEFITS", max_count=2)
    add_best_by_slot("PROOF", max_count=1)

    # 6) CTA: elegimos la mejor, pero no la metemos todavía en timeline
    cta_candidates = [
        c
        for c in kept
        if c.get("slot") == "CTA"
        and c.get("id") not in used_ids
        and safe_float(c.get("score", 0.0)) >= 0.3
    ]
    cta_clip: Optional[Dict[str, Any]] = None
    if cta_candidates:
        cta_candidates.sort(
            key=lambda c: (-safe_float(c.get("score", 0.0)), safe_float(c.get("start", 0.0)))
        )
        cta_clip = cta_candidates[0]

    # 7) Timeline en orden cronológico (sin CTA)
    timeline.sort(key=lambda c: safe_float(c.get("start", 0.0)))

    # 8) Recortar por duración objetivo
    final_timeline: List[Dict[str, Any]] = []
    total_dur = 0.0

    for c in timeline:
        start = safe_float(c.get("start", 0.0))
        end = safe_float(c.get("end", 0.0))
        dur = max(0.0, end - start)
        if COMPOSER_TARGET_DURATION and total_dur + dur > COMPOSER_TARGET_DURATION and total_dur > 0:
            break
        final_timeline.append(c)
        total_dur += dur

    # 9) Asegurar CTA al final si existe
    if cta_clip is not None:
        cta_id = cta_clip.get("id")
        if cta_id and not any(c.get("id") == cta_id for c in final_timeline):
            final_timeline.append(cta_clip)

    if not final_timeline:
        # fallback: meter al menos algo
        final_timeline = kept[:1]

    used_clip_ids = [c.get("id") for c in final_timeline if c.get("id")]

    # 10) Construir composer dict tomando los IDs por slot del timeline final
    def collect_ids(slot_name: str) -> List[str]:
        ids = [c.get("id") for c in final_timeline if c.get("slot") == slot_name and c.get("id")]
        # respetar COMPOSER_MAX_PER_SLOT
        return ids[:COMPOSER_MAX_PER_SLOT]

    hook_id = next(
        (c.get("id") for c in final_timeline if c.get("slot") == "HOOK" and c.get("id")), None
    )
    story_ids = collect_ids("STORY")
    problem_ids = collect_ids("PROBLEM")
    benefit_ids = collect_ids("BENEFITS")
    feature_ids = collect_ids("FEATURES")
    proof_ids = collect_ids("PROOF")
    cta_ids = collect_ids("CTA")
    cta_id = cta_ids[0] if cta_ids else None

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
    return composer


def pretty_print_composer(clips: List[Dict[str, Any]], composer: Dict[str, Any]) -> str:
    lookup = {c["id"]: c for c in clips}

    def line_for(cid: str) -> str:
        c = lookup.get(cid)
        if not c:
            return f"[{cid}] (no encontrado)"
        return f"[{cid}] score={c.get('score', 0.0):.2f} → \"{c.get('text', '').strip()}\""

    parts = ["===== EDITDNA FUNNEL COMPOSER ====="]

    # Slots resumen
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


# ==== FFMPEG RENDER (CONCAT VIDEO + AUDIO SEPARADOS) ====


def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    Render final:
      - Respeta 'used_clip_ids' (orden ya decidido).
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

        # VIDEO
        v_label = f"v{idx}"
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{v_label}]"
        )
        v_labels.append(f"[{v_label}]")

        # AUDIO (si existe)
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

    # Si hay audio pero algo raro en conteos → lo apagamos para no romper concat
    if has_audio and len(a_labels) != n:
        logger.warning(
            f"render_funnel_video: has_audio=True pero len(a_labels)={len(a_labels)} != len(v_labels)={n}, "
            f"desactivando audio para evitar media type mismatch."
        )
        has_audio = False

    filter_complex_parts: List[str] = list(filter_parts)

    # Concat de VIDEO
    filter_complex_parts.append(
        f"{''.join(v_labels)}concat=n={n}:v=1:a=0[vout]"
    )

    # Concat de AUDIO separado (si sigue habilitado)
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
        raise ValueError("run_pipeline: se requiere 'files' o 'file_urls' como lista con al menos 1 URL")

    session_dir = ensure_session_dir(session_id)
    input_local = os.path.join(session_dir, "input.mp4")

    # Tomamos sólo el primer archivo por ahora
    download_to_local(effective_files[0], input_local)

    duration = probe_duration(input_local)

    # 1) ASR
    asr_segments = run_asr(input_local)

    # 2) Micro-cuts por oración
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 3) Heurísticas: slot + keep + semantic_score
    tag_clips_heuristic(clips)

    # 4) Dedupe simple por texto
    clips = dedupe_clips(clips)

    # 5) Slots agrupados (para el JSON final)
    slots = build_slots_dict(clips)

    # 6) Composer funnel V3
    composer = build_composer(clips)
    used_clip_ids = composer.get("used_clip_ids", [])

    # 7) Render video
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)

    # 8) S3 (opcional)
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
        "vision": False,
    }
    return result
