import os
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

import requests
from faster_whisper import WhisperModel
import boto3

import clip

logger = logging.getLogger("editdna.pipeline")
logger.setLevel(logging.INFO)

# =====================
# CONFIG GLOBAL
# =====================

TMP_DIR = os.environ.get("TMP_DIR", "/tmp/TMP/editdna")

# FFmpeg / ffprobe
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Whisper / ASR
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME") or os.environ.get(
    "WHISPER_MODEL", "medium"
)
WHISPER_DEVICE = os.environ.get(
    "WHISPER_DEVICE", os.environ.get("ASR_DEVICE", "auto")
)  # cuda / cpu / auto

ASR_ENABLED = os.environ.get("ASR_ENABLED", "1") == "1"

# Composer / scores
COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_MAX_PER_SLOT = int(os.environ.get("COMPOSER_MAX_PER_SLOT", "7"))
MICRO_SENTENCE_MAX_SECONDS = float(
    os.environ.get("MICRO_SENTENCE_MAX_SECONDS", "8.0")
)

EDITDNA_MIN_CLIP_SCORE = float(os.environ.get("EDITDNA_MIN_CLIP_SCORE", "0.7"))
EDITDNA_HOOK_MIN_SCORE = float(os.environ.get("EDITDNA_HOOK_MIN_SCORE", "0.7"))
EDITDNA_CTA_MIN_SCORE = float(os.environ.get("EDITDNA_CTA_MIN_SCORE", "0.6"))

# LLM
EDITDNA_USE_LLM = os.environ.get("EDITDNA_USE_LLM", "0") == "1"
EDITDNA_LLM_MODEL = os.environ.get("EDITDNA_LLM_MODEL", "gpt-5.1")

# Visión
VISION_ENABLED = os.environ.get("VISION_ENABLED", "0") == "1"
VISION_INTERVAL_SEC = float(os.environ.get("VISION_INTERVAL_SEC", "2.0"))
VISION_MAX_SAMPLES = int(os.environ.get("VISION_MAX_SAMPLES", "50"))
W_VISION = float(os.environ.get("W_VISION", "0.7"))  # peso visual vs semántico (0-1)

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")


# =====================
# HELPERS BÁSICOS
# =====================


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
        FFPROBE_BIN,
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


def make_base_clip(cid: str, start: float, end: float, text: str) -> Dict[str, Any]:
    clip = {
        "id": cid,
        "slot": "STORY",  # se corregirá luego
        "start": start,
        "end": end,
        "score": 0.0,
        "semantic_score": 0.0,
        "visual_score": 0.0,
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
            "visual_score": 0.0,
            "score": 0.0,
            "chain_ids": [],
            "keep": True,
        },
    }
    return clip


# =====================
# WHISPER ASR
# =====================

_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    device = "cpu"
    compute_type = "int8"
    if WHISPER_DEVICE in ("cuda", "gpu", "auto"):
        try:
            import torch  # noqa

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
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
    if not ASR_ENABLED:
        raise RuntimeError("ASR_ENABLED=0 pero run_asr fue llamado")

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


# =====================
# SENTENCE-BOUNDARY MICRO-CUTS
# =====================


def sentence_boundary_micro_cuts(
    asr_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
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


# =====================
# HEURÍSTICA BÁSICA
# =====================

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

    if "?" in t or t.startswith(("if ", "hey ", "listen", "stop scrolling", "ladies", "guys")):
        return "HOOK"

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

    if any(
        p in t
        for p in [
            "because i found",
            "let me tell you",
            "when i",
            "the first time",
            "my experience",
        ]
    ):
        return "STORY"

    return "STORY"


def tag_clips_heuristic(clips: List[Dict[str, Any]]) -> None:
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


# =====================
# LLM SEMÁNTICO (GPT-5.1)
# =====================


def llm_classify_clips(clips: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY no está configurado, saltando LLM.")
        return None

    client = OpenAI(api_key=api_key)

    payload_clips = [
        {
            "id": c["id"],
            "text": c.get("text", ""),
        }
        for c in clips
        if c.get("text", "").strip()
    ]

    if not payload_clips:
        return None

    system_msg = (
        "You are an expert TikTok ad editor for feminine health products. "
        "You receive short transcript segments from a spoken ad (with slang). "
        "For each segment, you must classify its funnel role and how strong it is."
    )

    user_instruction = {
        "task": "classify_clips",
        "instructions": {
            "slots": [
                "HOOK",
                "STORY",
                "PROBLEM",
                "BENEFITS",
                "FEATURES",
                "PROOF",
                "CTA",
            ],
            "output_schema": {
                "id": "string",
                "slot": "string",
                "keep": "boolean",
                "semantic_score": "float (0-1)",
                "reason": "short explanation in spanish",
            },
        },
        "clips": payload_clips,
    }

    try:
        completion = client.chat.completions.create(
            model=EDITDNA_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Devuélveme SOLO un JSON con este formato exacto:\n\n"
                        "{\n"
                        '  "clips": [\n'
                        '    {"id": "...", "slot": "...", "keep": true/false, "semantic_score": 0.xx, "reason": "..."}\n'
                        "  ]\n"
                        "}\n\n"
                        "No agregues texto fuera del JSON.\n\n"
                        f"Aquí están los clips:\n{json.dumps(user_instruction, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        result_map = {}
        for item in data.get("clips", []):
            cid = item.get("id")
            if not cid:
                continue
            result_map[cid] = {
                "slot": item.get("slot", "STORY"),
                "keep": bool(item.get("keep", True)),
                "semantic_score": safe_float(
                    item.get("semantic_score", 0.0)
                ),
                "reason": item.get("reason", ""),
            }
        return result_map
    except Exception as e:
        logger.exception(f"Error llamando al LLM: {e}")
        return None


def enrich_clips_semantic(clips: List[Dict[str, Any]]) -> bool:
    tag_clips_heuristic(clips)

    if not EDITDNA_USE_LLM:
        return False

    llm_result = llm_classify_clips(clips)
    if not llm_result:
        return False

    for c in clips:
        cid = c["id"]
        if cid not in llm_result:
            continue
        info = llm_result[cid]
        slot = info.get("slot", c.get("slot", "STORY"))
        keep = bool(info.get("keep", c["meta"].get("keep", True)))
        sem = safe_float(info.get("semantic_score", c.get("semantic_score", 0.0)))
        reason = info.get("reason", "")

        c["slot"] = slot
        c["semantic_score"] = sem
        c["score"] = sem
        c["llm_reason"] = reason or c.get("llm_reason", "")
        c["meta"]["slot"] = slot
        c["meta"]["semantic_score"] = sem
        c["meta"]["score"] = sem
        c["meta"]["keep"] = keep

    return True


# =====================
# VISIÓN: CLIP + CUDA
# =====================

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = "cpu"


def load_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    try:
        import torch
        import clip  # type: ignore
    except Exception as e:
        logger.warning(f"No se pudo importar torch/clip para visión: {e}")
        return None, None, "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device

    logger.info(f"Modelo CLIP cargado en dispositivo {device}")
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def grab_frame_at_timestamp(input_local: str, t: float, out_path: str) -> bool:
    cmd = [
        FFMPEG_BIN,
        "-ss",
        f"{t:.3f}",
        "-i",
        input_local,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.debug(f"ffmpeg frame grab fallo para t={t:.3f}: {proc.stderr}")
        return False
    return True


def run_visual_pass(
    input_local: str, session_dir: str, clips: List[Dict[str, Any]]
) -> bool:
    if not VISION_ENABLED:
        logger.info("VISION_ENABLED=0, saltando vision pass.")
        return False

    model, preprocess, device = load_clip_model()
    if model is None:
        logger.warning("CLIP no disponible, vision=false.")
        return False

    try:
        import torch
        from PIL import Image
    except Exception as e:
        logger.warning(f"No se pudo importar torch/PIL para vision: {e}")
        return False

    if not clips:
        return False

    step = max(1, len(clips) // max(1, VISION_MAX_SAMPLES))
    logger.info(
        f"Vision pass: num_clips={len(clips)}, step={step}, device={device}"
    )

    for idx, c in enumerate(clips):
        if idx % step != 0:
            continue

        text = c.get("text", "").strip()
        if not text:
            continue

        mid_t = (
            safe_float(c.get("start", 0.0)) + safe_float(c.get("end", 0.0))
        ) / 2.0
        frame_path = os.path.join(session_dir, f"frame_{c['id']}.jpg")

        if not grab_frame_at_timestamp(input_local, mid_t, frame_path):
            continue

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception:
            continue

        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize([text]).to(device)  # type: ignore

            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            similarity = (image_features @ text_features.T).item()
            visual_score = (similarity + 1.0) / 2.0

        c["visual_score"] = float(visual_score)
        c["meta"]["visual_score"] = float(visual_score)

    return True


def apply_min_score_rules(clips: List[Dict[str, Any]]) -> None:
    for c in clips:
        slot = c.get("slot", "STORY")
        score = safe_float(c.get("score", 0.0))

        threshold = EDITDNA_MIN_CLIP_SCORE
        if slot == "HOOK":
            threshold = max(threshold, EDITDNA_HOOK_MIN_SCORE)
        elif slot == "CTA":
            threshold = max(threshold, EDITDNA_CTA_MIN_SCORE)

        if score < threshold:
            c["meta"]["keep"] = False


def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())


def dedupe_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# =====================
# SLOTS + COMPOSER
# =====================


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
    usable = [
        c
        for c in clips
        if c["meta"].get("keep", True)
        and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
    ]

    for c in usable:
        sem = safe_float(c.get("semantic_score", 0.0))
        vis = safe_float(c.get("visual_score", 0.0))
        if vis > 0.0:
            combined = (1.0 - W_VISION) * sem + W_VISION * vis
        else:
            combined = sem
        c["score"] = combined
        c["meta"]["score"] = combined

    apply_min_score_rules(usable)

    usable = [c for c in usable if c["meta"].get("keep", True)]
    usable.sort(key=lambda c: safe_float(c.get("start", 0.0)))

    ctas = [c for c in usable if c.get("slot") == "CTA"]
    cta_clip = None
    if ctas:
        ctas.sort(key=lambda c: safe_float(c.get("score", 0.0)), reverse=True)
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
        "hook_id": next(
            (c["id"] for c in timeline if c.get("slot") == "HOOK"), None
        ),
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


def pretty_print_composer(
    clips: List[Dict[str, Any]], composer: Dict[str, Any]
) -> str:
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
            parts.append(
                f"{i}) {cid} → \"{c.get('text', '').strip()}\""
            )

    parts.append("\n=====================================")
    return "\n".join(parts)


# =====================
# FFMPEG RENDER
# =====================


def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
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
            "render_funnel_video: has_audio=True pero len(a_labels)"
            f"={len(a_labels)} != len(v_labels)={n}, "
            "desactivando audio para evitar media type mismatch."
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
        FFMPEG_BIN,
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
        logger.error(
            "ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr
        )
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    return out_path


# =====================
# ENTRYPOINT PRINCIPAL
# =====================


def run_pipeline(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.info(
        f"run_pipeline session_id={session_id} files={files} file_urls={file_urls}"
    )

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

    download_to_local(effective_files[0], input_local)

    duration = probe_duration(input_local)

    asr_segments = run_asr(input_local)
    clips = sentence_boundary_micro_cuts(asr_segments)
    llm_used = enrich_clips_semantic(clips)
    clips = dedupe_clips(clips)
    vision_used = run_visual_pass(input_local, session_dir, clips)
    slots = build_slots_dict(clips)
    composer = build_composer(clips)
    used_clip_ids = composer.get("used_clip_ids", [])
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)

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
        "llm_used": llm_used,
    }
    return result
