import os
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

import requests
import boto3
from faster_whisper import WhisperModel

# LLM
import openai

logger = logging.getLogger("editdna.pipeline")
logger.setLevel(logging.INFO)

# ========= CONFIG GENERAL =========

TMP_DIR = os.environ.get("TMP_DIR", "/tmp/TMP/editdna")

# ffmpeg / ffprobe (los tienes en el ENV)
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Whisper
WHISPER_MODEL_NAME = (
    os.environ.get("WHISPER_MODEL_NAME")
    or os.environ.get("WHISPER_MODEL")
    or "medium"
)
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # auto / cuda / cpu

# Composer / funnel
COMPOSER_MIN_SCORE = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_MAX_PER_SLOT = int(os.environ.get("COMPOSER_MAX_PER_SLOT", "7"))

# LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EDITDNA_LLM_MODEL = os.environ.get("EDITDNA_LLM_MODEL", "gpt-5.1")
EDITDNA_USE_LLM = os.environ.get("EDITDNA_USE_LLM", "1") == "1"

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Visión
VISION_ENABLED = os.environ.get("VISION_ENABLED", "0") == "1"
W_VISION = float(os.environ.get("W_VISION", "0.7"))

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

# ========= HELPERS BÁSICOS =========


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


# ========= WHISPER ASR =========

_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    # Intentamos usar GPU si se pidió y está disponible
    device = "cpu"
    compute_type = "int8"

    if WHISPER_DEVICE in ("cuda", "gpu", "auto"):
        try:
            # Faster-Whisper usa CTranslate2, no PyTorch,
            # pero si la build soporta CUDA, esto funciona.
            device = "cuda"
            compute_type = "float16"
        except Exception:
            logger.warning("No se pudo usar CUDA para Whisper, cayendo a CPU.")
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
# ========= SENTENCE-BOUNDARY MICRO-CUTS =========


def make_base_clip(cid: str, start: float, end: float, text: str) -> Dict[str, Any]:
    # Valores por defecto para visión (se rellenan luego)
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
    - Split por puntuación (. ? !) o duración > 8s (hard-coded).
    - Mantiene timestamps precisos.
    """
    MICRO_SENTENCE_MAX_SECONDS = float(
        os.environ.get("MICRO_SENTENCE_MAX_SECONDS", "8.0")
    )

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
            clips.append(make_base_clip(cid, start, end, text))
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
            clips.append(make_base_clip(cid, start_ts, end_ts, text))
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


# ========= HEURÍSTICA BASE (FILLERS, fallback) =========

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


def classify_slot_heuristic(text: str) -> str:
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
    if any(p in t for p in ["i've been using", "i've tried", "i think they're really good",
                            "i get so many compliments", "honestly", "for me"]):
        return "PROOF"

    # Benefits
    if any(p in t for p in ["so you can", "you can", "you'll", "you will", "feel",
                            "helps you", "so freaking", "elevates any outfit",
                            "feel fresh", "confident"]):
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


def apply_heuristic_tags(clips: List[Dict[str, Any]]) -> None:
    for c in clips:
        text = c.get("text", "").strip()
        slot = classify_slot_heuristic(text)
        keep = not looks_like_filler(text)

        if not text:
            sem = 0.0
        elif keep:
            length = len(text.split())
            sem = min(0.95, 0.4 + 0.03 * length)
        else:
            sem = 0.0

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


# ========= LLM TAGGING (GPT-5.1) =========

def tag_clips_with_llm(clips: List[Dict[str, Any]]) -> bool:
    """
    Usa GPT-5.1 para clasificar clips en HOOK/STORY/PROBLEM/BENEFITS/FEATURES/PROOF/CTA
    y asignar puntuación semántica + 'keep'. Si falla, devolvemos False.
    """
    if not (EDITDNA_USE_LLM and OPENAI_API_KEY):
        logger.info("LLM desactivado o sin API key; usando solo heurística.")
        apply_heuristic_tags(clips)
        return False

    try:
        payload_clips = [
            {"id": c["id"], "text": c.get("text", "").strip()}
            for c in clips
            if c.get("text", "").strip()
        ]

        system_msg = (
            "You are a performance-marketing video editor for TikTok Shop. "
            "You receive micro-sentences from a talking-head UGC ad. "
            "For each clip, classify its role in the funnel and decide if it should be kept.\n\n"
            "Valid slots:\n"
            "- HOOK: attention-grabber, question, bold statement, pattern interrupt.\n"
            "- STORY: context, setup, narrative, explanation.\n"
            "- PROBLEM: pain, frustration, issue described.\n"
            "- BENEFITS: outcomes, transformations, how user feels.\n"
            "- FEATURES: product details, ingredients, specs, how it works.\n"
            "- PROOF: testimonial, social proof, 'I get so many compliments', etc.\n"
            "- CTA: direct call to action (click, buy, link below, etc.).\n\n"
            "Mark 'keep' = false for obvious bloopers, restarts, doubts, meta-talk.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            "  \"clips\": [\n"
            "    {\"id\": \"ASR0000_c0\", \"slot\": \"HOOK\", \"keep\": true, \"semantic_score\": 0.85, \"reason\": \"...\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "semantic_score must be between 0 and 1, higher means more important for the funnel."
        )

        user_msg = json.dumps({"clips": payload_clips}, ensure_ascii=False)

        logger.info("Llamando a OpenAI LLM para tagging de clips...")
        resp = openai.ChatCompletion.create(
            model=EDITDNA_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )

        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)

        by_id = {c["id"]: c for c in clips}
        for item in data.get("clips", []):
            cid = item.get("id")
            if cid not in by_id:
                continue
            c = by_id[cid]
            slot = str(item.get("slot", "STORY")).upper()
            if slot not in {"HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"}:
                slot = "STORY"
            keep = bool(item.get("keep", True))
            sem = float(item.get("semantic_score", 0.0))
            sem = max(0.0, min(1.0, sem))
            reason = str(item.get("reason", ""))

            # Si el texto es filler obvio, forzamos keep=False y score bajo
            if looks_like_filler(c.get("text", "")):
                keep = False
                sem = min(sem, 0.1)
                if not reason:
                    reason = "Filler detectado por heurística."

            c["slot"] = slot
            c["semantic_score"] = sem
            c["score"] = sem
            c["llm_reason"] = reason
            c["meta"]["slot"] = slot
            c["meta"]["semantic_score"] = sem
            c["meta"]["score"] = sem
            c["meta"]["keep"] = keep

        # Por si algún clip no vino etiquetado, aplicamos heurística a esos
        for c in clips:
            if "slot" not in c["meta"] or c["meta"].get("semantic_score", 0.0) == 0.0:
                # Solo si no tiene nada útil
                if not c.get("text", "").strip():
                    continue
                apply_heuristic_tags([c])

        return True

    except Exception as e:
        logger.exception(f"Fallo LLM tagging, usando solo heurística: {e}")
        apply_heuristic_tags(clips)
        return False
# ========= DEDUPE & SLOTS =========

def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())


def dedupe_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup simple: se queda con la primera aparición de cada texto normalizado
    (solo para clips keep=True).
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


# ========= VISIÓN GPU (CLIP) =========

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_CLIP_TEXT_EMB = {}  # cache de prompts de texto


def extract_frame(input_video: str, t_sec: float, out_path: str) -> bool:
    """
    Extrae un frame usando ffmpeg en el timestamp t_sec.
    """
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        input_video,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        out_path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        logger.warning(f"ffmpeg extract_frame fallo: {proc.stderr}")
        return False
    return True


def get_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    import torch
    import clip  # de openai/CLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cargando CLIP ViT-B/32 en device={device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device
    return model, preprocess, device


def get_text_embedding(prompt: str):
    """
    Devuelve embedding normalizado de prompt de texto en espacio CLIP.
    """
    import torch
    import clip

    global _CLIP_TEXT_EMB
    if prompt in _CLIP_TEXT_EMB:
        return _CLIP_TEXT_EMB[prompt]

    model, _, device = get_clip_model()
    with torch.no_grad():
        tokens = clip.tokenize([prompt]).to(device)
        txt_emb = model.encode_text(tokens)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    _CLIP_TEXT_EMB[prompt] = txt_emb
    return txt_emb


VISION_TEXT_PROMPTS = {
    "HOOK": "close-up talking head hook shot for a product ad, expressive face, strong eye contact, vertical video",
    "STORY": "medium shot of a person talking naturally on camera about a product story, vertical video",
    "PROBLEM": "talking head explaining a pain or problem, concerned expression, vertical video",
    "BENEFITS": "person smiling and happy, showing the result or benefit of using a product, vertical video",
    "FEATURES": "close-up of the product in hand or near the face, clear packaging and label, vertical video",
    "PROOF": "selfie style testimonial, person talking confidently about great results, vertical video",
    "CTA": "talking head or hand gesture pointing to link or button, inviting to click or buy, vertical video",
}


def run_visual_pass(input_video: str, session_dir: str, clips: List[Dict[str, Any]]) -> bool:
    """
    Aplica CLIP sobre 1 frame por clip (frame medio) y calcula visual_score
    como similitud con un prompt de texto según el slot.
    Luego mezcla semantic_score y visual_score en 'score' usando W_VISION.
    Devuelve True si se pudo usar visión, False si no.
    """
    if not VISION_ENABLED:
        logger.info("Visión desactivada por ENV; visual_score se queda en 1.0.")
        return False

    try:
        import torch
        from PIL import Image

        model, preprocess, device = get_clip_model()

        for c in clips:
            # clips que ya se marcaron como keep=False igual los procesamos rápido
            start = safe_float(c.get("start", 0.0))
            end = safe_float(c.get("end", start))
            if end <= start:
                continue

            mid_t = (start + end) / 2.0
            frame_path = os.path.join(session_dir, f"{c['id']}_frame.jpg")

            if not extract_frame(input_video, mid_t, frame_path):
                continue

            try:
                img = Image.open(frame_path).convert("RGB")
            except Exception:
                logger.warning(f"No se pudo abrir frame {frame_path}")
                continue

            img_in = preprocess(img).unsqueeze(0).to(device)

            slot = c.get("slot", "STORY")
            prompt = VISION_TEXT_PROMPTS.get(slot, VISION_TEXT_PROMPTS["STORY"])
            txt_emb = get_text_embedding(prompt)

            with torch.no_grad():
                img_emb = model.encode_image(img_in)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                sim = (img_emb @ txt_emb.T).squeeze().item()

            # CLIP da ~[-1,1]; lo mapeamos a [0,1]
            visual_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            c["visual_score"] = visual_score
            c["meta"]["visual_score"] = visual_score

            sem = safe_float(c.get("semantic_score", 0.0))
            combined = (1.0 - W_VISION) * sem + W_VISION * visual_score
            c["score"] = combined
            c["meta"]["score"] = combined

        return True

    except Exception as e:
        logger.exception(f"Fallo visual pass; seguimos sin visión: {e}")
        return False


# ========= COMPOSER =========

def build_composer(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Free-Flow Composer:
      - Mantiene el orden cronológico.
      - Sólo conserva clips keep=True & score >= COMPOSER_MIN_SCORE.
      - CTA (si existe) se mueve al final.
    """
    usable = [
        c
        for c in clips
        if c["meta"].get("keep", True)
        and safe_float(c.get("score", 0.0)) >= COMPOSER_MIN_SCORE
    ]

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
        "hook_id": next((c["id"] for c in timeline if c.get("slot") == "HOOK"), None),
        "story_ids": cap_ids("STORY"),
        "problem_ids": cap_ids("PROBLEM"),
        "benefit_ids": cap_ids("BENEFITS"),
        "feature_ids": cap_ids("FEATURES"),
        "proof_ids": cap_ids("PROOF"),
        "cta_id": cta_clip["id"] if cta_clip else None,
        "used_clip_ids": used_ids,
        "min_score": COMPOSER_MIN_SCORE,
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
# ========= FFMPEG RENDER =========


def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    Render final:
      - Respeta 'used_clip_ids' (orden ya decidido).
      - Corta [0:v] y [0:a] en paralelo.
      - Concat separado para video y audio.
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

    logger.info("Ejecutando ffmpeg para render final")
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        logger.error(
            "ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
            proc.stdout,
            proc.stderr,
        )
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    return out_path


# ========= ENTRYPOINT PRINCIPAL =========


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

    # 1) ASR
    asr_segments = run_asr(input_local)

    # 2) Micro-cortes por oración
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 3) Tagging con LLM (con heurística como backup)
    llm_used = tag_clips_with_llm(clips)

    # 4) Dedupe textual
    clips = dedupe_clips(clips)

    # 5) Visión (CLIP en GPU) si está activado
    vision_used = run_visual_pass(input_local, session_dir, clips)

    # 6) Slots agrupados
    slots = build_slots_dict(clips)

    # 7) Composer
    composer = build_composer(clips)
    used_clip_ids = composer.get("used_clip_ids", [])

    # 8) Render final
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)

    # 9) S3
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
