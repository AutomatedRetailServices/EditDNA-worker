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
# GLOBAL CONFIG
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
)
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

# VISION
VISION_ENABLED = os.environ.get("VISION_ENABLED", "0") == "1"
VISION_INTERVAL_SEC = float(os.environ.get("VISION_INTERVAL_SEC", "2.0"))
VISION_MAX_SAMPLES = int(os.environ.get("VISION_MAX_SAMPLES", "50"))
W_VISION = float(os.environ.get("W_VISION", "0.7"))

# BAD TAKES (visual face check) – opcional
BAD_TAKES_ENABLED = os.environ.get("BAD_TAKES_ENABLED", "0") == "1"

# BOUNDARY TRIM (head/tail dentro del clip, opcional)
BOUNDARY_REFINER_ENABLED = os.environ.get("BOUNDARY_REFINER_ENABLED", "0") == "1"
BOUNDARY_REFINER_MIN_DURATION_SEC = float(
    os.environ.get("BOUNDARY_REFINER_MIN_DURATION_SEC", "3.0")
)
BOUNDARY_REFINER_HEAD_STEP_SEC = float(
    os.environ.get("BOUNDARY_REFINER_HEAD_STEP_SEC", "1.5")
)
BOUNDARY_REFINER_TAIL_STEP_SEC = float(
    os.environ.get("BOUNDARY_REFINER_TAIL_STEP_SEC", "1.5")
)

# TAKE JUDGE (multi-take selection)
TAKE_JUDGE_ENABLED = os.environ.get("TAKE_JUDGE_ENABLED", "0") == "1"
TAKE_JUDGE_MODEL = os.environ.get("TAKE_JUDGE_MODEL", "gpt-4o-mini")
TAKE_JUDGE_MAX_GROUPS = int(os.environ.get("TAKE_JUDGE_MAX_GROUPS", "6"))
TAKE_JUDGE_MAX_TAKES = int(os.environ.get("TAKE_JUDGE_MAX_TAKES", "3"))
TAKE_JUDGE_FRAMES = int(os.environ.get("TAKE_JUDGE_FRAMES", "1"))

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

# Head/tail trims globales (post selección)
HEAD_TRIM_SEC = float(os.environ.get("HEAD_TRIM_SEC", "0.0"))
TAIL_TRIM_SEC = float(os.environ.get("TAIL_TRIM_SEC", "0.0"))


# =====================
# HELPERS
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
    logger.info(f"Downloading input: {url} -> {dst_path}")
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
        logger.warning(f"ffprobe failed with code={proc.returncode}: {proc.stderr}")
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
        logger.info(f"Uploading to S3 s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except Exception as e:
        logger.exception(f"Error uploading to S3: {e}")
        return None


# =====================
# CLIP OBJECT CREATION
# =====================

def make_base_clip(cid: str, start: float, end: float, text: str) -> Dict[str, Any]:
    clip_obj = {
        "id": cid,
        "slot": "STORY",  # provisional, luego se corrige
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
    return clip_obj


# =====================
# ASR
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
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass

    logger.info(
        f"Loading Whisper model={WHISPER_MODEL_NAME} device={device} compute_type={compute_type}"
    )
    _WHISPER_MODEL = WhisperModel(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type=compute_type,
    )
    return _WHISPER_MODEL


def run_asr(input_local: str) -> List[Dict[str, Any]]:
    """
    Run faster-whisper with word timestamps.
    """
    if not ASR_ENABLED:
        raise RuntimeError("ASR_ENABLED=0 but run_asr was called")

    model = get_whisper_model()
    logger.info(f"Running ASR over {input_local}")

    segments_iter, info = model.transcribe(
        input_local,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    out = []
    idx = 0
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": safe_float(w.start),
                    "end": safe_float(w.end),
                    "word": w.word,
                })

        out.append({
            "id": f"S{idx:04d}",
            "start": safe_float(seg.start),
            "end": safe_float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })
        idx += 1

    return out


# =====================
# MICRO CUTS
# =====================

def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Break Whisper segments into micro-sentences using punctuation & max length.
    """
    clips = []
    clip_index = 0

    for seg_idx, seg in enumerate(asr_segments):
        words = seg.get("words") or []

        # no word-level timestamps → treat as single chunk
        if not words:
            text = seg.get("text", "").strip()
            if not text:
                continue
            cid = f"ASR{seg_idx:04d}_c0"
            start = safe_float(seg.get("start", 0.0))
            end = safe_float(seg.get("end", start))
            if end <= start:
                continue

            clip_obj = make_base_clip(cid, start, end, text)
            clips.append(clip_obj)
            clip_index += 1
            continue

        # split by punctuation or max duration
        buffer_words = []
        sent_start = None

        def flush_sentence():
            nonlocal buffer_words, sent_start, clip_index
            if not buffer_words:
                return
            s = buffer_words[0]
            e = buffer_words[-1]
            start_ts = safe_float(s.get("start", 0.0))
            end_ts = safe_float(e.get("end", start_ts))
            text_local = "".join([bw.get("word", "") for bw in buffer_words]).strip()
            if not text_local or end_ts <= start_ts:
                buffer_words = []
                sent_start = None
                return

            cid_local = f"ASR{seg_idx:04d}_c{clip_index}"
            clip_obj_local = make_base_clip(
                cid_local,
                start_ts,
                end_ts,
                text_local,
            )
            clips.append(clip_obj_local)
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

        # final flush
        flush_sentence()

    return clips


# =====================
# MERGE INTELIGENTE DE FRASES INCOMPLETAS
# =====================

def looks_incomplete(text: str) -> bool:
    """
    Detecta frases que NO terminan una idea.
    """
    t = text.strip().lower()
    if len(t.split()) <= 2:
        return True

    bad_endings = ("for", "the", "a", "my", "your", "our", "this", "that")
    if t.endswith(bad_endings):
        return True

    if t.startswith(("and ", "so ", "but ", "or ")):
        return True

    if not t.endswith((".", "?", "!")):
        return True

    return False


def merge_incomplete_phrases(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    UNE frases incompletas con la siguiente si tiene sentido.
    """
    if not clips:
        return clips

    merged = []
    i = 0

    while i < len(clips):
        c = clips[i]
        text = c["text"].strip()

        if not looks_incomplete(text):
            merged.append(c)
            i += 1
            continue

        if i + 1 < len(clips):
            nxt = clips[i + 1]
            next_text = nxt["text"].strip()

            can_merge = (
                len(next_text.split()) >= 2 and
                not looks_incomplete(next_text) and
                text[0].isalpha()
            )

            if can_merge:
                new_text = (text + " " + next_text).strip()
                merged_clip = {
                    **c,
                    "text": new_text,
                    "end": nxt["end"],
                    "chain_ids": c["chain_ids"] + nxt["chain_ids"],
                }
                merged.append(merged_clip)
                i += 2
                continue

        i += 1

    return merged


# =====================
# HEURISTIC TAGGING
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

TAIL_DEPENDENT_ENDINGS = [
    "as well",
    "too",
    "either",
]

TAIL_DEPENDENT_STARTS = [
    "and",
    "so",
    "but",
    "because",
]


def looks_like_filler(text: str) -> bool:
    t = text.lower().strip()
    for pat in FILLER_PATTERNS:
        if pat in t:
            return True
    if len(t.split()) <= 1 and t in {"and", "uh", "um", "hmm", "like"}:
        return True
    return False


def looks_like_dependent_tail(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False

    words = t.split()
    if len(words) > 4:
        return False

    for suf in TAIL_DEPENDENT_ENDINGS:
        if t.endswith(suf):
            return True

    if words[0] in TAIL_DEPENDENT_STARTS:
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
        text = c.get("text", "") or ""
        t = text.strip()

        slot = classify_slot(t)
        is_tail = looks_like_dependent_tail(t)
        is_filler = looks_like_filler(t)

        keep = not is_filler and not is_tail

        if not t:
            sem = 0.0
        elif keep:
            length = len(t.split())
            sem = min(0.95, 0.4 + 0.03 * length)
        else:
            sem = 0.0

        if not keep:
            if is_tail:
                reason = "Dependent tail without full context (cola tipo 'available as well')."
            else:
                reason = "Marked as filler / meta (redo, wait, etc.)."
        else:
            if slot == "HOOK":
                reason = "Attention-grabbing phrase or question."
            elif slot == "PROBLEM":
                reason = "Describes a problem or painful situation."
            elif slot == "BENEFITS":
                reason = "Highlights positive outcomes for the user."
            elif slot == "FEATURES":
                reason = "Describes specific product features."
            elif slot == "PROOF":
                reason = "Acts as testimonial or personal opinion."
            elif slot == "CTA":
                reason = "Direct call to action (click / buy)."
            else:
                reason = "Connects story or context."

        c["slot"] = slot
        c["semantic_score"] = sem
        c["score"] = sem
        c["llm_reason"] = reason
        c["meta"]["slot"] = slot
        c["meta"]["semantic_score"] = sem
        c["meta"]["score"] = sem
        c["meta"]["keep"] = keep


# =====================
# SEMANTIC LLM (GPT-5.1)
# =====================

def llm_classify_clips(clips: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set, skipping LLM.")
        return None

    client = OpenAI(api_key=api_key)

    payload_clips = [
        {"id": c["id"], "text": c.get("text", "")}
        for c in clips
        if (c.get("text") or "").strip()
    ]

    if not payload_clips:
        return None

    system_msg = (
        "You are an expert TikTok ad editor. "
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

    user_text = (
        "Devuélveme SOLO un JSON con este formato exacto:\n\n"
        "{\n"
        '  \"clips\": [\n'
        '    {\"id\": \"...\", \"slot\": \"...\", \"keep\": true/false, '
        '\"semantic_score\": 0.xx, \"reason\": \"...\"}\n'
        "  ]\n"
        "}\n\n"
        "No agregues texto fuera del JSON.\n\n"
        f"Aquí están los clips:\n{json.dumps(user_instruction, ensure_ascii=False)}"
    )

    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """
        Parsing robusto para cuando el modelo devuelve ```json ...``` o texto extra.
        """
        if not raw:
            return None
        txt = raw.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 3:
                txt = parts[1]
            txt = txt.strip()
        start_i = txt.find("{")
        end_i = txt.rfind("}")
        if start_i == -1 or end_i == -1 or end_i <= start_i:
            return None
        txt = txt[start_i : end_i + 1]
        try:
            return json.loads(txt)
        except Exception:
            return None

    try:
        completion = client.chat.completions.create(
            model=EDITDNA_LLM_MODEL,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ],
            temperature=0.2,
        )
    except Exception as e:
        logger.exception(f"Error calling LLM: {e}")
        return None

    try:
        content = completion.choices[0].message.content or ""
    except Exception as e:
        logger.exception(f"Error reading LLM content: {e}")
        return None

    data = _safe_parse_json(content)
    if not data:
        logger.warning(f"LLM JSON parse error, raw={content[:120]!r}")
        return None

    result_map: Dict[str, Any] = {}
    for item in data.get("clips", []):
        cid = item.get("id")
        if not cid:
            continue
        result_map[cid] = {
            "slot": item.get("slot", "STORY"),
            "keep": bool(item.get("keep", True)),
            "semantic_score": safe_float(item.get("semantic_score", 0.0)),
            "reason": item.get("reason", ""),
        }
    return result_map


def enrich_clips_semantic(clips: List[Dict[str, Any]]) -> bool:
    """
    1) Heurística local (filler, slots, semantic_score)
    2) LLM opcional, con reglas de seguridad de embudo.
    """
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
        original_keep = c["meta"].get("keep", True)

        slot_from_llm = info.get("slot", c.get("slot", "STORY"))
        sem_from_llm = safe_float(info.get("semantic_score", c.get("semantic_score", 0.0)))
        reason = info.get("reason", "")

        c["slot"] = slot_from_llm
        c["semantic_score"] = sem_from_llm
        c["score"] = sem_from_llm
        c["llm_reason"] = reason or c.get("llm_reason", "")

        c["meta"]["slot"] = slot_from_llm
        c["meta"]["semantic_score"] = sem_from_llm
        c["meta"]["score"] = sem_from_llm

        llm_keep_raw = bool(info.get("keep", original_keep))

        text = (c.get("text") or "").strip()
        word_count = len(text.split())
        slot_upper = slot_from_llm
        strong_semantic = sem_from_llm >= COMPOSER_MIN_SEMANTIC
        core_slot = slot_upper in {"HOOK", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"}

        if not original_keep:
            final_keep = False
        else:
            if core_slot and strong_semantic and word_count >= 3:
                final_keep = True
            else:
                final_keep = original_keep and llm_keep_raw

        c["meta"]["keep"] = final_keep

    return True


# =====================
# VISION: CLIP + CUDA
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
        import clip as clip_lib  # type: ignore
    except Exception as e:
        logger.warning(f"Could not import torch/clip for vision: {e}")
        return None, None, "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_lib.load("ViT-B/32", device=device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device

    logger.info(f"CLIP model loaded on device {device}")
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
        logger.debug(f"ffmpeg frame grab failed for t={t:.3f}: {proc.stderr}")
        return False
    return True


def run_visual_pass(
    input_local: str, session_dir: str, clips: List[Dict[str, Any]]
) -> bool:
    if not VISION_ENABLED:
        logger.info("VISION_ENABLED=0, skipping vision pass.")
        return False

    model, preprocess, device = load_clip_model()
    if model is None:
        logger.warning("CLIP not available, vision=false.")
        return False

    try:
        import torch
        from PIL import Image
    except Exception as e:
        logger.warning(f"Could not import torch/PIL for vision: {e}")
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

        text = (c.get("text") or "").strip()
        if not text:
            continue

        clean_text = text.replace("\n", " ").strip()
        if len(clean_text) > 250:
            clean_text = clean_text[:250]
        if len(clean_text) < 5:
            clean_text = "short video clip"

        mid_t = (safe_float(c.get("start", 0.0)) + safe_float(c.get("end", 0.0))) / 2.0
        frame_path = os.path.join(session_dir, f"frame_{c['id']}.jpg")

        if not grab_frame_at_timestamp(input_local, mid_t, frame_path):
            continue

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception:
            continue

        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize([clean_text]).to(device)  # type: ignore

            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
            visual_score = (similarity + 1.0) / 2.0

        c["visual_score"] = float(visual_score)
        c["meta"]["visual_score"] = float(visual_score)

    return True


def reject_visual_bad_takes(clips: List[Dict[str, Any]], session_dir: str, input_local: str):
    """
    Marca takes visualmente MUY malos como keep=False (nivel entero de clip).
    """
    from openai import OpenAI
    import base64

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return

    client = OpenAI(api_key=api_key)

    for c in clips:
        if not c["meta"].get("keep", True):
            continue

        mid = (safe_float(c["start"]) + safe_float(c["end"])) / 2.0
        frame_path = os.path.join(session_dir, f"facecheck_{c['id']}.jpg")

        if not grab_frame_at_timestamp(input_local, mid, frame_path):
            continue

        try:
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            continue

        prompt = f"""
        Analyze if this take is a GOOD or BAD acting take for a TikTok ad.
        Say ONLY: 'GOOD' or 'BAD'.
        Criteria:
        - GOOD: Confident, connected, intentional.
        - BAD: Looks confused, thinking, adjusting, flat face, or doesn’t match the energy of the text.
        Text of the clip: "{c.get('text')}"
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You strictly output 'GOOD' or 'BAD'."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ],
        )

        verdict = (response.choices[0].message.content or "").strip().upper()
        if verdict == "BAD":
            c["meta"]["keep"] = False
            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed for visual bad-take."


# =====================
# VISION BOUNDARY (HEAD/TAIL) TRIMMER
# =====================

def refine_clip_boundaries_with_vision(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
) -> bool:
    """
    Cirugía fina DENTRO del clip:
    - Mira HEAD / MID / TAIL (3 frames) por clip.
    - Si HEAD = BAD, MID = GOOD → mueve start hacia head_t.
    - Si TAIL = BAD, MID = GOOD → mueve end hacia tail_t.
    - Si HEAD+MID+TAIL = BAD → mata el clip completo.
    Sólo afecta clips con meta.keep=True.
    """

    if not BOUNDARY_REFINER_ENABLED:
        logger.info("BOUNDARY_REFINER_ENABLED=0, skipping boundary refiner.")
        return False

    from openai import OpenAI
    import base64

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("Boundary refiner needs OPENAI_API_KEY, skipping.")
        return False

    client = OpenAI(api_key=api_key)
    changed_any = False

    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """
        Hace parsing robusto:
        - quita ```json ... ``` si viene en code-fence
        - recorta hasta el primer '{' y el último '}'
        """
        if not raw:
            return None
        txt = raw.strip()
        if txt.startswith("```"):
            # quitar ```json\n y el cierre ```
            parts = txt.split("```")
            if len(parts) >= 3:
                txt = parts[1]
            txt = txt.strip()
        # recortar a rango [primera '{' .. última '}']
        start_i = txt.find("{")
        end_i = txt.rfind("}")
        if start_i == -1 or end_i == -1 or end_i <= start_i:
            return None
        txt = txt[start_i : end_i + 1]
        try:
            return json.loads(txt)
        except Exception:
            return None

    for c in clips:
        if not c["meta"].get("keep", True):
            continue

        start = safe_float(c.get("start", 0.0))
        end = safe_float(c.get("end", start))
        duration = end - start

        # clips muy cortos no se tocan
        if duration <= BOUNDARY_REFINER_MIN_DURATION_SEC:
            continue

        text = (c.get("text") or "").strip()

        head_t = start + min(BOUNDARY_REFINER_HEAD_STEP_SEC, duration / 3.0)
        mid_t = start + duration / 2.0
        tail_t = max(start, end - min(BOUNDARY_REFINER_TAIL_STEP_SEC, duration / 3.0))

        frames_payload: List[Dict[str, Any]] = []
        for label, t in (("head", head_t), ("mid", mid_t), ("tail", tail_t)):
            frame_path = os.path.join(
                session_dir, f"boundary_{c['id']}_{label}.jpg"
            )
            if not grab_frame_at_timestamp(input_local, t, frame_path):
                continue
            try:
                with open(frame_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                continue
            frames_payload.append({"label": label, "t": t, "image_b64": img_b64})

        if not frames_payload:
            continue

        system_msg = (
            "You are a TikTok ad editor. You will see several frames from the SAME take: "
            "head, mid, and tail. For each frame decide if the acting moment is GOOD or BAD. "
            "BAD examples: laughing out of character, fixing hair, adjusting clothes, "
            "looking away confused, mouth half-open mid-word, clearly not speaking to camera. "
            "GOOD examples: speaking naturally to camera, looks intentional and confident."
        )

        meta_for_prompt = [
            {"label": f["label"], "t": f["t"]} for f in frames_payload
        ]

        user_text = (
            "Return ONLY a JSON like:\n"
            "{\n"
            '  "frames": [\n'
            '    {"label": "head", "verdict": "GOOD"|"BAD"},\n'
            '    {"label": "mid",  "verdict": "GOOD"|"BAD"},\n'
            '    {"label": "tail", "verdict": "GOOD"|"BAD"}\n'
            "  ]\n"
            "}\n\n"
            "Do not add explanations outside the JSON.\n\n"
            f"Spoken text of this clip:\n{json.dumps(text, ensure_ascii=False)}\n\n"
            f"Frames metadata (label + timestamp_seconds):\n"
            f"{json.dumps(meta_for_prompt, ensure_ascii=False)}\n"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": user_text}]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{f['image_b64']}"
                            },
                        }
                        for f in frames_payload
                    ]
                ),
            },
        ]

        try:
            resp = client.chat.completions.create(
                model="gpt-5.1",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
            )
        except Exception as e:
            logger.warning(f"Boundary refiner LLM error for clip {c['id']}: {e}")
            continue

        try:
            content = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Boundary refiner empty response for clip {c['id']}: {e}")
            continue

        data = _safe_parse_json(content)
        if not data:
            logger.warning(
                f"Boundary refiner JSON parse error for clip {c['id']}: raw={content[:120]!r}"
            )
            continue

        verdicts_map = {
            item.get("label", ""): (item.get("verdict", "") or "").upper()
            for item in data.get("frames", [])
            if isinstance(item, dict)
        }

        head_bad = verdicts_map.get("head") == "BAD"
        mid_bad = verdicts_map.get("mid") == "BAD"
        tail_bad = verdicts_map.get("tail") == "BAD"

        # Caso extremo: todo malo → matar el clip completo.
        if head_bad and mid_bad and tail_bad:
            c["meta"]["keep"] = False
            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed by boundary refiner: full take visually bad."
            changed_any = True
            continue

        # Empezamos asumiendo mismo rango
        new_start = start
        new_end = end

        # Sólo recortamos alrededor si el mid es razonablemente bueno.
        if not mid_bad:
            if head_bad and head_t < end - 0.30:
                new_start = max(head_t, start)
            if tail_bad and tail_t > start + 0.30:
                new_end = min(tail_t, end)

        # Si después del recorte queda ridículamente corto, lo matamos
        if new_end <= new_start or (new_end - new_start) < max(0.5, duration * 0.25):
            c["meta"]["keep"] = False
            c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed by boundary refiner: too short after trim."
            changed_any = True
            continue

        if new_start != start or new_end != end:
            c["start"] = new_start
            c["end"] = new_end
            c["llm_reason"] = (c.get("llm_reason") or "") + " | Head/tail refined by boundary refiner."
            changed_any = True

    return changed_any

# =====================
# TEXT UTILS (para TakeJudge y dedupe)
# =====================

def normalize_text(t: str) -> str:
    return " ".join((t or "").lower().strip().split())


def text_overlap_ratio(t1: str, t2: str) -> float:
    a = normalize_text(t1)
    b = normalize_text(t2)
    if not a or not b:
        return 0.0
    set1 = set(a.split())
    set2 = set(b.split())
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    if union <= 0:
        return 0.0
    return inter / union


def text_overlap_shorter(t1: str, t2: str) -> float:
    a = normalize_text(t1)
    b = normalize_text(t2)
    if not a or not b:
        return 0.0
    set1 = set(a.split())
    set2 = set(b.split())
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    denom = min(len(set1), len(set2))
    if denom <= 0:
        return 0.0
    return inter / denom


# =====================
# TAKE JUDGE (MULTI-TAKE HUMANO)
# =====================

def find_sibling_groups(
    clips: List[Dict[str, Any]],
    window_sec: float = 18.0,
    min_overlap: float = 0.55,
) -> List[List[Dict[str, Any]]]:
    usable = [
        c for c in clips
        if c["meta"].get("keep", True)
        and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
    ]
    usable = sorted(usable, key=lambda c: safe_float(c.get("start", 0.0)))

    groups: List[List[Dict[str, Any]]] = []
    used_ids = set()

    for i, c1 in enumerate(usable):
        if c1["id"] in used_ids:
            continue

        group = [c1]
        t1 = safe_float(c1.get("start", 0.0))
        slot1 = c1.get("slot", "STORY")
        text1 = normalize_text(c1.get("text", ""))

        for j in range(i + 1, len(usable)):
            c2 = usable[j]
            if c2["id"] in used_ids:
                continue
            if c2.get("slot", "STORY") != slot1:
                continue

            t2 = safe_float(c2.get("start", 0.0))
            if t2 - t1 > window_sec:
                break

            text2 = normalize_text(c2.get("text", ""))
            overlap = text_overlap_ratio(text1, text2)
            if overlap >= min_overlap:
                group.append(c2)

        if len(group) >= 2:
            for g in group:
                used_ids.add(g["id"])
            groups.append(group)

    return groups


def run_take_judge(
    clips: List[Dict[str, Any]],
    session_dir: str,
    input_local: str,
) -> bool:
    """
    TakeJudgeAI:
    - Para cada grupo de takes similares (mismo slot + texto casi igual)
      pregunta a un LLM visión+texto cuál es la mejor actuación.
    - Sólo 1 ganador por grupo (LOSER → keep=False).
    """
    from openai import OpenAI
    import base64

    if not TAKE_JUDGE_ENABLED:
        return False

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("TAKE_JUDGE_ENABLED=1 but OPENAI_API_KEY is missing.")
        return False

    client = OpenAI(api_key=api_key)

    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        txt = raw.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 3:
                txt = parts[1]
            txt = txt.strip()
        start_i = txt.find("{")
        end_i = txt.rfind("}")
        if start_i == -1 or end_i == -1 or end_i <= start_i:
            return None
        txt = txt[start_i : end_i + 1]
        try:
            return json.loads(txt)
        except Exception:
            return None

    groups = find_sibling_groups(clips)
    if not groups:
        return False

    # Limitar por presupuesto
    groups = groups[:TAKE_JUDGE_MAX_GROUPS]
    groups = [g[:TAKE_JUDGE_MAX_TAKES] for g in groups]

    used_any = False

    for group in groups:
        payload = []
        for c in group:
            if not c["meta"].get("keep", True):
                continue

            mid = (safe_float(c["start"]) + safe_float(c["end"])) / 2.0
            frame_path = os.path.join(session_dir, f"takejudge_{c['id']}.jpg")

            if not grab_frame_at_timestamp(input_local, mid, frame_path):
                continue

            try:
                with open(frame_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                continue

            payload.append(
                {
                    "id": c["id"],
                    "text": (c.get("text") or "").strip(),
                    "image_b64": img_b64,
                }
            )

        if not payload:
            continue

        system_msg = (
            "You are a senior TikTok ad editor. "
            "You receive multiple takes of the SAME line. "
            "Pick the best acting take: natural, confident, persuasive. "
            "Ignore small wording differences if the performance is better."
        )

        user_json = {
            "task": "choose_best_take",
            "takes": [{"id": p["id"], "text": p["text"]} for p in payload],
        }

        user_text = (
            "Return ONLY a JSON like:\n"
            "{\n"
            '  \"winner_id\": \"...\",\n'
            '  \"scores\": [\n'
            '    {\"id\": \"...\", \"score\": 0.xx}\n'
            "  ]\n"
            "}\n\n"
            "Do not add any explanation outside the JSON.\n\n"
            f"Takes metadata:\n{json.dumps(user_json, ensure_ascii=False)}"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": user_text}]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{p['image_b64']}"
                            },
                        }
                        for p in payload
                    ]
                ),
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=TAKE_JUDGE_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=150,
            )
        except Exception as e:
            logger.warning(f"TakeJudgeAI error for group: {e}")
            continue

        try:
            content = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"TakeJudgeAI empty response: {e}")
            continue

        data = _safe_parse_json(content)
        if not data:
            logger.warning(
                f"TakeJudgeAI JSON parse error: raw={content[:120]!r}"
            )
            continue

        winner_id = data.get("winner_id")
        scores_map = {
            s.get("id"): safe_float(s.get("score", 0.0))
            for s in data.get("scores", [])
            if isinstance(s, dict) and s.get("id")
        }

        if not winner_id:
            logger.warning("TakeJudgeAI response without winner_id, skipping group.")
            continue

        # Guardamos scores + WINNER / LOSER
        for c in group:
            cid = c["id"]
            tj_score = scores_map.get(cid, 0.0)
            c["meta"]["take_judge_score"] = tj_score
            c["meta"]["take_judge_verdict"] = (
                "WINNER" if cid == winner_id else "LOSER"
            )

        # Sólo dejamos vivo el ganador
        for c in group:
            cid = c["id"]
            if cid != winner_id and c["meta"].get("keep", True):
                c["meta"]["keep"] = False
                c["llm_reason"] = (c.get("llm_reason") or "") + " | Removed by TakeJudgeAI (better take exists)."

        used_any = True

    return used_any


# =====================
# SCORE THRESHOLDS + DEDUPE
# =====================

def apply_min_score_rules(clips: List[Dict[str, Any]]) -> None:
    for c in clips:
        slot = c.get("slot", "STORY")
        combined = safe_float(c.get("score", 0.0))
        sem = safe_float(c.get("semantic_score", 0.0))

        threshold = EDITDNA_MIN_CLIP_SCORE
        if slot == "HOOK":
            threshold = max(threshold, EDITDNA_HOOK_MIN_SCORE)
        elif slot == "CTA":
            threshold = max(threshold, EDITDNA_CTA_MIN_SCORE)

        effective_score = sem if sem > 0 else combined

        if effective_score < threshold:
            c["meta"]["keep"] = False


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
# SLOTS + COMPOSER HELPERS
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


def suppress_near_duplicates_by_slot(
    clips: List[Dict[str, Any]],
    window_sec: float = 60.0,
    min_overlap: float = 0.35,
) -> None:
    n = len(clips)
    for i in range(n):
        c1 = clips[i]
        if not c1["meta"].get("keep", True):
            continue
        slot1 = c1.get("slot", "STORY")
        t1 = safe_float(c1.get("start", 0.0))

        for j in range(i + 1, n):
            c2 = clips[j]
            if not c2["meta"].get("keep", True):
                continue

            if c2.get("slot", "STORY") != slot1:
                continue

            t2 = safe_float(c2.get("start", 0.0))
            if t2 - t1 > window_sec:
                break

            text1 = c1.get("text", "") or ""
            text2 = c2.get("text", "") or ""

            ratio = text_overlap_ratio(text1, text2)
            shorter_overlap = text_overlap_shorter(text1, text2)

            if ratio < min_overlap and shorter_overlap < 0.65:
                continue

            sem1 = safe_float(c1.get("semantic_score", 0.0))
            sem2 = safe_float(c2.get("semantic_score", 0.0))
            len1 = len(text1.split())
            len2 = len(text2.split())

            if sem2 > sem1 or (sem2 == sem1 and len2 >= len1):
                c1["meta"]["keep"] = False
                break
            else:
                c2["meta"]["keep"] = False


def suppress_cross_slot_redundant_clips(
    clips: List[Dict[str, Any]],
    window_sec: float = 18.0,
    min_overlap: float = 0.35,
) -> None:
    n = len(clips)
    for i in range(n):
        c1 = clips[i]
        if not c1["meta"].get("keep", True):
            continue

        t1 = safe_float(c1.get("start", 0.0))
        text1 = normalize_text(c1.get("text", ""))

        for j in range(i + 1, n):
            c2 = clips[j]
            if not c2["meta"].get("keep", True):
                continue

            t2 = safe_float(c2.get("start", 0.0))
            if t2 - t1 > window_sec:
                break

            text2 = normalize_text(c2.get("text", ""))
            if not text1 or not text2:
                continue

            overlap = text_overlap_ratio(text1, text2)

            sem1 = safe_float(c1.get("semantic_score", 0.0))
            sem2 = safe_float(c2.get("semantic_score", 0.0))

            if overlap >= min_overlap and len(text2.split()) > len(text1.split()):
                if sem2 >= sem1:
                    c1["meta"]["keep"] = False
                    break

            if text1 in text2 and len(text2.split()) - len(text1.split()) > 3:
                c1["meta"]["keep"] = False
                break


def group_contiguous_blocks_by_slot(
    clips: List[Dict[str, Any]],
    slot: str,
    max_gap_sec: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    ordered = sorted(
        [c for c in clips if c.get("slot") == slot and c["meta"].get("keep", True)],
        key=lambda c: safe_float(c.get("start", 0.0)),
    )

    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_end: Optional[float] = None

    for c in ordered:
        start = safe_float(c.get("start", 0.0))
        end = safe_float(c.get("end", start))

        if not current:
            current = [c]
            last_end = end
            continue

        gap = start - safe_float(last_end if last_end is not None else start)
        if gap <= max_gap_sec:
            current.append(c)
            last_end = end
        else:
            if current:
                blocks.append(current)
            current = [c]
            last_end = end

    if current:
        blocks.append(current)

    return blocks


def pick_best_block(blocks: List[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    best_block: Optional[List[Dict[str, Any]]] = None
    best_score = -1.0

    for block in blocks:
        if not block:
            continue
        scores = [
            safe_float(c.get("score", c.get("semantic_score", 0.0))) for c in block
        ]
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_block = block

    return best_block


def build_composer(clips: List[Dict[str, Any]], mode: str = "human") -> Dict[str, Any]:
    mode = (mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

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

    suppress_near_duplicates_by_slot(usable)
    usable = [c for c in usable if c["meta"].get("keep", True)]

    suppress_cross_slot_redundant_clips(usable)
    usable = [c for c in usable if c["meta"].get("keep", True)]

    if mode == "clean":
        for c in usable:
            if c.get("slot") == "STORY":
                c["meta"]["keep"] = False
        usable = [c for c in usable if c["meta"].get("keep", True)]

    elif mode == "blooper":
        for c in usable:
            slot = c.get("slot")
            if slot not in {"STORY", "HOOK", "CTA"}:
                c["meta"]["keep"] = False
        usable = [c for c in usable if c["meta"].get("keep", True)]

    if not usable:
        usable = [
            c
            for c in clips
            if c["meta"].get("keep", True)
            and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
        ]
        usable.sort(key=lambda c: safe_float(c.get("start", 0.0)))

    hook_blocks = group_contiguous_blocks_by_slot(usable, "HOOK", max_gap_sec=1.0)
    best_hook_block = pick_best_block(hook_blocks) if hook_blocks else None
    hook_block_ids = [c["id"] for c in best_hook_block] if best_hook_block else []

    if hook_block_ids:
        first_start = safe_float(best_hook_block[0].get("start", 0.0))
        if first_start > 10.0:
            hook_block_ids = []

    problem_blocks = group_contiguous_blocks_by_slot(usable, "PROBLEM", max_gap_sec=1.0)
    best_problem_block = pick_best_block(problem_blocks) if problem_blocks else None
    problem_block_ids = [c["id"] for c in best_problem_block] if best_problem_block else []

    cta_blocks = group_contiguous_blocks_by_slot(usable, "CTA", max_gap_sec=1.0)
    best_cta_block = pick_best_block(cta_blocks) if cta_blocks else None
    cta_block_ids = [c["id"] for c in best_cta_block] if best_cta_block else []

    ctas = [c for c in usable if c.get("slot") == "CTA"]
    single_best_cta = None
    if ctas:
        ctas_sorted = sorted(
            ctas, key=lambda c: safe_float(c.get("score", 0.0)), reverse=True
        )
        single_best_cta = ctas_sorted[0]

    if not cta_block_ids and single_best_cta is not None:
        cta_block_ids = [single_best_cta["id"]]

    cta_final_ids = cta_block_ids[:]
    cta_final_ids_set = set(cta_final_ids)

    timeline: List[Dict[str, Any]] = []
    used_ids: List[str] = []

    for c in usable:
        if c["id"] in cta_final_ids_set:
            continue
        timeline.append(c)
        used_ids.append(c["id"])

    if cta_final_ids:
        cta_block_clips = [c for c in usable if c["id"] in cta_final_ids_set]
        cta_block_clips.sort(key=lambda c: safe_float(c.get("start", 0.0)))
        for c in cta_block_clips:
            timeline.append(c)
            used_ids.append(c["id"])

    def ids_for_slot(slot_name: str) -> List[str]:
        return [c["id"] for c in timeline if c.get("slot") == slot_name]

    if hook_block_ids:
        hook_id = hook_block_ids[0]
    else:
        hook_id = next(
            (c["id"] for c in timeline if c.get("slot") == "HOOK"), None
        )

    problem_ids = ids_for_slot("PROBLEM")
    if problem_block_ids:
        filtered = [cid for cid in problem_ids if cid in problem_block_ids]
        if filtered:
            problem_ids = filtered
    problem_ids = problem_ids[:COMPOSER_MAX_PER_SLOT]

    benefit_ids = ids_for_slot("BENEFITS")[:COMPOSER_MAX_PER_SLOT]
    feature_ids = ids_for_slot("FEATURES")[:COMPOSER_MAX_PER_SLOT]
    proof_ids = ids_for_slot("PROOF")[:COMPOSER_MAX_PER_SLOT]

    cta_id = cta_final_ids[-1] if cta_final_ids else None

    composer = {
        "mode": mode,
        "hook_id": hook_id,
        "story_ids": ids_for_slot("STORY")[:COMPOSER_MAX_PER_SLOT],
        "problem_ids": problem_ids,
        "benefit_ids": benefit_ids,
        "feature_ids": feature_ids,
        "proof_ids": proof_ids,
        "cta_id": cta_id,
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
            return f"[{cid}] (not found)"
        return f"[{cid}] score={c.get('score', 0.0):.2f} → \"{c.get('text', '').strip()}\""

    parts = ["===== EDITDNA FUNNEL COMPOSER ====="]

    mode = composer.get("mode")
    if mode:
        parts.append(f"MODE: {mode}")

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
            parts.append(f"{i}) {cid} (not found)")
        else:
            parts.append(f"{i}) {cid} → \"{c.get('text', '').strip()}\"")

    parts.append("\n=====================================")
    return "\n".join(parts)


# =====================
# FFMPEG RENDER  ❗ MODULO QUE PEGA LOS CORTES
# =====================

def render_funnel_video(
    input_local: str,
    session_dir: str,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> str:
    """
    ESTE ES EL MÓDULO QUE:
      - Toma los clips ya elegidos (used_clip_ids)
      - Usa su start/end FINAL
      - Corta y PEGA en ffmpeg para crear final.mp4
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

        raw_start = safe_float(c.get("start", 0.0))
        raw_end = safe_float(c.get("end", 0.0))

        start = max(0.0, raw_start + HEAD_TRIM_SEC)
        end = max(start, raw_end - TAIL_TRIM_SEC)

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
            "render_funnel_video: has_audio=True but len(a_labels)"
            f"={len(a_labels)} != len(v_labels)={n}, "
            "disabling audio to avoid media type mismatch."
        )
    has_audio = has_audio and len(a_labels) == n

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
# MAIN ENTRYPOINT
# =====================

def run_pipeline(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Main pipeline entrypoint.

    mode:
      - "human"
      - "clean"
      - "blooper"
    """
    logger.info(
        f"run_pipeline session_id={session_id} mode={mode} files={files} file_urls={file_urls}"
    )

    effective_files: Optional[List[str]] = None
    if files and isinstance(files, list):
        effective_files = files
    elif file_urls and isinstance(file_urls, list):
        effective_files = file_urls

    if not effective_files:
        raise ValueError(
            "run_pipeline: 'files' or 'file_urls' must be a list with at least 1 URL"
        )

    mode = (mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

    session_dir = ensure_session_dir(session_id)
    input_local = os.path.join(session_dir, "input.mp4")

    download_to_local(effective_files[0], input_local)

    duration = probe_duration(input_local)

    # =====================
    # PHASE 1: ANALYSIS
    # =====================
    asr_segments = run_asr(input_local)

    clips = sentence_boundary_micro_cuts(asr_segments)
    clips = merge_incomplete_phrases(clips)

    llm_used = enrich_clips_semantic(clips)
    clips = dedupe_clips(clips)

    vision_used = run_visual_pass(input_local, session_dir, clips)

    bad_takes_used = False
    if VISION_ENABLED and BAD_TAKES_ENABLED:
        reject_visual_bad_takes(clips, session_dir, input_local)
        bad_takes_used = True

    boundaries_refined = False
    if VISION_ENABLED and BOUNDARY_REFINER_ENABLED:
        boundaries_refined = refine_clip_boundaries_with_vision(
            input_local=input_local,
            session_dir=session_dir,
            clips=clips,
        )

    take_judge_used = False
    if TAKE_JUDGE_ENABLED:
        take_judge_used = run_take_judge(
            clips=clips,
            session_dir=session_dir,
            input_local=input_local,
        )

    slots = build_slots_dict(clips)

    # =====================
    # PHASE 2: COMPOSER + RENDER
    # =====================
    composer = build_composer(clips, mode=mode)
    used_clip_ids = composer.get("used_clip_ids", [])
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)

    output_url: Optional[str] = None
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
        "take_judge_used": take_judge_used,
        "bad_takes_used": bad_takes_used,
        "boundaries_refined": boundaries_refined,
        "composer_mode": mode,
    }
    return result
