import os
import json
import logging
import subprocess
import copy
import shutil
import tempfile
import re
import inspect
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Union, Sequence

import requests
from faster_whisper import WhisperModel
import boto3


from worker.models.config import load_model_config
from worker.models.openai_client import OpenAIProviderError, is_openai_available
from worker.models.openai_provider import (
    Verdict, classify_semantic_v2, detect_bad_take, judge_takes_v2, refine_boundaries,
)
from worker.semantic_slot_v2 import build_clause_inputs, publicize_canonical_slot
from worker.take_judge_v2 import TakeJudgeCandidate, delivery_features, sample_candidate_frames
from worker.diagnostics import (
    sanitize_clean_cut_discard_diagnostics,
    sanitize_source_identifier,
)
from worker.text_normalization import (
    APOSTROPHE_TRANSLATION,
    NormalizedText,
    comparison_units,
    normalize_match_text,
    normalized_text,
    semantic_content_measure,
    semantic_content_score,
    unicode_word_tokens,
)

from pipeline_errors import (
    JobCanceledError,
    MissingSelectedClipsError,
    PipelineError,
    SelectionError,
    UploadError,
)

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
MODEL_CONFIG = load_model_config()
WHISPER_MODEL_NAME = MODEL_CONFIG.asr.model_name
WHISPER_DEVICE = MODEL_CONFIG.asr.device
ASR_ENABLED = MODEL_CONFIG.asr.enabled

# Composer / scores
COMPOSER_MIN_SEMANTIC = MODEL_CONFIG.global_composer.min_semantic_score
COMPOSER_MAX_PER_SLOT = MODEL_CONFIG.global_composer.max_per_slot
MICRO_SENTENCE_MAX_SECONDS = float(
    os.environ.get("MICRO_SENTENCE_MAX_SECONDS", "8.0")
)

EDITDNA_MIN_CLIP_SCORE = float(os.environ.get("EDITDNA_MIN_CLIP_SCORE", "0.7"))
EDITDNA_HOOK_MIN_SCORE = float(os.environ.get("EDITDNA_HOOK_MIN_SCORE", "0.7"))
EDITDNA_CTA_MIN_SCORE = float(os.environ.get("EDITDNA_CTA_MIN_SCORE", "0.6"))

# LLM
EDITDNA_USE_LLM = MODEL_CONFIG.semantic_llm.enabled
EDITDNA_LLM_MODEL = MODEL_CONFIG.semantic_llm.model_name
SEMANTIC_V2_MIN_CONFIDENCE = MODEL_CONFIG.semantic_llm.min_confidence

# VISION
VISION_ENABLED = MODEL_CONFIG.vision.enabled
VISION_MODEL = MODEL_CONFIG.vision.model_name
VISION_INTERVAL_SEC = MODEL_CONFIG.vision.interval_seconds
VISION_MAX_SAMPLES = MODEL_CONFIG.vision.max_samples
W_VISION = MODEL_CONFIG.vision.weight

# BAD TAKES (visual face check) – opcional
BAD_TAKES_ENABLED = MODEL_CONFIG.visual_bad_take.enabled
BAD_TAKES_MODEL = MODEL_CONFIG.visual_bad_take.model_name

# BOUNDARY TRIM (head/tail dentro del clip, opcional)
BOUNDARY_REFINER_ENABLED = MODEL_CONFIG.boundary_refiner.enabled
BOUNDARY_REFINER_MODEL = MODEL_CONFIG.boundary_refiner.model_name
BOUNDARY_REFINER_MIN_DURATION_SEC = MODEL_CONFIG.boundary_refiner.min_duration_seconds
BOUNDARY_REFINER_HEAD_STEP_SEC = MODEL_CONFIG.boundary_refiner.head_step_seconds
BOUNDARY_REFINER_TAIL_STEP_SEC = MODEL_CONFIG.boundary_refiner.tail_step_seconds

# TAKE JUDGE (multi-take selection)
TAKE_JUDGE_ENABLED = MODEL_CONFIG.take_judge.enabled
TAKE_JUDGE_MODEL = MODEL_CONFIG.take_judge.model_name
TAKE_JUDGE_MAX_GROUPS = MODEL_CONFIG.take_judge.max_groups
TAKE_JUDGE_MAX_TAKES = MODEL_CONFIG.take_judge.max_takes
TAKE_JUDGE_FRAMES = MODEL_CONFIG.take_judge.frame_count
TAKE_JUDGE_V2_MIN_CONFIDENCE = MODEL_CONFIG.take_judge.min_confidence

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

# Head/tail trims globales (post selección)
HEAD_TRIM_SEC = float(os.environ.get("HEAD_TRIM_SEC", "0.0"))
TAIL_TRIM_SEC = float(os.environ.get("TAIL_TRIM_SEC", "0.0"))
OUTPUT_WIDTH = int(os.environ.get("OUTPUT_WIDTH", "1080"))
OUTPUT_HEIGHT = int(os.environ.get("OUTPUT_HEIGHT", "1920"))
OUTPUT_FPS = int(os.environ.get("OUTPUT_FPS", "30"))


# =====================
# HELPERS
# =====================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def ensure_session_dir(session_id: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    # A unique directory makes ownership unambiguous and avoids concurrent jobs
    # with the same externally supplied session ID deleting each other's files.
    safe_session_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return tempfile.mkdtemp(prefix=f"{safe_session_id or 'session'}-", dir=TMP_DIR)


def download_to_local(url: str, dst_path: str) -> None:
    logger.info("Downloading input to job temporary storage")
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


def upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    try:
        s3 = boto3.client("s3")
        logger.info("Uploading rendered output to configured S3 bucket")
        s3.upload_file(local_path, bucket, key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        if not isinstance(url, str) or not url.strip():
            raise ValueError("S3 did not return a non-empty output URL")
        return url
    except Exception as exc:
        # Do not interpolate the provider exception: it may contain a signed URL.
        logger.error("S3 output upload failed")
        raise UploadError("Failed to upload rendered output") from exc
        
def save_result_json_to_s3(result: Dict[str, Any]) -> Optional[str]:
    """
    Guarda el JSON completo del pipeline en S3 como dataset.
    Devuelve el path s3://... o None si falla.
    """
    bucket = S3_BUCKET
    if not bucket:
        logger.warning("S3_BUCKET is not set, skipping JSON upload.")
        return None

    try:
        s3 = boto3.client("s3")
        session_id = result.get("session_id", "unknown")
        # Puedes cambiar 'dataset/bloopers' por lo que quieras
        key = f"{S3_PREFIX}/dataset/bloopers/{session_id}.json"

        persistable_result = copy.deepcopy(result)
        persistable_result["clean_cut_discard_diagnostics"] = sanitize_clean_cut_discard_diagnostics(
            persistable_result.get("clean_cut_discard_diagnostics", [])
        )
        body = json.dumps(persistable_result, ensure_ascii=False).encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        logger.info(f"Uploaded JSON result to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"
    except Exception as e:
        logger.exception(f"Error uploading JSON result to S3: {e}")
        return None


# =====================
# CLIP OBJECT CREATION
# =====================

def make_base_clip(
    cid: str, start: float, end: float, text: str,
    words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
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
    if words:
        clip_obj["words"] = [
            {"start": safe_float(word.get("start")),
             "end": safe_float(word.get("end", word.get("start"))),
             "word": str(word.get("word", ""))}
            for word in words
            if (safe_float(word.get("end", word.get("start"))) > start
                and safe_float(word.get("start")) < end)
        ]
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
                words=buffer_words,
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
    if semantic_content_measure(t).effective_semantic_units <= 2:
        return True

    bad_endings = ("for", "the", "a", "my", "your", "our", "this", "that")
    if t.endswith(bad_endings):
        return True

    if not t.endswith((".", "?", "!")):
        if t.startswith(("and ", "so ", "but ", "or ")):
            return True
        return True

    return False


def looks_like_completed_spoken_thought(text: str) -> bool:
    """Clean-cut boundary cue that does not depend on commercial slot labels."""
    n = normalized_text(text)
    if n.text.endswith((".", "?", "!")):
        return True
    return has_any_phrase(n, (
        "drop it down below", "check these out", "check them out",
        "link below", "link in bio", "get yours",
    ))


def is_semantic_restart_after_complete(previous_text: str, next_text: str) -> bool:
    prev = normalized_text(previous_text)
    nxt = normalized_text(next_text)
    if not prev.compact or not nxt.compact:
        return False
    if not looks_like_completed_spoken_thought(previous_text):
        return False
    return any(starts_phrase(nxt, phrase) for phrase in (
        "i found", "i found the perfect", "here's", "here is", "let me show",
        "wait until", "if you", "have you", "are you", "stop scrolling",
    ))


def merge_incomplete_phrases(
    clips: List[Dict[str, Any]],
    discarded_diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
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
            if i + 1 < len(clips) and is_semantic_restart_after_complete(text, clips[i + 1].get("text", "")):
                c.setdefault("meta", {})["merge_diagnostic"] = "merge_prevented_semantic_restart"
            merged.append(c)
            i += 1
            continue

        if i + 1 < len(clips):
            nxt = clips[i + 1]
            next_text = nxt["text"].strip()

            can_merge = (
                semantic_content_measure(next_text).effective_semantic_units >= 2 and
                not looks_incomplete(next_text) and
                text[0].isalpha() and
                not is_semantic_restart_after_complete(text, next_text)
            )
            restart_prevented = not can_merge and is_semantic_restart_after_complete(text, next_text)
            if restart_prevented:
                c.setdefault("meta", {})["merge_diagnostic"] = "merge_prevented_semantic_restart"
                merged.append(c)
                i += 1
                continue

            if can_merge:
                new_text = (text + " " + next_text).strip()
                combined_words = []
                seen_words = set()
                for word in sorted(
                    (c.get("words") or []) + (nxt.get("words") or []),
                    key=lambda item: (safe_float(item.get("start")), safe_float(item.get("end"))),
                ):
                    normalized = {
                        "start": safe_float(word.get("start")),
                        "end": safe_float(word.get("end", word.get("start"))),
                        "word": str(word.get("word", "")),
                    }
                    identity = (normalized["start"], normalized["end"], normalized["word"])
                    if identity not in seen_words:
                        seen_words.add(identity)
                        combined_words.append(normalized)
                merged_clip = {
                    **c,
                    "text": new_text,
                    "end": nxt["end"],
                    "chain_ids": c["chain_ids"] + nxt["chain_ids"],
                }
                if "source_end" in c or "source_end" in nxt:
                    merged_clip["source_end"] = safe_float(nxt.get("source_end", nxt["end"]))
                if c.get("words") or nxt.get("words"):
                    merged_clip["words"] = combined_words
                merged.append(merged_clip)
                i += 2
                continue

        i += 1

    return validate_clip_boundaries(merged, discarded_diagnostics=discarded_diagnostics)


def _word_bounds(clip: Dict[str, Any]) -> Optional[tuple[float, float]]:
    words = clip.get("words") or []
    valid = [
        (safe_float(w.get("start")), safe_float(w.get("end", w.get("start"))))
        for w in words
        if safe_float(w.get("end", w.get("start"))) > safe_float(w.get("start"))
    ]
    if not valid:
        return None
    return min(s for s, _ in valid), max(e for _, e in valid)


def _clip_source_identity(clip: Dict[str, Any]) -> tuple[Any, Any]:
    return (clip.get("source_index"), clip.get("source_local"))


def _shares_chain_or_source_boundary(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    previous_chain = set(previous.get("chain_ids") or previous.get("meta", {}).get("chain_ids") or [])
    current_chain = set(current.get("chain_ids") or current.get("meta", {}).get("chain_ids") or [])
    if previous_chain and current_chain and previous_chain.intersection(current_chain):
        return True

    previous_source = _clip_source_identity(previous)
    current_source = _clip_source_identity(current)
    known_source = any(value is not None for value in previous_source + current_source)
    if known_source and previous_source != current_source:
        return False

    previous_source_end = previous.get("source_end")
    current_source_start = current.get("source_start")
    if previous_source_end is not None and current_source_start is not None:
        return abs(safe_float(current_source_start) - safe_float(previous_source_end)) <= 0.25

    return not known_source


def _is_adjacent_residual_duplicate(previous: Dict[str, Any], current: Dict[str, Any], normalized_text: str) -> bool:
    if not previous or not normalized_text:
        return False
    previous_text = " ".join(_normalized_words(previous.get("text") or ""))
    if previous_text != normalized_text:
        return False

    previous_end = safe_float(previous.get("end", previous.get("start", 0.0)))
    current_start = safe_float(current.get("start", previous_end))
    current_end = safe_float(current.get("end", current_start))
    temporal_residual = current_start <= previous_end + 0.25 and current_end >= previous_end - 0.25
    return temporal_residual and _shares_chain_or_source_boundary(previous, current)


def _record_boundary_discard(
    diagnostics: Optional[List[Dict[str, Any]]], clip_item: Dict[str, Any], reason: str
) -> None:
    if diagnostics is None:
        return
    source_index = int(clip_item.get("source_index", 0))
    source_local = sanitize_source_identifier(clip_item.get("source_local"), source_index)
    original_clip_id = str(clip_item.get("original_clip_id", clip_item.get("id", "")))
    namespaced_clip_id = str(
        clip_item.get("namespaced_clip_id")
        or f"source_{source_index:03d}:{original_clip_id}"
    )
    diagnostics.append({
        # clip_id remains the original local ID for single-source consumers;
        # diagnostic_id/namespaced_clip_id provide deterministic global identity.
        "clip_id": original_clip_id,
        "original_clip_id": original_clip_id,
        "namespaced_clip_id": namespaced_clip_id,
        "diagnostic_id": f"{namespaced_clip_id}:{reason}",
        "source_index": source_index,
        "source_local": source_local,
        "reason": reason,
        "start": safe_float(clip_item.get("start")),
        "end": safe_float(clip_item.get("end", clip_item.get("start"))),
        "source_start": safe_float(clip_item.get("source_start", clip_item.get("start"))),
        "source_end": safe_float(clip_item.get("source_end", clip_item.get("end", clip_item.get("start")))),
        "text": (clip_item.get("text") or "").strip(),
    })


def validate_clip_boundaries(
    clips: List[Dict[str, Any]],
    min_duration: float = 0.08,
    discarded_diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for clip_item in clips:
        text = (clip_item.get("text") or "").strip()
        bounds = _word_bounds(clip_item)
        start = safe_float(clip_item.get("start"))
        end = safe_float(clip_item.get("end", start))
        if bounds and (end <= start or end - start < min_duration or start > bounds[0] + 0.02 or end < bounds[1] - 0.02):
            start, end = bounds
            clip_item["start"], clip_item["end"] = start, end
            if "source_start" in clip_item:
                clip_item["source_start"] = start
            if "source_end" in clip_item:
                clip_item["source_end"] = end
            clip_item.setdefault("meta", {})["boundary_diagnostic"] = "repaired_from_word_timestamps"
        duration = end - start
        if end <= start or (duration < min_duration and len(_normalized_words(text)) > 1):
            reason = "discarded_invalid_microfragment"
            clip_item.setdefault("meta", {})["boundary_diagnostic"] = reason
            _record_boundary_discard(discarded_diagnostics, clip_item, reason)
            continue
        normalized_text = " ".join(_normalized_words(text))
        if _is_adjacent_residual_duplicate(valid[-1] if valid else {}, clip_item, normalized_text):
            reason = "discarded_duplicate_residual_text"
            clip_item.setdefault("meta", {})["boundary_diagnostic"] = reason
            _record_boundary_discard(discarded_diagnostics, clip_item, reason)
            continue
        valid.append(clip_item)
    return valid


# =====================
# HEURISTIC TAGGING
# =====================

from worker.commercial_fallback import (
    _normalized_words,
    has_phrase,
    starts_phrase,
    has_any_phrase,
    is_camera_rolling_slate,
    _strip_token_prefixes,
    is_clear_recording_direction,
    is_compound_take_slate,
    production_meta_rule,
    filler_rule,
    looks_like_filler,
    looks_like_dependent_tail,
    CTAActionFrame,
    _action_target,
    _starts_explicit_cta,
    _cta_clauses,
    _strip_cta_discourse,
    _is_reported_or_narrated_action,
    cta_action_frames,
    is_explicit_link_cta,
    cta_action_rule,
    CommandIntent,
    command_intent,
    has_cta_action_context,
    has_product_enumeration_evidence,
    classify_slot_rule,
    classify_slot
)

def tag_clips_heuristic(clips: List[Dict[str, Any]]) -> None:
    for c in clips:
        text = c.get("text", "") or ""
        t = text.strip()

        slot, slot_rule = classify_slot_rule(t)
        is_tail = looks_like_dependent_tail(t)
        filler_reason = filler_rule(t)
        is_filler = filler_reason is not None

        # Clean Cut deletion is limited to explicit production/meta filler.
        # Linguistic dependency is useful context for ranking/composition but
        # must not delete otherwise valid speech.
        keep = not is_filler
        content = semantic_content_measure(t)

        if not t:
            sem = 0.0
        elif keep:
            sem = semantic_content_score(t)
        else:
            sem = 0.0

        if not keep:
            reason = "Explicit production/meta speech removed by Clean Cut."
        elif is_tail:
            reason = "Dependent context retained for semantic/composer evaluation."
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
        c["meta"]["semantic_content_measure"] = {
            "normalized_tokens": list(content.normalized_tokens),
            "normalized_token_count": content.token_count,
            "whitespace_token_count": content.whitespace_token_count,
            "unicode_alphanumeric_count": content.alphanumeric_count,
            "measurement_strategy": content.measurement_strategy,
            "effective_semantic_units": content.effective_semantic_units,
            "scoring_rule": content.scoring_rule,
            "repetition_noise_adjusted": content.repetition_noise_adjusted,
        }
        c["meta"]["score"] = sem
        c["meta"]["keep"] = keep
        c["meta"]["fallback_slot_rule"] = slot_rule
        if filler_reason:
            c["meta"]["filler_rule"] = filler_reason


# =====================
# SEMANTIC LLM (GPT-5.1)
# =====================

def llm_classify_clips(clips: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Run Semantic V2 through the provider layer; failures are deliberately fail-open."""
    clauses = build_clause_inputs(clips)
    if not clauses:
        return None
    try:
        return classify_semantic_v2(EDITDNA_LLM_MODEL, clauses, temperature=0.2)
    except OpenAIProviderError:
        logger.warning("Semantic classifier unavailable", extra={
            "operation": "semantic_classification_v2", "clause_count": len(clauses),
            "validation": "failed",
        })
        return None


def enrich_clips_semantic(clips: List[Dict[str, Any]], force_v2: bool = False) -> bool:
    """
    1) Heurística local (filler, slots, semantic_score)
    2) LLM opcional, con reglas de seguridad de embudo.
    """
    tag_clips_heuristic(clips)

    if not EDITDNA_USE_LLM and not force_v2:
        return False

    llm_result = llm_classify_clips(clips)
    if not llm_result:
        return False

    for c in clips:
        info = llm_result.get(c["id"])
        if info is None:
            continue
        c["meta"]["semantic_v2"] = {
            "primary_slot": info.primary_slot.value,
            "secondary_slot": info.secondary_slot.value if info.secondary_slot else None,
            "confidence": info.confidence,
            "secondary_confidence": info.secondary_confidence,
            "completeness": info.completeness,
            "sales_relevance": info.sales_relevance,
            "standalone_quality": info.standalone_quality,
            "abstain": info.abstain,
            "reason": info.reason,
            "evidence_tags": tuple(tag.value for tag in info.evidence_tags),
            "applied": False,
        }
        if info.abstain or info.confidence < SEMANTIC_V2_MIN_CONFIDENCE or info.completeness < 0.5:
            continue
        if info.primary_slot.value == "OTHER":
            # OTHER is a semantic label, never deletion authority. Preserve valid
            # speech and let the editable draft/composer decide presentation.
            c["slot"] = "OTHER"
            c["meta"]["slot"] = "OTHER"
            c["meta"]["semantic_v2"]["application_status"] = "applied_other_preserved"
            c["meta"]["semantic_v2"]["application_reason"] = "semantic_label_not_deletion_authority"
            c["meta"]["semantic_v2"]["applied"] = True
            continue
        slot_from_llm = publicize_canonical_slot(info.primary_slot)
        c["slot"] = slot_from_llm
        c["meta"]["slot"] = slot_from_llm
        c["llm_reason"] = info.reason or c.get("llm_reason", "")
        c["meta"]["semantic_v2"]["applied"] = True

    return True


# =====================
# VISION: CLIP + CUDA
# =====================

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = "cpu"
_CLIP_LIB = None


def load_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE, _CLIP_LIB
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    try:
        import torch  # sigue siendo necesario para CLIP
        import clip as clip_lib  # type: ignore
    except Exception as e:
        logger.warning(f"Could not import torch/clip for vision: {e}")
        return None, None, "cpu"

    # 🔒 FORZAMOS CPU SIEMPRE, aunque Torch diga que hay CUDA disponible
    device = "cpu"
    model, preprocess = clip_lib.load(VISION_MODEL, device=device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device
    _CLIP_LIB = clip_lib

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
        if semantic_content_measure(clean_text).effective_semantic_units < 2:
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
            text_tokens = _CLIP_LIB.tokenize([clean_text]).to(device)  # type: ignore

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
    import base64

    if not is_openai_available():
        logger.warning("Visual bad-take check unavailable; failing open", extra={"operation": "visual_bad_take"})
        return

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

        try:
            verdict = detect_bad_take(BAD_TAKES_MODEL, [
                {"role": "system", "content": "You strictly output 'GOOD' or 'BAD'."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ])
        except OpenAIProviderError:
            logger.warning("Visual bad-take check failed open", extra={"operation": "visual_bad_take"})
            continue
        if verdict is Verdict.BAD:
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

    if not is_openai_available():
        logger.warning("Boundary refiner unavailable; failing open", extra={"operation": "boundary_refinement"})
        return False

    import base64

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
            result = refine_boundaries(
                BOUNDARY_REFINER_MODEL, messages,
                temperature=0.1,
                max_completion_tokens=1000,
            )
        except OpenAIProviderError:
            logger.warning("Boundary refiner failed open", extra={"operation": "boundary_refinement"})
            continue
        head_bad = result.HEAD is Verdict.BAD
        mid_bad = result.MID is Verdict.BAD
        tail_bad = result.TAIL is Verdict.BAD

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
    return normalized_text(t).compact


def text_overlap_ratio(t1: str, t2: str) -> float:
    a = normalize_text(t1)
    b = normalize_text(t2)
    if not a or not b:
        return 0.0
    counts1 = Counter(comparison_units(a))
    counts2 = Counter(comparison_units(b))
    if not counts1 or not counts2:
        return 0.0
    inter = sum((counts1 & counts2).values())
    union = sum((counts1 | counts2).values())
    if union <= 0:
        return 0.0
    return inter / union


def text_overlap_shorter(t1: str, t2: str) -> float:
    a = normalize_text(t1)
    b = normalize_text(t2)
    if not a or not b:
        return 0.0
    counts1 = Counter(comparison_units(a))
    counts2 = Counter(comparison_units(b))
    if not counts1 or not counts2:
        return 0.0
    inter = sum((counts1 & counts2).values())
    denom = min(sum(counts1.values()), sum(counts2.values()))
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
            if int(c2.get("source_index", 0)) != int(c1.get("source_index", 0)):
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
    force_enabled: bool = False,
    execution_status: Optional[Dict[str, Any]] = None,
) -> bool:
    """Compare sibling takes, retaining all unless V2 produces a safe winner."""
    status = execution_status if execution_status is not None else {}
    status.update({"status": "not_requested", "groups_discovered": 0, "groups_evaluated": 0,
                   "winner_selected": 0, "abstained": 0, "low_confidence": 0,
                   "provider_error": 0})
    for clip_item in clips:
        clip_item["meta"]["take_judge_execution_status"] = "not_candidate"
    if not TAKE_JUDGE_ENABLED and not force_enabled:
        return False
    groups = [group[:TAKE_JUDGE_MAX_TAKES]
              for group in find_sibling_groups(clips)[:TAKE_JUDGE_MAX_GROUPS]]
    status["groups_discovered"] = len(groups)
    if not groups:
        status["status"] = "no_sibling_group"
        return False
    candidate_clips = [clip_item for group in groups for clip_item in group]
    if not is_openai_available():
        status["status"] = "provider_unavailable"
        for clip_item in candidate_clips:
            clip_item["meta"]["take_judge_execution_status"] = "provider_unavailable"
        logger.warning("Take Judge unavailable; failing open", extra={"operation": "take_judge_v2"})
        return False
    status["status"] = "abstained"
    used_any = False
    for group in groups:
        candidates = []
        samples = []
        for clip_item in group:
            if not clip_item["meta"].get("keep", True):
                continue
            sample = sample_candidate_frames(
                clip_item, TAKE_JUDGE_FRAMES, input_local, grab_frame_at_timestamp
            )
            samples.append(sample)
            candidates.append(TakeJudgeCandidate(
                candidate_id=str(clip_item["id"]),
                slot=str(clip_item.get("slot") or "STORY"),
                transcript=str(clip_item.get("text") or "").strip(),
                duration_sec=max(0.0, safe_float(clip_item.get("end")) - safe_float(clip_item.get("start"))),
                delivery=delivery_features(clip_item),
                frame_timestamps=sample.successful_frame_timestamps,
                image_count=len(sample.image_content),
            ))
        if len(candidates) < 2:
            status["abstained"] += 1
            for clip_item in group:
                clip_item["meta"]["take_judge_execution_status"] = "abstained"
            continue
        try:
            result = judge_takes_v2(
                TAKE_JUDGE_MODEL, candidates[0].slot, candidates, samples,
                temperature=0.1, max_completion_tokens=700,
            )
        except OpenAIProviderError:
            status["provider_error"] += 1
            for clip_item in group:
                clip_item["meta"]["take_judge_execution_status"] = "provider_error"
            logger.warning("Take Judge abstained", extra={
                "operation": "take_judge_v2", "group_size": len(candidates),
                "frame_count": sum(candidate.image_count for candidate in candidates),
                "validation": "failed", "abstention": True,
            })
            continue
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        status["groups_evaluated"] += 1
        if result.abstain:
            status["abstained"] += 1
            for clip_item in group:
                clip_item["meta"]["take_judge_execution_status"] = "abstained"
            continue
        if result.confidence < TAKE_JUDGE_V2_MIN_CONFIDENCE:
            status["low_confidence"] += 1
            for clip_item in group:
                clip_item["meta"]["take_judge_execution_status"] = "low_confidence"
            continue
        if result.winner_id not in candidate_ids:
            status["abstained"] += 1
            for clip_item in group:
                clip_item["meta"]["take_judge_execution_status"] = "abstained"
            continue
        scores = {score.candidate_id: score.overall_score for score in result.candidate_scores}
        for clip_item in group:
            candidate_id = clip_item["id"]
            if candidate_id not in scores:
                continue
            clip_item["meta"]["take_judge_score"] = scores[candidate_id]
            clip_item["meta"]["take_judge_verdict"] = (
                "WINNER" if candidate_id == result.winner_id else "LOSER"
            )
            clip_item["meta"]["take_judge_execution_status"] = (
                "candidate_winner" if candidate_id == result.winner_id else "candidate_loser"
            )
            if candidate_id != result.winner_id and clip_item["meta"].get("keep", True):
                clip_item["meta"]["keep"] = False
                clip_item["llm_reason"] = (clip_item.get("llm_reason") or "") + (
                    " | Removed by TakeJudgeAI (better take exists)."
                )
        used_any = True
        status["winner_selected"] += 1
    if status["winner_selected"]:
        status["status"] = "winner_selected"
    elif status["provider_error"]:
        status["status"] = "provider_error"
    elif status["low_confidence"]:
        status["status"] = "low_confidence"
    else:
        status["status"] = "abstained"
    return used_any


# =====================
# SCORE THRESHOLDS + DEDUPE
# =====================

def apply_min_score_rules(clips: List[Dict[str, Any]]) -> None:
    """
    Reglas mínimas de calidad:
      - Score semántico por slot (HOOK/CTA más exigentes)
      - Si hay visual_score muy bajo, se descarta el clip
    """
    for c in clips:
        slot = c.get("slot", "STORY")
        combined = safe_float(c.get("score", 0.0))
        sem = safe_float(c.get("semantic_score", 0.0))
        vis = safe_float(c.get("visual_score", 0.0))

        # 🔹 Umbral base por slot
        threshold = EDITDNA_MIN_CLIP_SCORE
        if slot == "HOOK":
            threshold = max(threshold, EDITDNA_HOOK_MIN_SCORE)
        elif slot == "CTA":
            threshold = max(threshold, EDITDNA_CTA_MIN_SCORE)

        effective_score = sem if sem > 0 else combined

        # 1) Si el score semántico es bajo → fuera
        if effective_score < threshold:
            c["meta"]["keep"] = False
            continue

        # 2) Si hay visión y se ve MUY flojo → fuera también
        if vis > 0.0 and vis < 0.58:
            c["meta"]["keep"] = False
            continue
            
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


def canonical_source_order(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return clips in input-file order, then source timestamp order."""
    return sorted(clips, key=lambda c: (
        int(c.get("source_index", 0)), safe_float(c.get("start", 0.0))
    ))


def add_source_metadata(
    clips: List[Dict[str, Any]],
    source_index: int,
    source_local: str,
    namespace_ids: bool = True,
) -> None:
    """Attach source identity before clean-cut validation and optionally namespace IDs."""
    prefix = f"source_{source_index:03d}:"
    id_map = {
        str(c.get("id", "")): (
            prefix + str(c.get("id", "")) if namespace_ids else str(c.get("id", ""))
        )
        for c in clips
    }
    for c in clips:
        original_clip_id = str(c.get("original_clip_id", c.get("id", "")))
        namespaced_clip_id = prefix + original_clip_id
        c["original_clip_id"] = original_clip_id
        c["namespaced_clip_id"] = namespaced_clip_id
        c["id"] = id_map[str(c.get("id", ""))]
        c["source_index"] = source_index
        c["source_local"] = source_local
        c.setdefault("source_start", safe_float(c.get("start")))
        c.setdefault("source_end", safe_float(c.get("end", c.get("start"))))
        for owner in (c, c.get("meta", {})):
            if isinstance(owner, dict) and isinstance(owner.get("chain_ids"), list):
                owner["chain_ids"] = [id_map.get(str(cid), prefix + str(cid)) for cid in owner["chain_ids"]]


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
            if int(c2.get("source_index", 0)) != int(c1.get("source_index", 0)):
                continue

            t2 = safe_float(c2.get("start", 0.0))
            if t2 - t1 > window_sec:
                break

            text1 = c1.get("text", "") or ""
            text2 = c2.get("text", "") or ""

            content1 = semantic_content_measure(text1)
            content2 = semantic_content_measure(text2)
            ratio = text_overlap_ratio(text1, text2)
            shorter_overlap = text_overlap_shorter(text1, text2)

            if ratio < min_overlap and shorter_overlap < 0.65:
                continue

            sem1 = safe_float(c1.get("semantic_score", 0.0))
            sem2 = safe_float(c2.get("semantic_score", 0.0))
            len1 = content1.effective_semantic_units
            len2 = content2.effective_semantic_units

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
            if int(c2.get("source_index", 0)) != int(c1.get("source_index", 0)):
                continue

            t2 = safe_float(c2.get("start", 0.0))
            if t2 - t1 > window_sec:
                break

            text2 = normalize_text(c2.get("text", ""))
            if not text1 or not text2:
                continue

            content1 = semantic_content_measure(text1)
            content2 = semantic_content_measure(text2)
            overlap = text_overlap_ratio(text1, text2)

            sem1 = safe_float(c1.get("semantic_score", 0.0))
            sem2 = safe_float(c2.get("semantic_score", 0.0))

            units1 = content1.effective_semantic_units
            units2 = content2.effective_semantic_units
            if overlap >= min_overlap and units2 > units1:
                if sem2 >= sem1:
                    c1["meta"]["keep"] = False
                    break

            if text1 in text2 and units2 - units1 > 3:
                c1["meta"]["keep"] = False
                break


def group_contiguous_blocks_by_slot(
    clips: List[Dict[str, Any]],
    slot: str,
    max_gap_sec: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    ordered = sorted(
        [c for c in clips if c.get("slot") == slot and c["meta"].get("keep", True)],
        key=lambda c: (
            int(c.get("source_index", 0)), safe_float(c.get("start", 0.0))
        ),
    )

    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_end: Optional[float] = None
    current_source: Optional[int] = None

    for c in ordered:
        start = safe_float(c.get("start", 0.0))
        end = safe_float(c.get("end", start))

        source_index = int(c.get("source_index", 0))
        if not current or source_index != current_source:
            if current:
                blocks.append(current)
            current = [c]
            current_source = source_index
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


def select_clean_cut_clip_ids(clips: List[Dict[str, Any]]) -> List[str]:
    """Return surviving analyzed clips in source-time order without mutation."""
    return [
        c["id"]
        for c in canonical_source_order(clips)
        if c.get("meta", {}).get("keep", True)
    ]


def _composer_hard_eligible(clip: Dict[str, Any]) -> bool:
    """Apply composer gates that semantic-threshold fallback may never relax."""
    meta = clip.get("meta", {})
    if not meta.get("keep", True):
        return False
    if filler_rule(clip.get("text", "")) is not None:
        return False
    if meta.get("boundary_diagnostic") in {
        "discarded_invalid_microfragment", "discarded_duplicate_residual_text",
    }:
        return False
    # Visual quality is a ranking signal. A low embedding score is not enough
    # to delete intelligible, valid speech from an editable draft.
    return True


def build_composer(clips: List[Dict[str, Any]], mode: str = "human") -> Dict[str, Any]:
    mode = (mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

    # Composer selection is intentionally isolated from Clean Cut state. Legacy
    # scoring/dedupe functions may mutate meta.keep, so operate on deep copies
    # and retain every valid original clip for alternates/restore.
    keepable = [
        copy.deepcopy(c)
        for c in clips
        if _composer_hard_eligible(c)
    ]
    usable = [
        c for c in keepable
        if safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
    ]
    # A valid short English/Spanish transcript must not become a selection
    # error merely because deterministic scoring cannot reach a model-tuned
    # threshold. Production meta and dependent fragments were already removed.
    short_transcript_fallback = not usable and bool(keepable)
    if short_transcript_fallback:
        usable = keepable
        for c in usable:
            c.setdefault("meta", {})["composer_fallback"] = "keepable_short_transcript"

    for c in usable:
        sem = safe_float(c.get("semantic_score", 0.0))
        vis = safe_float(c.get("visual_score", 0.0))
        if vis > 0.0:
            combined = (1.0 - W_VISION) * sem + W_VISION * vis
        else:
            combined = sem
        c["score"] = combined
        c["meta"]["score"] = combined

    if not short_transcript_fallback:
        apply_min_score_rules(usable)

    usable = [c for c in usable if c["meta"].get("keep", True)]
    usable = canonical_source_order(usable)

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
            fallback_rule = c.get("meta", {}).get("fallback_slot_rule")
            keepable_unclassified = slot == "OTHER" and fallback_rule == "unclassified_product_context"
            if slot not in {"STORY", "HOOK", "CTA"} and not keepable_unclassified:
                c["meta"]["keep"] = False
        usable = [c for c in usable if c["meta"].get("keep", True)]

    if not usable:
        usable = [
            c
            for c in clips
            if c["meta"].get("keep", True)
            and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC
        ]
        usable = canonical_source_order(usable)

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

    selected_cta_ids = set(cta_final_ids)

    # Slots are semantic functions, not mandatory timeline positions. Preserve
    # source-time order among selected clips instead of moving CTA clips to the
    # end or forcing a linear HOOK→...→CTA funnel sequence. Non-selected CTA
    # alternatives are excluded so the chosen CTA block/id stays consistent with
    # the rendered timeline.
    timeline: List[Dict[str, Any]] = [
        c for c in usable if c.get("slot") != "CTA" or c["id"] in selected_cta_ids
    ]
    used_ids: List[str] = [c["id"] for c in timeline]

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
    input_local: Union[str, List[str]], session_dir: str,
    clips: List[Dict[str, Any]], used_clip_ids: List[str],
) -> str:
    """Normalize, cut, and concatenate selected clips from one or more sources."""
    if not used_clip_ids:
        raise RuntimeError("render_funnel_video: no used_clip_ids provided")
    sources = [input_local] if isinstance(input_local, str) else list(input_local)
    if not sources:
        raise RuntimeError("render_funnel_video: no input sources provided")
    matches: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in clips:
        matches.setdefault(candidate["id"], []).append(candidate)
    missing = list(dict.fromkeys(cid for cid in used_clip_ids if not matches.get(cid)))
    ambiguous = list(dict.fromkeys(cid for cid in used_clip_ids if len(matches.get(cid, [])) > 1))
    if missing or ambiguous:
        details = []
        if missing:
            details.append(f"missing selected clip IDs: {missing!r}")
        if ambiguous:
            details.append(f"non-unique selected clip IDs: {ambiguous!r}")
        raise MissingSelectedClipsError("; ".join(details))
    # Selection is set-like in the existing composer contract. Preserve first
    # occurrence order while preventing accidental duplicate rendering.
    unique_ids = list(dict.fromkeys(used_clip_ids))
    selected = [matches[cid][0] for cid in unique_ids]
    indices = {int(c.get("source_index", 0)) for c in selected}
    if any(i < 0 or i >= len(sources) for i in indices):
        raise RuntimeError("render_funnel_video: clip has invalid source_index")
    audio_by_source = {i: has_audio_stream(sources[i]) for i in indices}
    has_audio = any(audio_by_source.values())

    filters, videos, audios = [], [], []
    for c in selected:
        source_index = int(c.get("source_index", 0))
        start = max(0.0, safe_float(c.get("start")) + HEAD_TRIM_SEC)
        end = max(start, safe_float(c.get("end")) - TAIL_TRIM_SEC)
        if end <= start:
            continue
        number = len(videos); vlabel = f"v{number}"
        filters.append(
            f"[{source_index}:v]trim=start={start:.3f}:end={end:.3f},"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={OUTPUT_FPS},format=yuv420p,setpts=PTS-STARTPTS[{vlabel}]"
        )
        videos.append(f"[{vlabel}]")
        if has_audio:
            alabel = f"a{number}"
            if audio_by_source[source_index]:
                audio_input = f"[{source_index}:a]atrim=start={start:.3f}:end={end:.3f}"
            else:
                audio_input = f"anullsrc=r=48000:cl=stereo,atrim=duration={end-start:.3f}"
            filters.append(
                f"{audio_input},aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS[{alabel}]"
            )
            audios.append(f"[{alabel}]")
    count = len(videos)
    if not count:
        raise RuntimeError("render_funnel_video: no valid clips after trimming")
    filters.append(f"{''.join(videos)}concat=n={count}:v=1:a=0[vout]")
    if has_audio:
        filters.append(f"{''.join(audios)}concat=n={count}:v=0:a=1[aout]")
    cmd=[FFMPEG_BIN,"-y"]
    for source in sources: cmd += ["-i",source]
    cmd += ["-filter_complex","; ".join(filters),"-map","[vout]"]
    cmd += ["-map","[aout]","-c:a","aac"] if has_audio else ["-an"]
    output=os.path.join(session_dir,"final.mp4")
    cmd += ["-c:v","libx264","-pix_fmt","yuv420p","-r",str(OUTPUT_FPS),"-movflags","+faststart",output]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if proc.returncode != 0:
        logger.error("ffmpeg failed:\nSTDOUT:\n%s\nSTDERR:\n%s",proc.stdout,proc.stderr)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    return output


# =====================
# MAIN ENTRYPOINT
# =====================

def run_pipeline(session_id: str, files: Optional[List[str]] = None,
                 file_urls: Optional[List[str]] = None, mode: str = "human",
                 progress: Optional[Callable[[str, int, str], None]] = None,
                 check_canceled: Optional[Callable[[], None]] = None,
                 local_files: Optional[List[str]] = None,
                 output_key: Optional[str] = None,
                 render_output: bool = True,
                 persist_result_json: bool = True,
                 retain_local_files: bool = True,
                 use_semantic_v2: bool = False,
                 use_take_judge_v2: bool = False) -> Dict[str, Any]:
    """Analyze every source once, compose once, and render one output."""
    logger.info("run_pipeline session_id=%s mode=%s", session_id, mode)
    effective_files = local_files or (files if files and isinstance(files, list) else file_urls)
    if not effective_files or not isinstance(effective_files, list):
        raise ValueError("run_pipeline: 'files' or 'file_urls' must be a list with at least 1 URL")
    mode=(mode or "human").lower()
    if mode not in ("human","clean","blooper"): mode="human"
    session_dir=ensure_session_dir(session_id)
    retain_local_result = False
    report = progress or (lambda _stage, _percent, _message: None)
    check = check_canceled or (lambda: None)
    try:
        if local_files:
            local_sources = list(local_files)
        elif len(effective_files)==1:
            local_sources=[os.path.join(session_dir,"input.mp4")]
        else:
            local_sources=[os.path.join(session_dir,f"input_{i:03d}.mp4") for i in range(len(effective_files))]
        for i,(url,path) in enumerate(zip(effective_files,local_sources)):
            if local_files:
                continue
            check()
            report("downloading", 5 + (15 * i // len(local_sources)), f"Downloading source {i + 1} of {len(local_sources)}")
            try: download_to_local(url,path)
            except Exception as exc:
                raise RuntimeError(f"Failed to download input source_index={i} (URL position {i+1})") from exc
            check()
            report("downloading", 5 + (15 * (i + 1) // len(local_sources)), f"Downloaded source {i + 1} of {len(local_sources)}")

        clips=[]; durations=[]; clean_cut_discard_diagnostics=[]
        llm_used=vision_used=bad_takes_used=boundaries_refined=take_judge_used=False
        take_judge_execution = {"requested": bool(TAKE_JUDGE_ENABLED or use_take_judge_v2), "sources": []}
        for i,path in enumerate(local_sources):
          check()
          report("analyzing", 20 + (35 * i // len(local_sources)), f"Analyzing source {i + 1} of {len(local_sources)}")
          try:
            durations.append(probe_duration(path))
            raw_clips = sentence_boundary_micro_cuts(run_asr(path))
            # Source identity must exist before merge/boundary validation because
            # discarded clips never reach later enrichment stages.
            add_source_metadata(
                raw_clips,
                i,
                path,
                namespace_ids=len(local_sources) > 1,
            )
            if "discarded_diagnostics" in inspect.signature(merge_incomplete_phrases).parameters:
                current=merge_incomplete_phrases(
                    raw_clips,
                    discarded_diagnostics=clean_cut_discard_diagnostics,
                )
            else:
                current=merge_incomplete_phrases(raw_clips)
            semantic_used = (enrich_clips_semantic(current, force_v2=True)
                             if use_semantic_v2 else enrich_clips_semantic(current))
            llm_used = semantic_used or llm_used
            current=dedupe_clips(current)
            vision_used=run_visual_pass(path,session_dir,current) or vision_used
            if VISION_ENABLED and BAD_TAKES_ENABLED:
                reject_visual_bad_takes(current,session_dir,path); bad_takes_used=True
            if VISION_ENABLED and BOUNDARY_REFINER_ENABLED:
                boundaries_refined=refine_clip_boundaries_with_vision(input_local=path,session_dir=session_dir,clips=current) or boundaries_refined
            if TAKE_JUDGE_ENABLED or use_take_judge_v2:
                source_take_judge_status = {"source_index": i}
                take_judge_used = (run_take_judge(clips=current, session_dir=session_dir,
                                                  input_local=path, force_enabled=True,
                                                  execution_status=source_take_judge_status)
                                   if use_take_judge_v2 else
                                   run_take_judge(clips=current, session_dir=session_dir,
                                                  input_local=path,
                                                  execution_status=source_take_judge_status)) or take_judge_used
                take_judge_execution["sources"].append(source_take_judge_status)
            else:
                for clip_item in current:
                    clip_item["meta"]["take_judge_execution_status"] = "not_candidate"
            for c in current:
                c["source_start"],c["source_end"]=safe_float(c.get("start")),safe_float(c.get("end"))
            clips.extend(current)
            check()
            report("analyzing", 20 + (35 * (i + 1) // len(local_sources)), f"Analyzed source {i + 1} of {len(local_sources)}")
          except JobCanceledError:
            raise
          except Exception as exc:
            raise RuntimeError(f"Failed to analyze input source_index={i} (URL position {i+1})") from exc
        check()
        report("selecting", 60, "Selecting clips")
        slots=build_slots_dict(clips)
        clean_ids=select_clean_cut_clip_ids(clips) if mode=="clean" else []
        composer=build_composer(copy.deepcopy(clips) if mode=="clean" else clips,mode=mode)
        used=clean_ids if mode=="clean" else composer.get("used_clip_ids",[])
        if mode!="clean" and len(local_sources)>1:
            selected=set(used)
            used=[c["id"] for c in canonical_source_order(clips) if c["id"] in selected]
            known_ids=set(used)
            used.extend(dict.fromkeys(cid for cid in composer.get("used_clip_ids",[]) if cid not in known_ids))
            composer["used_clip_ids"]=used
        input_local=local_sources[0]; clean_rendered=False
        if used and render_output:
            check()
            report("rendering", 65, "Rendering video")
            final=render_funnel_video(input_local if len(local_sources)==1 else local_sources,session_dir,clips,used)
            report("rendering", 90, "Video rendered")
            check()
            clean_rendered=mode=="clean"
        elif mode in ("human", "blooper") and render_output:
            raise SelectionError(f"No clips were selected for requested mode '{mode}'")
        else: final=input_local if render_output else None
        check()
        if render_output:
            report("uploading", 92, "Uploading rendered video")
        output_url=upload_to_s3(final,S3_BUCKET,output_key or f"{S3_PREFIX}/{session_id}-final.mp4") if S3_BUCKET and render_output else None
        if render_output:
            report("uploading", 98, "Upload complete")
        if S3_BUCKET and render_output and not output_url:
            raise UploadError("Rendered output upload returned no output URL")
        clean_cut_discard_diagnostics = sanitize_clean_cut_discard_diagnostics(
            clean_cut_discard_diagnostics
        )
        result={"ok":True,"session_id":session_id,"input_local":input_local,
      "input_files_local":local_sources,"input_file_count":len(local_sources),
      "processed_source_indices":list(range(len(local_sources))),"duration_sec":sum(durations),
      "input_durations_sec":durations,"clips":clips,"slots":slots,"composer":composer,
      "composer_human":pretty_print_composer(clips,composer),"output_video_local":final,
      "output_video_url":output_url,"clean_cut_used_clip_ids":clean_ids,
      "clean_cut_discard_diagnostics":clean_cut_discard_diagnostics,
      "clean_cut_output_video_local":final if clean_rendered else None,
      "clean_cut_output_video_url":output_url if clean_rendered else None,"asr":True,"semantic":True,
      "vision":vision_used,"llm_used":llm_used,"take_judge_used":take_judge_used,
      "take_judge_execution":take_judge_execution,
      "provider_usage_instrumentation":{"available":False,"reason":"Provider token usage is not instrumented"},
      "estimated_cost_instrumentation":{"available":False,"reason":"Provider cost is not instrumented"},
      "bad_takes_used":bad_takes_used,"boundaries_refined":boundaries_refined,"composer_mode":mode}
        if output_url or not retain_local_files:
            # Uploaded jobs and explicitly non-retaining internal jobs return no
            # ephemeral paths; their temporary storage is removed below.
            result.pop("input_local")
            result.pop("input_files_local")
            result.pop("output_video_local")
            result.pop("clean_cut_output_video_local")
            for analyzed_clip in result["clips"]:
                analyzed_clip.pop("source_local", None)
        else:
            # No-S3 operation is an existing result contract used by direct
            # callers. Transfer ownership of the job directory to that caller so
            # its returned local paths remain usable.
            retain_local_result = retain_local_files
        result["output_json_s3_uri"] = save_result_json_to_s3(result) if persist_result_json else None
        return result
    finally:
        if not retain_local_result:
            try:
                shutil.rmtree(session_dir)
            except FileNotFoundError:
                pass
            except Exception:
                # Cleanup is best effort and must never replace the active failure.
                logger.warning("Failed to clean up job temporary directory", exc_info=True)
