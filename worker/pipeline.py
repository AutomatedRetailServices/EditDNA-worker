import os
import io
import json
import math
import uuid
import time
import shutil
import logging
import difflib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from faster_whisper import WhisperModel
from openai import OpenAI

logger = logging.getLogger("editdna.pipeline")
logger.setLevel(logging.INFO)

# ========= CONFIG =========

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "medium")

TMP_ROOT = Path(os.environ.get("EDITDNA_TMP_ROOT", "/tmp/TMP/editdna"))

COMPOSER_MIN_SEMANTIC = float(os.environ.get("COMPOSER_MIN_SEMANTIC", "0.75"))
COMPOSER_DUP_SIM_THRESHOLD = float(os.environ.get("COMPOSER_DUP_SIM_THRESHOLD", "0.80"))

# Slot caps (máximo por tipo; tú pediste “al menos 7 cada uno”)
COMPOSER_MAX_HOOK = int(os.environ.get("COMPOSER_MAX_HOOK", "7"))
COMPOSER_MAX_STORY = int(os.environ.get("COMPOSER_MAX_STORY", "7"))
COMPOSER_MAX_PROBLEM = int(os.environ.get("COMPOSER_MAX_PROBLEM", "7"))
COMPOSER_MAX_BENEFITS = int(os.environ.get("COMPOSER_MAX_BENEFITS", "7"))
COMPOSER_MAX_FEATURES = int(os.environ.get("COMPOSER_MAX_FEATURES", "7"))
COMPOSER_MAX_PROOF = int(os.environ.get("COMPOSER_MAX_PROOF", "7"))
COMPOSER_MAX_CTA = int(os.environ.get("COMPOSER_MAX_CTA", "3"))

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "editdna/outputs")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ========= UTILS =========


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path) -> None:
    logger.info(f"Downloading file from {url} -> {dst}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        shutil.copyfileobj(r.raw, f)


def load_whisper_model() -> WhisperModel:
    """
    Intenta usar GPU (cuda + float16). Si falla, cae a CPU.
    """
    logger.info("Loading Whisper model %s", WHISPER_MODEL_NAME)
    try:
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper loaded on GPU (cuda, float16)")
        return model
    except Exception:
        logger.exception("Failed to load Whisper on GPU, falling back to CPU")
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper loaded on CPU (int8)")
        return model


# ========= ASR =========


def run_asr(input_path: Path) -> List[Dict[str, Any]]:
    """
    Corre Whisper con word_timestamps=True y devuelve segmentos con palabras:

    [
      {
        "id": "ASR0000",
        "start": float,
        "end": float,
        "text": str,
        "words": [
          {"word": str, "start": float, "end": float},
          ...
        ]
      },
      ...
    ]
    """
    model = load_whisper_model()
    logger.info("Running Whisper ASR on %s", input_path)

    segments, info = model.transcribe(
        str(input_path),
        beam_size=5,
        word_timestamps=True,
    )

    asr_segments: List[Dict[str, Any]] = []
    idx = 0
    for seg in segments:
        words_list: List[Dict[str, Any]] = []
        if seg.words:
            for w in seg.words:
                words_list.append(
                    {
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                    }
                )

        asr_segments.append(
            {
                "id": f"ASR{idx:04d}",
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": words_list,
            }
        )
        idx += 1

    total_dur = asr_segments[-1]["end"] if asr_segments else 0.0
    logger.info("ASR produced %d segments, duration ~%.2fs", len(asr_segments), total_dur)
    return asr_segments


# ========= SENTENCE-BOUNDARY MICRO CUTS =========


def sentence_boundary_micro_cuts(asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Micro-cuts a nivel de oración usando las palabras con timestamps de Whisper.

    Espera que cada segmento en `asr_segments` tenga:
      {
        "id": str,
        "start": float,
        "end": float,
        "text": str,
        "words": [
            {"word": str, "start": float, "end": float},
            ...
        ]
      }

    Devuelve una lista de clips:
      {
        "id": str,
        "slot": "HOOK",           # slot placeholder; luego lo sobreescribe el LLM
        "start": float,
        "end": float,
        "score": 0.0,             # luego se rellena
        "semantic_score": 0.0,
        "visual_score": 1.0,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [str],
        "text": str,
        "llm_reason": "",
        "visual_flags": {"scene_jump": False, "motion_jump": False},
        "meta": {
            "slot": "HOOK",
            "semantic_score": 0.0,
            "visual_score": 1.0,
            "score": 0.0,
            "chain_ids": [],
            "keep": True,
        }
      }
    """
    MAX_SENT_DURATION = 10.0  # no queremos frases de 20s
    PAUSE_SPLIT_SEC = 0.6     # silencio / gap entre palabras

    clips: List[Dict[str, Any]] = []
    global_idx = 0

    for seg in asr_segments:
        seg_id = seg.get("id", f"seg{global_idx}")
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        words = seg.get("words") or []

        # Si no hay palabras, cae a un solo clip del segmento completo
        if not words:
            clip_id = f"{seg_id}_c{global_idx}"
            clips.append(
                {
                    "id": clip_id,
                    "slot": "HOOK",
                    "start": seg_start,
                    "end": seg_end,
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "visual_score": 1.0,
                    "face_q": 1.0,
                    "scene_q": 1.0,
                    "vtx_sim": 0.0,
                    "chain_ids": [clip_id],
                    "text": seg.get("text", "").strip(),
                    "llm_reason": "",
                    "visual_flags": {"scene_jump": False, "motion_jump": False},
                    "meta": {
                        "slot": "HOOK",
                        "semantic_score": 0.0,
                        "visual_score": 1.0,
                        "score": 0.0,
                        "chain_ids": [],
                        "keep": True,
                    },
                }
            )
            global_idx += 1
            continue

        buffer_tokens: List[str] = []
        sent_start: Optional[float] = None
        last_end: Optional[float] = None

        def flush_sentence():
            nonlocal buffer_tokens, sent_start, global_idx, last_end
            if not buffer_tokens or sent_start is None:
                return
            text = " ".join(buffer_tokens).strip()
            if not text:
                buffer_tokens = []
                sent_start = None
                return

            end_time = float(last_end) if last_end is not None else sent_start
            clip_id = f"{seg_id}_c{global_idx}"
            clip = {
                "id": clip_id,
                "slot": "HOOK",  # placeholder
                "start": float(sent_start),
                "end": float(end_time),
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
                    "slot": "HOOK",
                    "semantic_score": 0.0,
                    "visual_score": 1.0,
                    "score": 0.0,
                    "chain_ids": [],
                    "keep": True,
                },
            }
            clips.append(clip)
            global_idx += 1
            buffer_tokens = []
            sent_start = None

        for w in words:
            token = str(w.get("word", "")).strip()
            w_start = float(w.get("start", seg_start))
            w_end = float(w.get("end", w_start))

            if not token:
                continue

            if sent_start is None:
                sent_start = w_start

            buffer_tokens.append(token)

            end_sentence = False

            # 1) Puntuación fuerte
            if token.endswith((".", "?", "!", ";", ":")):
                end_sentence = True

            # 2) Pausa larga
            if last_end is not None and (w_start - last_end) >= PAUSE_SPLIT_SEC:
                end_sentence = True

            # 3) Oración demasiado larga
            if sent_start is not None and (w_end - sent_start) >= MAX_SENT_DURATION:
                end_sentence = True

            last_end = w_end

            if end_sentence:
                flush_sentence()

        # Flush final
        flush_sentence()

    logger.info("Sentence-boundary micro-cuts produced %d clips", len(clips))
    return clips


# ========= LLM SCORING / CLASSIFICATION =========

LLM_SYSTEM_PROMPT = """
You are an editing assistant for TikTok UGC ads.

Goal:
- For each SHORT sentence/clip, decide:
  - slot: one of [HOOK, STORY, PROBLEM, BENEFITS, FEATURES, PROOF, CTA]
  - keep: true if it's useful content, false if it's filler / redo / meta
  - semantic_score: 0.0 - 1.0 (relevance & usefulness for the ad)
  - reason: brief explanation.

Definitions (short version):
- HOOK: grabs attention, pattern break, strong first line.
- STORY: personal story or context narrative.
- PROBLEM: describes pain, frustration, symptoms, "does your X do Y?".
- BENEFITS: outcomes for the user (feel, result, transformation).
- FEATURES: ingredients, specs, "2 a day", "pineapple flavored", etc.
- PROOF: testimonial, credibility, social proof.
- CTA: call to action (click, buy, link in bio, etc).

Filler / redo examples => keep=false:
- "wait", "let me start again", "is that good?", "did I say that right?",
  "okay, okay", "I don't know how to do it like that" etc.

Return STRICT JSON ONLY.
"""


def classify_clip_with_llm(text: str) -> Dict[str, Any]:
    """
    Llama a OpenAI para clasificar un clip.
    Devuelve: {"slot": str, "keep": bool, "semantic_score": float, "reason": str}
    """
    text = text.strip()
    if not text:
        return {
            "slot": "HOOK",
            "keep": False,
            "semantic_score": 0.0,
            "reason": "Empty text",
        }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this line for a TikTok UGC ad:\n\"{text}\"",
                },
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        logger.exception("LLM classification failed, defaulting")
        return {
            "slot": "HOOK",
            "keep": True,
            "semantic_score": 0.5,
            "reason": f"Fallback due to error: {e}",
        }

    slot = str(data.get("slot", "HOOK")).upper().strip()
    if slot not in ["HOOK", "STORY", "PROBLEM", "BENEFITS", "FEATURES", "PROOF", "CTA"]:
        slot = "HOOK"

    keep = bool(data.get("keep", True))
    try:
        semantic_score = float(data.get("semantic_score", 0.5))
    except Exception:
        semantic_score = 0.5

    reason = str(data.get("reason", "")).strip()

    return {
        "slot": slot,
        "keep": keep,
        "semantic_score": max(0.0, min(1.0, semantic_score)),
        "reason": reason,
    }


def classify_all_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enriquecer cada clip con slot, keep, semantic_score, llm_reason y meta.
    """
    logger.info("Classifying %d clips with LLM", len(clips))
    for clip in clips:
        info = classify_clip_with_llm(clip["text"])
        slot = info["slot"]
        keep = info["keep"]
        sem = info["semantic_score"]
        reason = info["reason"]

        clip["slot"] = slot
        clip["semantic_score"] = sem
        clip["score"] = sem  # usamos semantic_score como score base
        clip["llm_reason"] = reason

        meta = clip.get("meta") or {}
        meta["slot"] = slot
        meta["semantic_score"] = sem
        meta["score"] = sem
        meta["keep"] = keep
        clip["meta"] = meta

    return clips


# ========= COMPOSER V2 (dedupe + best take + funnel order) =========


def normalized_text(s: str) -> str:
    s = s.lower().strip()
    return " ".join(s.split())


def text_similarity(a: str, b: str) -> float:
    a_n = normalized_text(a)
    b_n = normalized_text(b)
    if not a_n or not b_n:
        return 0.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def dedupe_and_select_best(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupe por similitud de texto:
      - Si dos frases son casi iguales (> COMPOSER_DUP_SIM_THRESHOLD),
        nos quedamos con la que tenga mayor semantic_score.
    """
    selected: List[Dict[str, Any]] = []
    for clip in clips:
        text = clip.get("text", "")
        best_for_this = clip
        skip = False

        for i, existing in enumerate(selected):
            sim = text_similarity(text, existing.get("text", ""))
            if sim >= COMPOSER_DUP_SIM_THRESHOLD:
                # Ya hay algo casi igual → nos quedamos con el mejor
                if clip.get("semantic_score", 0.0) > existing.get("semantic_score", 0.0):
                    selected[i] = clip
                skip = True
                break

        if not skip:
            selected.append(best_for_this)

    return selected


def composer_v2(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Composer V2:
      - filtra por keep=True y semantic_score >= COMPOSER_MIN_SEMANTIC
      - dedupe por texto
      - organiza en slots con caps suaves
      - orden tipo funnel: HOOK → PROBLEM → STORY → BENEFITS → FEATURES → PROOF → CTA
      - mantiene orden cronológico dentro de cada grupo
      - CTA (si existe) se fuerza al final
    """
    # 1) Filtrado por keep + score
    usable = [
        c
        for c in clips
        if c.get("meta", {}).get("keep", True)
        and c.get("semantic_score", 0.0) >= COMPOSER_MIN_SEMANTIC
    ]

    # 2) Dedupe
    usable = dedupe_and_select_best(usable)

    # 3) Agrupar por slot
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "STORY": [],
        "PROBLEM": [],
        "BENEFITS": [],
        "FEATURES": [],
        "PROOF": [],
        "CTA": [],
    }

    for c in usable:
        slot = c.get("slot", "HOOK").upper()
        if slot not in slots:
            slot = "HOOK"
        slots[slot].append(c)

    # 4) Sort por (semantic_score DESC, start ASC) dentro de cada slot
    for key, arr in slots.items():
        arr.sort(
            key=lambda c: (
                -float(c.get("semantic_score", 0.0)),
                float(c.get("start", 0.0)),
            )
        )

    # 5) Caps por slot
    slots["HOOK"] = slots["HOOK"][:COMPOSER_MAX_HOOK]
    slots["STORY"] = slots["STORY"][:COMPOSER_MAX_STORY]
    slots["PROBLEM"] = slots["PROBLEM"][:COMPOSER_MAX_PROBLEM]
    slots["BENEFITS"] = slots["BENEFITS"][:COMPOSER_MAX_BENEFITS]
    slots["FEATURES"] = slots["FEATURES"][:COMPOSER_MAX_FEATURES]
    slots["PROOF"] = slots["PROOF"][:COMPOSER_MAX_PROOF]
    slots["CTA"] = slots["CTA"][:COMPOSER_MAX_CTA]

    # 6) Funnel order (pero manteniendo orden cronológico dentro del grupo)
    def sort_by_start(arr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(arr, key=lambda c: float(c.get("start", 0.0)))

    hook_clips = sort_by_start(slots["HOOK"])
    problem_clips = sort_by_start(slots["PROBLEM"])
    story_clips = sort_by_start(slots["STORY"])
    benefit_clips = sort_by_start(slots["BENEFITS"])
    feature_clips = sort_by_start(slots["FEATURES"])
    proof_clips = sort_by_start(slots["PROOF"])
    cta_clips = sort_by_start(slots["CTA"])

    # Orden final:
    # 1) HOOKs
    # 2) PROBLEM
    # 3) STORY
    # 4) BENEFITS
    # 5) FEATURES
    # 6) PROOF
    # 7) CTA (al final sí o sí)
    ordered_clips: List[Dict[str, Any]] = []
    ordered_clips.extend(hook_clips)
    ordered_clips.extend(problem_clips)
    ordered_clips.extend(story_clips)
    ordered_clips.extend(benefit_clips)
    ordered_clips.extend(feature_clips)
    ordered_clips.extend(proof_clips)
    ordered_clips.extend(cta_clips)

    used_ids = [c["id"] for c in ordered_clips]

    # Para el resumen humano
    def slot_lines(name: str, arr: List[Dict[str, Any]]) -> str:
        if not arr:
            return "  (none)"
        lines = []
        for c in arr:
            lines.append(
                f'  [{c["id"]}] score={c.get("semantic_score", 0.0):.2f} → "{c.get("text","")}"'
            )
        return "\n".join(lines)

    composer_human_parts = [
        "===== EDITDNA FUNNEL COMPOSER =====",
        "HOOK:",
        slot_lines("HOOK", hook_clips),
        "STORY:",
        slot_lines("STORY", story_clips),
        "PROBLEM:",
        slot_lines("PROBLEM", problem_clips),
        "BENEFITS:",
        slot_lines("BENEFITS", benefit_clips),
        "FEATURES:",
        slot_lines("FEATURES", feature_clips),
        "PROOF:",
        slot_lines("PROOF", proof_clips),
        "CTA:",
        slot_lines("CTA", cta_clips),
        "",
        "FINAL ORDER TIMELINE:",
    ]

    for i, c in enumerate(ordered_clips, start=1):
        composer_human_parts.append(f'{i}) {c["id"]} → "{c.get("text","")}"')

    composer_human_parts.append("\n=====================================")

    composer_human = "\n".join(composer_human_parts)

    composer_dict = {
        "hook_id": hook_clips[0]["id"] if hook_clips else None,
        "story_ids": [c["id"] for c in story_clips],
        "problem_ids": [c["id"] for c in problem_clips],
        "benefit_ids": [c["id"] for c in benefit_clips],
        "feature_ids": [c["id"] for c in feature_clips],
        "proof_ids": [c["id"] for c in proof_clips],
        "cta_id": cta_clips[0]["id"] if cta_clips else None,
        "used_clip_ids": used_ids,
        "min_score": COMPOSER_MIN_SEMANTIC,
    }

    return {
        "ordered_clips": ordered_clips,
        "slots": slots,
        "composer": composer_dict,
        "composer_human": composer_human,
    }


# ========= VIDEO RENDER (NEW: concat demuxer) =========


def render_funnel_video(
    input_local: Path,
    session_dir: Path,
    clips: List[Dict[str, Any]],
    used_clip_ids: List[str],
) -> Path:
    """
    NUEVA VERSIÓN ROBUSTA (SIN filter_complex):

    1) Para cada clip seleccionado:
       - ffmpeg -ss start -to end -i input.mp4 -c copy segment_i.mp4
    2) Creamos un concat_list.txt con:
       file 'segment_0.mp4'
       file 'segment_1.mp4'
       ...
    3) ffmpeg -f concat -safe 0 -i concat_list.txt -c copy final.mp4

    Esto mantiene audio + video juntos y evita los errores de media type mismatch.
    """
    if not used_clip_ids:
        raise ValueError("No clips selected for final render")

    # Map ID -> clip
    clip_index: Dict[str, Dict[str, Any]] = {c["id"]: c for c in clips}
    selected_clips: List[Dict[str, Any]] = []
    for cid in used_clip_ids:
        if cid in clip_index:
            selected_clips.append(clip_index[cid])

    if not selected_clips:
        raise ValueError("selected_clips empty after mapping used_clip_ids")

    # 1) Extraer cada segmento a un mp4 temporal
    segment_paths: List[Path] = []
    for i, c in enumerate(selected_clips):
        start = float(c["start"])
        end = float(c["end"])
        if end <= start:
            continue

        seg_path = session_dir / f"segment_{i:03d}.mp4"
        # Corte rápido con copy (puede no ser frame-perfect, pero mantiene A/V sync simple)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(input_local),
            "-c",
            "copy",
            str(seg_path),
        ]
        logger.info("Extracting segment %s: start=%.3f end=%.3f", seg_path, start, end)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            logger.error(
                "ffmpeg segment extraction failed for %s:\nSTDOUT:\n%s\nSTDERR:\n%s",
                seg_path,
                proc.stdout,
                proc.stderr,
            )
            raise RuntimeError(f"ffmpeg segment extraction failed with code {proc.returncode}")
        segment_paths.append(seg_path)

    if not segment_paths:
        raise ValueError("No valid segments created for final render")

    # 2) Crear concat_list.txt
    list_path = session_dir / "concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            # Rutas absolutas → usamos -safe 0
            f.write(f"file '{p.as_posix()}'\n")

    # 3) Concatenar todos los segmentos
    out_path = session_dir / "final.mp4"
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    logger.info("Concatenating %d segments into %s", len(segment_paths), out_path)
    proc = subprocess.run(
        cmd_concat,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        logger.error(
            "ffmpeg concat failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
            proc.stdout,
            proc.stderr,
        )
        raise RuntimeError(f"ffmpeg concat failed with code {proc.returncode}")

    return out_path


# ========= S3 UPLOAD =========


def upload_to_s3(local_path: Path) -> Optional[str]:
    if not S3_BUCKET:
        return None
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed; skipping S3 upload")
        return None

    s3 = boto3.client("s3")
    key = f"{S3_PREFIX}/{local_path.name}"
    logger.info("Uploading %s to s3://%s/%s", local_path, S3_BUCKET, key)
    s3.upload_file(str(local_path), S3_BUCKET, key)
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=7 * 24 * 3600,
    )
    return url


# ========= MAIN PIPELINE =========


def run_pipeline(session_id: str, file_urls: List[str]) -> Dict[str, Any]:
    """
    Entry principal del worker.

    - session_id: string único (viene de web/api)
    - file_urls: lista de URLs (por ahora usamos el primero)
    """
    t0 = time.time()
    ensure_dir(TMP_ROOT)

    session_dir = TMP_ROOT / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    ensure_dir(session_dir)

    # 1) Descargar input
    input_url = file_urls[0]
    input_local = session_dir / "input.mp4"
    download_file(input_url, input_local)

    # 2) ASR
    asr_segments = run_asr(input_local)
    duration_sec = asr_segments[-1]["end"] if asr_segments else 0.0

    # 3) Micro-cuts (sentence-boundary)
    clips = sentence_boundary_micro_cuts(asr_segments)

    # 4) LLM classification
    clips = classify_all_clips(clips)

    # 5) Composer V2 (dedupe + best-take + funnel order)
    comp = composer_v2(clips)
    ordered_clips = comp["ordered_clips"]
    slots = comp["slots"]
    composer_meta = comp["composer"]
    composer_human = comp["composer_human"]
    used_clip_ids = composer_meta["used_clip_ids"]

    # 6) Render final video
    final_path = render_funnel_video(input_local, session_dir, clips, used_clip_ids)
    s3_url = upload_to_s3(final_path)

    elapsed = time.time() - t0
    logger.info("Pipeline finished in %.2fs", elapsed)

    result = {
        "session_id": session_id,
        "input_local": str(input_local),
        "duration_sec": float(duration_sec),
        "clips": clips,
        "slots": {
            slot: arr for slot, arr in slots.items()
        },
        "composer": composer_meta,
        "composer_human": composer_human,
        "output_video_local": str(final_path),
        "output_video_url": s3_url,
        "asr": True,
        "semantic": True,
        "vision": True,
    }

    return result
    
