import os
import logging
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
import subprocess
import tempfile
import math

import clip  # de openai/CLIP

logger = logging.getLogger("editdna.vision")

# Configs desde ENV
VISION_INTERVAL_SEC = float(os.environ.get("VISION_INTERVAL_SEC", "2.0"))
VISION_MAX_SAMPLES = int(os.environ.get("VISION_MAX_SAMPLES", "50"))
W_VISION = float(os.environ.get("W_VISION", "0.7"))  # peso visual vs semántico

_DEVICE = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def get_device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"
    logger.info(f"[VISION] usando device={_DEVICE}")
    return _DEVICE


def get_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS

    device = get_device()
    logger.info("[VISION] cargando CLIP (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _extract_frame(input_video: str, t: float, out_path: str) -> bool:
    """
    Extrae un frame con ffmpeg en el tiempo t (segundos).
    Devuelve True si se creó el archivo.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{t:.3f}",
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
        logger.warning(f"[VISION] ffmpeg frame extract fallo: {proc.stderr}")
        return False
    return True


def _clip_score_image_text(
    image_path: str,
    text_prompts: List[str],
) -> Optional[float]:
    """
    Calcula similitud CLIP imagen-texto; devuelve un score normalizado 0-1.
    """
    if not text_prompts:
        return None

    model, preprocess = get_clip_model()
    device = get_device()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"[VISION] error abriendo imagen {image_path}: {e}")
        return None

    with torch.no_grad():
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(text_prompts).to(device)

        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        # logits ~[-1, 1]; normalizamos a [0, 1]
        score = logits_per_image.max().item()
        score_01 = (score + 1.0) / 2.0
        return max(0.0, min(1.0, score_01))


def run_visual_pass(
    input_local: str,
    clips: List[Dict[str, Any]],
    funnel_prompts: Optional[Dict[str, str]] = None,
) -> None:
    """
    Enriquecemos cada clip con:
      - visual_score (0-1)
      - face_q / scene_q (por ahora igual a visual_score como proxy)
    NO devolvemos nada; modificamos in-place.
    """

    if not clips:
        return

    # Textos por slot (puedes tunearlos luego)
    default_prompts = {
        "HOOK": "person speaking directly to camera with strong eye contact and expressive face, good lighting, vertical video for tiktok",
        "PROBLEM": "person explaining a problem or frustration, serious face, clear view of face",
        "FEATURES": "close up of product in hand, label visible, packaging, clear details of product",
        "BENEFITS": "happy confident person, relaxed and smiling, lifestyle scene, vertical social media ad",
        "PROOF": "testimonial style shot, person talking to camera, real life setting",
        "CTA": "person pointing to bottom of screen or side, gesture to link, clear eye contact, strong gesture",
        "STORY": "natural talking shot, conversational, relaxed body language",
    }
    if funnel_prompts:
        default_prompts.update(funnel_prompts)

    device = get_device()
    if device != "cuda":
        logger.warning("[VISION] CUDA no disponible, vision correrá en CPU (más lento).")

    # Limitamos número de clips a muestrear
    samples = clips
    if len(samples) > VISION_MAX_SAMPLES:
        # muestreo uniforme
        step = math.ceil(len(samples) / VISION_MAX_SAMPLES)
        samples = clips[::step]

    logger.info(
        f"[VISION] corriendo CLIP sobre {len(samples)}/{len(clips)} clips "
        f"(W_VISION={W_VISION})"
    )

    tmp_dir = tempfile.mkdtemp(prefix="editdna_vision_")

    for c in samples:
        start = float(c.get("start", 0.0))
        end = float(c.get("end", start))
        if end <= start:
            continue

        mid_t = (start + end) / 2.0
        clip_id = c.get("id", "clip")

        # extraer frame
        frame_path = os.path.join(tmp_dir, f"{clip_id}.jpg")
        ok = _extract_frame(input_local, mid_t, frame_path)
        if not ok:
            continue

        slot = c.get("slot", "STORY")
        prompt = default_prompts.get(slot, default_prompts["STORY"])

        vs = _clip_score_image_text(frame_path, [prompt])
        if vs is None:
            continue

        # Guardamos visual_score
        c["visual_score"] = float(vs)
        # Como proxy simple:
        c["face_q"] = float(vs)
        c["scene_q"] = float(vs)

        # Recombinar en el meta (si existe) y score total
        meta = c.get("meta", {})
        sem = float(c.get("semantic_score", meta.get("semantic_score", 0.0)))
        # combinamos semántico + visual
        total = (1.0 - W_VISION) * sem + W_VISION * vs

        c["score"] = total
        meta["visual_score"] = vs
        meta["score"] = total
        c["meta"] = meta

    logger.info("[VISION] CLIP visual pass completado.")
    
