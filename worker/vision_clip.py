import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Simple, internal visual coherence checker using CLIP.
# Goal (for V2):
# - For each ASR clause, sample 2 frames (start-ish, end-ish)
# - If those frames are visually VERY different (low cosine sim),
#   we mark that clause as "visual_bad" so composer can drop it.

# Lazy globals
_clip_model = None
_clip_processor = None

# Threshold: how similar start vs end frame must be to be considered "visually coherent"
INTERNAL_SIM_THRESHOLD = float(os.getenv("EDITDNA_INTERNAL_VIS_SIM", "0.80"))


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        # Small-ish CLIP model; works fine on GPU or CPU
        model_name = os.getenv("EDITDNA_CLIP_MODEL", "openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)

        device = os.getenv("EDITDNA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        _clip_model = _clip_model.to(device)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _grab_frame_bgr(path: str, t_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame  # BGR


def _frame_to_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        model, processor = _load_clip()
        device = next(model.parameters()).device

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        emb = outputs.cpu().numpy().astype("float32")
        # Normalize
        norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
        emb = emb / norm
        return emb[0]
    except Exception:
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype("float32")
    b = b.astype("float32")
    dot = float(np.dot(a, b))
    return float(dot)


def enrich_clauses_with_vision(video_path: str, clauses: List["Clause"]) -> None:
    """
    Mutates each Clause in-place:
      - clause.visual_ok: bool
      - clause.visual_internal_sim: float
      - clause.clip_vec: np.ndarray or None (for future use)

    We:
      1) sample 2 frames per clause (start-ish, end-ish)
      2) get CLIP embeddings
      3) compute cosine similarity between them
      4) if sim < INTERNAL_SIM_THRESHOLD → visual_ok = False
    """
    # Fail-safe: if CLIP cannot load, do nothing.
    try:
        _load_clip()
    except Exception:
        for c in clauses:
            setattr(c, "visual_ok", True)
            setattr(c, "visual_internal_sim", 1.0)
            setattr(c, "clip_vec", None)
        return

    for c in clauses:
        duration = max(0.0, c.end - c.start)
        if duration <= 0.0:
            setattr(c, "visual_ok", True)
            setattr(c, "visual_internal_sim", 1.0)
            setattr(c, "clip_vec", None)
            continue

        # sample two times inside the clause
        t1 = c.start + 0.15 * duration
        t2 = c.start + 0.85 * duration

        f1 = _grab_frame_bgr(video_path, t1)
        f2 = _grab_frame_bgr(video_path, t2)

        if f1 is None or f2 is None:
            setattr(c, "visual_ok", False)
            setattr(c, "visual_internal_sim", 0.0)
            setattr(c, "clip_vec", None)
            continue

        e1 = _frame_to_embedding(f1)
        e2 = _frame_to_embedding(f2)

        if e1 is None or e2 is None:
            setattr(c, "visual_ok", False)
            setattr(c, "visual_internal_sim", 0.0)
            setattr(c, "clip_vec", None)
            continue

        sim = _cosine_sim(e1, e2)
        visual_ok = sim >= INTERNAL_SIM_THRESHOLD

        setattr(c, "visual_ok", bool(visual_ok))
        setattr(c, "visual_internal_sim", float(sim))
        # store mid embedding for future continuity logic if needed
        setattr(c, "clip_vec", (e1 + e2) / 2.0)
