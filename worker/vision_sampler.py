cat > /workspace/editdna/app/worker/vision_sampler.py <<'PY'
from __future__ import annotations
import cv2, os, numpy as np
from typing import Tuple, Optional, List

# Optional CLIP for vtx_sim
_HAS_CLIP = False
try:
    import torch
    import open_clip
    _clip_model = None
    _clip_tokenizer = None
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

def _lazy_clip_init() -> bool:
    global _clip_model, _clip_tokenizer, _HAS_CLIP
    if not _HAS_CLIP:
        return False
    if _clip_model is not None:
        return True
    try:
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _clip_model.eval()
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return True
    except Exception:
        _HAS_CLIP = False
        return False

def _center_sharpness(gray: np.ndarray) -> float:
    h, w = gray.shape
    ch0, ch1 = int(h*0.35), int(h*0.65)
    cw0, cw1 = int(w*0.35), int(w*0.65)
    center = gray[ch0:ch1, cw0:cw1]
    var = cv2.Laplacian(center, cv2.CV_64F).var()
    return float(max(0.0, min(1.0, var / 300.0)))

def _exposure_score(gray: np.ndarray) -> float:
    mean = gray.mean() / 255.0
    return float(max(0.0, 1.0 - abs(mean - 0.5) * 2.0))

def _motion_score(prev_gray: Optional[np.ndarray], cur_gray: np.ndarray) -> float:
    if prev_gray is None:
        return 1.0
    diff = cv2.absdiff(prev_gray, cur_gray)
    mag = float(diff.mean()) / 255.0
    return float(max(0.0, 1.0 - mag * 2.0))

def _clip_similarity(frames_bgr: List[np.ndarray], text: str) -> float:
    if not text or not _lazy_clip_init():
        return 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model.to(device)
    with torch.no_grad():
        toks = _clip_tokenizer([text]).to(device)
        t_feat = _clip_model.encode_text(toks)
        t_feat /= t_feat.norm(dim=-1, keepdim=True)
        sims = []
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            # CLIP normalization
            img = (img - 0.48145466) / 0.26862954
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            i_feat = _clip_model.encode_image(img)
            i_feat /= i_feat.norm(dim=-1, keepdim=True)
            sim = (i_feat @ t_feat.T).squeeze().item()
            sims.append(0.5 * (sim + 1.0))  # map [-1,1] to [0,1]
        return float(sum(sims)/len(sims)) if sims else 0.0

def sample_visuals(
    video_path: str,
    span: Tuple[float, float],
    *,
    text: Optional[str] = None,
    fps: int = 2,
    max_frames: int = 6,
) -> tuple[float, float, float, bool]:
    """
    Returns: (face_q, scene_q, vtx_sim, had_signal)
    """
    if not os.path.exists(video_path):
        return 0.0, 0.0, 0.0, False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0.0, 0.0, False

    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start, end = span
    start = max(0.0, float(start)); end = max(start, float(end))
    dur = end - start
    if dur <= 0.02:
        cap.release()
        return 0.0, 0.0, 0.0, False

    n = max(1, min(max_frames, int(dur * fps)))
    idxs = [int((start + (i + 0.5) * dur / n) * vfps) for i in range(n)]

    face_scores, exp_scores, mot_scores = [], [], []
    frames_for_clip = []
    prev = None
    had = False

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        had = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_scores.append(_center_sharpness(gray))
        exp_scores.append(_exposure_score(gray))
        mot_scores.append(_motion_score(prev, gray))
        prev = gray
        if len(frames_for_clip) < max_frames:
            frames_for_clip.append(frame)

    cap.release()
    if not had:
        return 0.0, 0.0, 0.0, False

    face_q = float(np.median(face_scores)) if face_scores else 0.0
    scene_q = float(min(
        np.median(exp_scores) if exp_scores else 0.0,
        np.median(mot_scores) if mot_scores else 0.0
    ))
    vtx_sim = _clip_similarity(frames_for_clip, text or "")
    if face_q < 0.05 and scene_q < 0.05 and vtx_sim < 0.05:
        return face_q, scene_q, vtx_sim, False
    return face_q, scene_q, vtx_sim, True
PY
