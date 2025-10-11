# worker/vision_sampler.py
import os, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# Optional CLIP
try:
    import open_clip
    import torch
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip_model.eval()
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False
    _clip_model = None

# ---- utilities ----
def _safe_read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _variance_of_laplacian(img: np.ndarray) -> float:
    # higher = sharper
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

def _exposure_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    hist /= (hist.sum() + 1e-6)
    # penalize extremes (too dark/bright)
    dark = hist[:16].sum()
    bright = hist[-16:].sum()
    score = 1.0 - min(1.0, dark + bright)  # 1 when well-exposed, 0 when crushed
    return float(max(0.0, min(1.0, score)))

def _face_center_score(img: np.ndarray, faces: List[Tuple[int,int,int,int]]) -> float:
    if not faces:
        return 0.0
    h, w = img.shape[:2]
    cx, cy = w/2, h/2
    # pick largest
    x,y,wf,hf = max(faces, key=lambda r: r[2]*r[3])
    fx, fy = x + wf/2, y + hf/2
    # distance from center normalized by diagonal
    d = math.hypot(fx - cx, fy - cy)
    dnorm = d / (math.hypot(w, h) + 1e-6)
    size = (wf*hf) / (w*h)
    # bigger face & closer to center -> higher
    return float(max(0.0, min(1.0, (1.0 - dnorm) * min(1.0, 3.0*size + 0.2))))

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_faces(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))
    return [(int(x),int(y),int(w),int(h)) for (x,y,w,h) in faces]

def _clip_similarity(img: np.ndarray, text: str) -> float:
    if not (_HAS_CLIP and text):
        return 0.0
    import PIL.Image as Image
    with torch.no_grad():
        pil = Image.fromarray(img)
        image = _clip_preprocess(pil).unsqueeze(0)
        text_tok = _clip_tokenizer([text])
        if torch.cuda.is_available():
            image = image.cuda()
            _clip_model.cuda()
        img_feat = _clip_model.encode_image(image)
        txt_feat = _clip_model.encode_text(text_tok)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze().item()
        # map [-1..1] to [0..1]
        return float((sim + 1.0) / 2.0)

@dataclass
class FrameScores:
    face_q: float
    scene_q: float
    vtx_sim: float

def sample_scores_for_span(
    video_path: str,
    start_s: float,
    end_s: float,
    text_hint: str = "",
    frames_per_take: int = 3,
) -> FrameScores:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if total > 0 else (end_s - start_s)

    # pick indices at start/mid/end of the span
    t0 = max(0.0, start_s)
    t1 = max(t0, (start_s + end_s) / 2.0)
    t2 = max(t1, end_s - 0.001)
    times = [t0, t1, t2][:frames_per_take]

    face_scores, scene_scores, vtx_scores = [], [], []
    for t in times:
        idx = int(t * fps)
        frame = _safe_read_frame(cap, idx)
        if frame is None:
            continue
        faces = _detect_faces(frame)
        face_scores.append(_face_center_score(frame, faces))
        # scene quality: sharpness * exposure
        sharp = _variance_of_laplacian(frame)
        sharp_norm = max(0.0, min(1.0, (sharp - 50.0) / 400.0))  # rough 0..1
        scene_scores.append(0.5 * sharp_norm + 0.5 * _exposure_score(frame))
        vtx_scores.append(_clip_similarity(frame, text_hint))

    cap.release()
    if not face_scores:
        face_scores = [0.0]
    if not scene_scores:
        scene_scores = [0.5]
    if not vtx_scores:
        vtx_scores = [0.0]

    return FrameScores(
        face_q=float(np.clip(np.mean(face_scores), 0.0, 1.0)),
        scene_q=float(np.clip(np.mean(scene_scores), 0.0, 1.0)),
        vtx_sim=float(np.clip(np.mean(vtx_scores), 0.0, 1.0)),
    )
