cat > /workspace/editdna/app/worker/vision_sampler.py <<'PY'
from __future__ import annotations
import os, cv2, numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Optional CLIP (for vtx_sim); if not present, we’ll return 0.0
try:
    import open_clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

@dataclass
class VSample:
    t: float
    face_q: float
    blur_q: float
    exposure_q: float
    motion_q: float
    vtx_sim: float

def _face_score(frame: np.ndarray) -> float:
    # Very light face proxy: center/crop score + presence of face-like blobs
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    # Brightness/contrast quick check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    # Penalize too dark or blown
    exp = 1.0 - min(abs(m-128.0)/128.0, 1.0)

    # Center energy proxy (encourages face near center)
    cy, cx = h//2, w//2
    box = gray[cy-h//8:cy+h//8, cx-w//8:cx+w//8] if h>16 and w>16 else gray
    center = float(np.mean(box))/255.0

    return float(max(0.0, min(1.0, 0.6*exp + 0.4*center)))

def _blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Map Laplacian variance (0..~3000 typical) to 0..1
    return float(max(0.0, min(1.0, fm / 800.0)))

def _exposure_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    return float(1.0 - min(abs(m-128.0)/128.0, 1.0))

def _motion_score(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    if prev is None: 
        return 1.0
    g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    v = float(np.mean(diff)) / 255.0
    # prefer modest motion (0.02..0.15), penalize extreme
    if v < 0.02: return 0.6 + v*10
    if v > 0.25: return max(0.0, 1.2 - v*2.5)
    return 0.9

class _ClipHelper:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def sim(self, frame: np.ndarray, text: str) -> float:
        import torch
        if not text: 
            return 0.0
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.preprocess(open_clip.Image.fromarray(img)).unsqueeze(0)
        toks = self.tokenizer([text])
        with torch.no_grad():
            i_f = self.model.encode_image(img)
            t_f = self.model.encode_text(toks)
            i_f /= i_f.norm(dim=-1, keepdim=True)
            t_f /= t_f.norm(dim=-1, keepdim=True)
            s = (i_f @ t_f.T).squeeze().item()
        # map cosine (-1..1) to 0..1
        return float((s + 1.0) / 2.0)

_CLIP: Optional[_ClipHelper] = None
def _get_clip():
    global _CLIP
    if _CLIP is None and _HAS_CLIP:
        _CLIP = _ClipHelper()
    return _CLIP

def sample_visuals(
    video_path: str,
    start_s: float,
    end_s: float,
    text_hint: str = "",
    max_samples: int = 3
) -> Tuple[float, float, float, float]:
    """
    Returns (face_q, scene_q, vtx_sim, scene_cut_flag[0/1])
    scene_q is the mean of blur_q, exposure_q, motion_q.
    """
    if not os.path.exists(video_path):
        return 0.0, 0.0, 0.0, 0.0

    dur = max(0.001, end_s - start_s)
    ts = np.linspace(start_s, end_s, num=max_samples, endpoint=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_frame = None
    samples: List[VSample] = []

    clip = _get_clip()

    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        f = _face_score(frame)
        b = _blur_score(frame)
        e = _exposure_score(frame)
        m = _motion_score(prev_frame, frame)
        vtx = clip.sim(frame, text_hint) if clip else 0.0

        samples.append(VSample(t=float(t), face_q=f, blur_q=b, exposure_q=e, motion_q=m, vtx_sim=vtx))
        prev_frame = frame

    cap.release()

    if not samples:
        return 0.0, 0.0, 0.0, 0.0

    face_q = float(np.mean([s.face_q for s in samples]))
    scene_q = float(np.mean([(s.blur_q + s.exposure_q + s.motion_q)/3.0 for s in samples]))
    vtx_sim = float(np.mean([s.vtx_sim for s in samples]))

    # scene-cut flag (very rough): large motion dip or big brightness jump
    sc = 0.0
    if len(samples) >= 2:
        diffs = []
        br = []
        for i in range(1, len(samples)):
            diffs.append(abs(samples[i].motion_q - samples[i-1].motion_q))
            br.append(abs(samples[i].exposure_q - samples[i-1].exposure_q))
        if (max(diffs) if diffs else 0) > 0.45 or (max(br) if br else 0) > 0.5:
            sc = 1.0

    return face_q, scene_q, vtx_sim, sc
PY
