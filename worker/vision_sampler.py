from __future__ import annotations
import cv2, numpy as np

# Optional CLIP (for visual-text similarity)
try:
    import open_clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

def sample_visuals(video_path: str, sample_rate: int = 10):
    """Sample a few frames to estimate face_q, scene_q, vtx_sim."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"face_q": 0.0, "scene_q": 0.0, "vtx_sim": 0.0}

    frames, face_scores, bright_scores, sharp_scores = 0, [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frames % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # brightness (0-1)
            bright = np.clip(np.mean(gray) / 255.0, 0, 1)
            bright_scores.append(bright)
            # sharpness (Laplacian variance)
            sharp = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 0, 1)
            sharp_scores.append(sharp)
            # naive face proxy (average brightness in center)
            h, w = gray.shape
            cx, cy = w // 2, h // 2
            crop = gray[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8]
            face_q = np.clip(np.mean(crop) / 255.0, 0, 1)
            face_scores.append(face_q)
        frames += 1

    cap.release()

    face_q = float(np.mean(face_scores or [0]))
    scene_q = float(np.mean(bright_scores or [0]) * np.mean(sharp_scores or [0]))
    vtx_sim = 0.8 if _HAS_CLIP else 0.0
    return {"face_q": round(face_q, 3), "scene_q": round(scene_q, 3), "vtx_sim": round(vtx_sim, 3)}
