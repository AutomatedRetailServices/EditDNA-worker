"""Smart zoom framing (PO 16 Nov 2026): WHERE an Auto Zoom AI push-in goes, by what is being said.

- hook / benefit / call to action -> her FACE (placed a little above the middle, so the head is never cut);
- product / price / "look at this" -> the area just below her face, where the product in her hand usually is,
  keeping her face in the picture;
- no face found (or no OpenCV in the worker) -> the centre, exactly as before.

Face finding is OpenCV's built-in face detector on one small grey frame per zoom (CPU, no paid AI). The answer is
the pan of the zoomed picture (keyframes.py x / y: the picture centre's offset as a fraction of the frame).
"""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple

from .keyframes import clamp_offset

FACE_KINDS = ("hook", "benefit", "cta")
PRODUCT_KINDS = ("product", "price", "look")
FRAME_WIDTH = 320
#: Where the target ends up on screen (fraction of the frame, from the top-left).
FACE_SPOT = (0.5, 0.40)
PRODUCT_SPOT = (0.5, 0.55)
#: The product spot: this many face heights below the centre of her face.
PRODUCT_BELOW_FACE = 1.6
#: Her whole face stays at least this far inside the zoomed picture; above it, room for her hair too
#: (the face finder's box stops at the forehead).
FACE_MARGIN = 0.03
HAIR_ROOM = 0.6

Face = Tuple[float, float, float, float]     # centre x, centre y, width, height (fractions of the frame)


def grab_gray(path: str, t: float, *, width: int = FRAME_WIDTH) -> Optional[tuple[bytes, int, int]]:
    """One grey frame at source second t, `width` px wide (rotation applied by ffmpeg): (pixels, w, h)."""
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale={width}:-2,format=gray", "-f", "rawvideo", "-"],
            capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    data = done.stdout
    if done.returncode != 0 or not data or len(data) % width:
        return None
    return data, width, len(data) // width


_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        import cv2  # noqa: PLC0415 -- optional: no OpenCV = centred zooms

        _DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _DETECTOR


def find_face(pixels: bytes, width: int, height: int) -> Optional[Face]:
    """The biggest face in a grey frame, or None."""
    try:
        import numpy as np

        detector = _detector()
        gray = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
        found = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(max(24, width // 12), max(24, width // 12)))
    except Exception:                       # noqa: BLE001 -- no OpenCV / bad frame: centred zoom
        return None
    if found is None or len(found) == 0:
        return None
    x, y, w, h = max((tuple(int(v) for v in f) for f in found), key=lambda f: f[2] * f[3])
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


#: Two looks, a little apart, must agree on where her face is (a hand, a bottle or her chest can look like a face
#: in one frame): centres this close, as a fraction of the frame.
AGREE_WITHIN = 0.08
LOOKS = (-0.5, -0.25, 0.0, 0.25, 0.5)


def face_at(path: str, t: float) -> Optional[Face]:
    """Her face around source second t: five looks (from half a second before to half a second after); two must agree,
    and the answer is their average. No agreement = None (a centred zoom, as before)."""
    faces = []
    for delta in LOOKS:
        when = float(t) + delta
        if when < 0:
            continue
        frame = grab_gray(path, when)
        face = find_face(*frame) if frame else None
        if face:
            faces.append(face)
    best: list[Face] = []
    for face in faces:
        group = [f for f in faces if abs(f[0] - face[0]) <= AGREE_WITHIN and abs(f[1] - face[1]) <= AGREE_WITHIN]
        if len(group) > len(best):
            best = group
    if len(best) < 2:
        return None
    return tuple(round(sum(f[k] for f in best) / len(best), 4) for k in range(4))  # type: ignore[return-value]


def pan_for(kind: str, face: Optional[Face], scale: float) -> tuple[float, float]:
    """The (x, y) pan of a zoom of `scale` for a moment of `kind` (0, 0 = centred, as before)."""
    if face is None or scale <= 1.0:
        return 0.0, 0.0
    fx, fy, fw, fh = face
    if kind in FACE_KINDS:
        (tx, ty), (px, py) = (fx, fy), FACE_SPOT
    elif kind in PRODUCT_KINDS:
        (tx, ty), (px, py) = (fx, min(0.85, fy + PRODUCT_BELOW_FACE * fh)), PRODUCT_SPOT
    else:
        return 0.0, 0.0
    # A point p of the picture shows at 0.5 + (p - 0.5) * scale + offset.
    x = clamp_offset(scale, px - 0.5 - (tx - 0.5) * scale)
    y = clamp_offset(scale, py - 0.5 - (ty - 0.5) * scale)
    # Keep her whole face inside the picture (never a cut head or chin).
    top = 0.5 + (fy - fh / 2 - HAIR_ROOM * fh - 0.5) * scale + y
    bottom = 0.5 + (fy + fh / 2 - 0.5) * scale + y
    if top < FACE_MARGIN:
        y = clamp_offset(scale, y + FACE_MARGIN - top)
    elif bottom > 1 - FACE_MARGIN:
        y = clamp_offset(scale, y - (bottom - (1 - FACE_MARGIN)))
    left = 0.5 + (fx - fw / 2 - 0.5) * scale + x
    right = 0.5 + (fx + fw / 2 - 0.5) * scale + x
    if left < FACE_MARGIN:
        x = clamp_offset(scale, x + FACE_MARGIN - left)
    elif right > 1 - FACE_MARGIN:
        x = clamp_offset(scale, x - (right - (1 - FACE_MARGIN)))
    return round(x, 4), round(y, 4)
