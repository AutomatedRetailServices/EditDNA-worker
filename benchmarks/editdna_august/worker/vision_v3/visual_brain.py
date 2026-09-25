# worker/vision_v3/visual_brain.py

import math
from typing import Dict, Tuple

import cv2
import numpy as np


def _grab_frame(path: str, t_sec: float):
    """
    Grab a single frame (BGR) at time t_sec.
    Returns None if anything fails.
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return frame
    except Exception:
        return None


def _hist_diff(f1, f2) -> float:
    """
    Compare color histograms of two frames.
    Returns a distance 0..1 (0 = identical, 1 = very different).
    """
    if f1 is None or f2 is None:
        return 0.0

    try:
        f1_small = cv2.resize(f1, (160, 90))
        f2_small = cv2.resize(f2, (160, 90))

        f1_hsv = cv2.cvtColor(f1_small, cv2.COLOR_BGR2HSV)
        f2_hsv = cv2.cvtColor(f2_small, cv2.COLOR_BGR2HSV)

        hist_size = [32, 32]
        ranges = [0, 180, 0, 256]

        h1 = cv2.calcHist([f1_hsv], [0, 1], None, hist_size, ranges)
        h2 = cv2.calcHist([f2_hsv], [0, 1], None, hist_size, ranges)

        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)

        # correlation → 1 is identical; -1 is opposite
        corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        corr = max(min(corr, 1.0), -1.0)
        dist = (1.0 - corr) / 2.0  # 0..1
        return float(dist)
    except Exception:
        return 0.0


def _motion_diff(f1, f2) -> float:
    """
    Approximate motion difference between two frames.
    Returns 0..1 (0 = very similar, 1 = very different).
    """
    if f1 is None or f2 is None:
        return 0.0

    try:
        f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        f1_small = cv2.resize(f1_gray, (160, 90))
        f2_small = cv2.resize(f2_gray, (160, 90))

        diff = cv2.absdiff(f1_small, f2_small)
        mean_diff = float(np.mean(diff)) / 255.0
        return max(0.0, min(1.0, mean_diff))
    except Exception:
        return 0.0


def score_segment(
    path: str,
    start: float,
    end: float,
) -> Tuple[float, Dict[str, bool]]:
    """
    Core V3 visual scoring for a single textual clause.

    We:
      - grab a frame near the start and near the end
      - measure histogram-based scene difference
      - measure pixel-motion difference
      - produce:
          visual_score 0..1
          flags = { "scene_jump": bool, "motion_jump": bool }

    This is deliberately simple but real:
      - it actually looks at video frames
      - no ML model required
      - can be extended later with YOLO/OCR/etc.
    """
    # If the segment is extremely short, treat as visually OK.
    if end <= start:
        return 1.0, {"scene_jump": False, "motion_jump": False}

    mid = (start + end) / 2.0

    # We compare:
    #   - frame at start + small epsilon
    #   - frame at end - small epsilon
    eps = 0.05 * (end - start)
    t1 = max(start, start + eps)
    t2 = max(t1, end - eps)

    f1 = _grab_frame(path, t1)
    f2 = _grab_frame(path, t2)

    scene_dist = _hist_diff(f1, f2)
    motion_dist = _motion_diff(f1, f2)

    # Heuristics:
    # - scene_dist > 0.5 ⇒ likely hard shot change
    # - motion_dist > 0.4 ⇒ big pose/hand/head change
    scene_jump = scene_dist > 0.5
    motion_jump = motion_dist > 0.4

    # Start from perfect visual score and penalize jumps
    visual_score = 1.0
    if scene_jump:
        visual_score -= 0.4
    if motion_jump:
        visual_score -= 0.3

    visual_score = max(0.0, min(1.0, visual_score))

    flags = {
        "scene_jump": scene_jump,
        "motion_jump": motion_jump,
    }
    return visual_score, flags
