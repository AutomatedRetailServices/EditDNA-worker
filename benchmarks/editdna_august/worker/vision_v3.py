import cv2
import math
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class VisualBrainV3:
    """
    Lightweight visual scorer for EditDNA.

    Goals:
    - Detect obviously bad takes:
        * Super blurry
        * Super dark / blown out
        * Huge jump between start/end frames (scene change)
    - Return:
        score in [0,1] and flags:
        {
          "scene_jump": bool,
          "motion_jump": bool,
          "too_blurry": bool,
          "too_dark": bool,
          "too_bright": bool,
        }
    """

    def __init__(
        self,
        blur_good_threshold: float = 150.0,
        blur_bad_threshold: float = 40.0,
        dark_threshold: float = 40.0,
        bright_threshold: float = 220.0,
        scene_jump_threshold: float = 35.0,
        motion_jump_threshold: float = 25.0,
    ):
        self.blur_good_threshold = blur_good_threshold
        self.blur_bad_threshold = blur_bad_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.scene_jump_threshold = scene_jump_threshold
        self.motion_jump_threshold = motion_jump_threshold

    # ---------------------- frame helpers ---------------------- #

    def _grab_frame(self, path: str, t_sec: float):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return None
        return frame

    def _analyze_frame(self, frame) -> Dict[str, float]:
        """
        Return blur_var, brightness_mean.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())
        return {
            "blur_var": blur_var,
            "brightness_mean": brightness_mean,
        }

    def _frame_diff_score(self, f1, f2) -> float:
        """
        Simple difference between two frames: mean absolute difference.
        """
        if f1 is None or f2 is None:
            return 0.0
        try:
            f1r = cv2.resize(f1, (320, 180))
            f2r = cv2.resize(f2, (320, 180))
            diff = cv2.absdiff(f1r, f2r)
            mean_diff = float(diff.mean())
            return mean_diff
        except Exception:
            return 0.0

    # ---------------------- main API ---------------------- #

    def score_segment(self, path: str, start: float, end: float) -> Tuple[float, Dict[str, bool]]:
        """
        Score a segment visually.

        We sample:
        - mid frame (quality)
        - near start + near end (for scene / motion jumps)
        """

        flags = {
            "scene_jump": False,
            "motion_jump": False,
            "too_blurry": False,
            "too_dark": False,
            "too_bright": False,
        }

        try:
            if end <= start:
                return 0.0, flags

            seg_len = max(0.01, end - start)
            mid_t = start + seg_len / 2.0
            start_t = start + min(0.1 * seg_len, 0.3)
            end_t = end - min(0.1 * seg_len, 0.3)

            mid_frame = self._grab_frame(path, mid_t)
            start_frame = self._grab_frame(path, start_t)
            end_frame = self._grab_frame(path, end_t)

            if mid_frame is None:
                # no visual info; neutral-ish but not high
                return 0.5, flags

            stats = self._analyze_frame(mid_frame)
            blur_var = stats["blur_var"]
            brightness = stats["brightness_mean"]

            # Blur score
            if blur_var <= self.blur_bad_threshold:
                blur_score = 0.0
                flags["too_blurry"] = True
            elif blur_var >= self.blur_good_threshold:
                blur_score = 1.0
            else:
                # scale between bad→good
                blur_score = (blur_var - self.blur_bad_threshold) / (
                    self.blur_good_threshold - self.blur_bad_threshold
                )

            # Brightness score (penalize too dark or too bright)
            if brightness <= self.dark_threshold:
                bright_score = 0.0
                flags["too_dark"] = True
            elif brightness >= self.bright_threshold:
                bright_score = 0.0
                flags["too_bright"] = True
            else:
                # ideal around mid (~128)
                dist = abs(brightness - 128.0)
                # 0 at dist >= 90, 1 at dist = 0
                bright_score = max(0.0, 1.0 - (dist / 90.0))

            # Scene / motion jump
            scene_diff = self._frame_diff_score(start_frame, end_frame)
            motion_diff = self._frame_diff_score(mid_frame, end_frame)

            if scene_diff >= self.scene_jump_threshold:
                flags["scene_jump"] = True
            if motion_diff >= self.motion_jump_threshold:
                flags["motion_jump"] = True

            # Convert diffs to penalty in [0,1]
            scene_penalty = min(1.0, scene_diff / (self.scene_jump_threshold * 2.0))
            motion_penalty = min(1.0, motion_diff / (self.motion_jump_threshold * 2.0))

            # Combine
            quality_score = 0.6 * blur_score + 0.4 * bright_score
            penalty = 0.5 * scene_penalty + 0.5 * motion_penalty

            score = quality_score * (1.0 - 0.5 * penalty)
            score = max(0.0, min(1.0, score))

            return score, flags

        except Exception as e:
            logger.exception(f"VisualBrainV3.score_segment failed: {e}")
            return 0.5, flags


# Singleton instance imported by pipeline.py
visual_brain = VisualBrainV3()
