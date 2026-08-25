"""General speech-safe, frame-aware visual reset microtrim.

This layer never chooses takes or changes semantic order. It operates only on an
already-rendered timeline. Candidate cuts must sit between ASR word envelopes, overlap
objective source silence, and show a strong face/head disengagement after speech ends.
Ambiguous regions fail open unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Any

from .asr import FasterWhisperASR


@dataclass(frozen=True)
class _FaceState:
    visible: bool
    center_y: float | None
    height: float | None


def _silences(path: str) -> tuple[tuple[float, float], ...]:
    command = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-vn", "-af", "silencedetect=noise=-37dB:d=0.05", "-f", "null", "-",
    ]
    completed = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if completed.returncode != 0:
        return ()
    starts: list[float] = []
    out: list[tuple[float, float]] = []
    for line in completed.stderr.splitlines():
        sm = re.search(r"silence_start:\s*([0-9.]+)", line)
        if sm:
            starts.append(float(sm.group(1)))
            continue
        em = re.search(r"silence_end:\s*([0-9.]+)", line)
        if em and starts:
            start = starts.pop(0)
            end = float(em.group(1))
            if end > start:
                out.append((start, end))
    return tuple(out)


def _quiet_ratio(intervals: tuple[tuple[float, float], ...], start: float, end: float) -> float:
    duration = max(1e-6, end - start)
    quiet = sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals)
    return quiet / duration


def _face_state(frame, face_mesh) -> _FaceState:
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    faces = getattr(result, "multi_face_landmarks", None) or ()
    if not faces:
        return _FaceState(False, None, None)
    pts = faces[0].landmark
    ys = [float(p.y) for p in pts]
    if not ys:
        return _FaceState(False, None, None)
    low = min(ys)
    high = max(ys)
    return _FaceState(True, sum(ys) / len(ys), high - low)


def _sample_states(path: str, times: list[float]) -> list[_FaceState]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    states: list[_FaceState] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                states.append(_FaceState(False, None, None))
                continue
            states.append(_face_state(frame, face_mesh))
    finally:
        face_mesh.close()
        cap.release()
    return states


def _visual_reset_onset(path: str, left_word_end: float, safe_start: float, safe_end: float) -> tuple[float | None, dict[str, Any]]:
    # Use real frame cadence. A 30 fps video yields ~33 ms timing granularity.
    frame_step = 1.0 / 30.0
    baseline_times = [max(0.0, left_word_end - 0.10), max(0.0, left_word_end - 0.05), left_word_end]
    candidate_times: list[float] = []
    t = safe_start
    while t <= safe_end + 1e-6:
        candidate_times.append(t)
        t += frame_step
    times = baseline_times + candidate_times
    states = _sample_states(path, times)
    if len(states) != len(times):
        return None, {"reason": "visual_sampling_unavailable"}

    baseline = [s for s in states[:3] if s.visible and s.center_y is not None and s.height is not None]
    if len(baseline) < 2:
        return None, {"reason": "baseline_face_not_stable"}
    base_y = sum(float(s.center_y) for s in baseline) / len(baseline)
    base_h = sum(float(s.height) for s in baseline) / len(baseline)

    hits: list[bool] = []
    metrics: list[dict[str, float | bool | None]] = []
    for state in states[3:]:
        if not state.visible or state.center_y is None or state.height is None:
            hits.append(True)
            metrics.append({"face_visible": False, "dy": None, "scale_delta": None})
            continue
        dy = float(state.center_y) - base_y
        scale_delta = abs(float(state.height) - base_h) / max(1e-6, base_h)
        # Positive dy means the face moved down in the frame. Large absolute shifts also
        # count because a creator may lean sideways/up during a reset.
        reset = dy >= 0.016 or abs(dy) >= 0.028 or scale_delta >= 0.045
        hits.append(reset)
        metrics.append({"face_visible": True, "dy": round(dy, 4), "scale_delta": round(scale_delta, 4)})

    # Require persistence for at least two consecutive sampled frames. One-frame blips
    # are treated as ambiguity and are left untouched.
    for index in range(max(0, len(hits) - 1)):
        if hits[index] and hits[index + 1]:
            onset = candidate_times[index]
            return onset, {
                "reason": "persistent_face_head_reset_after_word",
                "frame_step_sec": round(frame_step, 4),
                "baseline_center_y": round(base_y, 4),
                "baseline_face_height": round(base_h, 4),
                "onset_metrics": metrics[index:index + 2],
            }
    return None, {"reason": "no_persistent_visual_reset"}


def detect_speech_safe_visual_microtrims(
    path: str,
    *,
    asr_model: str = "medium",
    language_hint: str | None = None,
    max_total_trim_sec: float = 2.0,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Find sub-second reset slack strictly between spoken-word envelopes."""
    transcript = FasterWhisperASR(model_name=asr_model).transcribe(
        path, source_asset_id="rendered-output", language_hint=language_hint,
    )
    words = sorted(
        [word for segment in transcript for word in segment.words],
        key=lambda w: (float(w.start), float(w.end)),
    )
    silences = _silences(path)
    if len(words) < 2 or not silences:
        return (), {"speech_lock_ok": True, "candidate_count": 0, "reason": "insufficient_evidence"}

    cuts: list[dict[str, Any]] = []
    candidates = 0
    total_trim = 0.0
    for left, right in zip(words, words[1:]):
        left_end = float(left.end)
        right_start = float(right.start)
        raw_gap = right_start - left_end
        # This layer is only for micro-reset slack, never long editorial pauses.
        if raw_gap < 0.12 or raw_gap > 0.62:
            continue
        safe_start = left_end + 0.035
        safe_end = right_start - 0.045
        if safe_end - safe_start < 0.075:
            continue
        quiet_ratio = _quiet_ratio(silences, safe_start, safe_end)
        if quiet_ratio < 0.72:
            continue
        candidates += 1
        onset, visual = _visual_reset_onset(path, left_end, safe_start, safe_end)
        if onset is None:
            continue
        cut_start = max(safe_start, float(onset))
        cut_end = safe_end
        cut_duration = cut_end - cut_start
        if cut_duration < 0.075 or cut_duration > 0.42:
            continue
        if total_trim + cut_duration > max_total_trim_sec:
            break
        # Physical speech lock: the removal interval is strictly after left word end and
        # strictly before right word start, with guards on both sides.
        if cut_start <= left_end + 0.02 or cut_end >= right_start - 0.02:
            continue
        cuts.append({
            "start": round(cut_start, 3),
            "end": round(cut_end, 3),
            "duration_sec": round(cut_duration, 3),
            "reason": "auto_speech_safe_visual_reset_microtrim",
            "left_word": str(left.text),
            "right_word": str(right.text),
            "left_word_end": round(left_end, 3),
            "right_word_start": round(right_start, 3),
            "quiet_ratio": round(quiet_ratio, 3),
            "visual_evidence": visual,
        })
        total_trim += cut_duration

    return tuple(cuts), {
        "speech_lock_ok": True,
        "word_count": len(words),
        "candidate_count": candidates,
        "auto_microtrim_count": len(cuts),
        "auto_microtrim_duration_sec": round(total_trim, 3),
        "frame_aware": True,
        "rule": "word_end_plus_acoustic_guard_then_visual_reset_until_next_word_guard",
    }
