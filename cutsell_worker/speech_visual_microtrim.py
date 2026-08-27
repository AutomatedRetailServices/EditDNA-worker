"""General speech-safe, frame-aware visual reset microtrim.

This layer never chooses takes or changes semantic order. It operates only on an
already-rendered timeline. Candidate cuts must sit between ASR word envelopes, overlap
objective source silence, and show a strong post-speech visual disengagement. The
visual evidence may be face/head movement or a persistent silent wrist/hand drop such
as lowering a handheld microphone after finishing a take. Ambiguous regions fail open
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
import subprocess
from typing import Any

from .asr import FasterWhisperASR


@dataclass(frozen=True)
class _VisualState:
    face_visible: bool
    face_center_y: float | None
    face_height: float | None
    left_wrist_y: float | None
    right_wrist_y: float | None


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


def _visual_state(frame, face_mesh, pose) -> _VisualState:
    cv2 = importlib.import_module("cv2")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb)
    pose_result = pose.process(rgb)

    face_visible = False
    face_center_y: float | None = None
    face_height: float | None = None
    faces = getattr(face_result, "multi_face_landmarks", None) or ()
    if faces:
        ys = [float(p.y) for p in faces[0].landmark]
        if ys:
            face_visible = True
            face_center_y = sum(ys) / len(ys)
            face_height = max(ys) - min(ys)

    left_wrist_y: float | None = None
    right_wrist_y: float | None = None
    landmarks = getattr(pose_result, "pose_landmarks", None)
    if landmarks is not None:
        pts = landmarks.landmark
        if len(pts) > 16:
            left = pts[15]
            right = pts[16]
            if float(getattr(left, "visibility", 0.0) or 0.0) >= 0.45:
                left_wrist_y = float(left.y)
            if float(getattr(right, "visibility", 0.0) or 0.0) >= 0.45:
                right_wrist_y = float(right.y)

    return _VisualState(
        face_visible=face_visible,
        face_center_y=face_center_y,
        face_height=face_height,
        left_wrist_y=left_wrist_y,
        right_wrist_y=right_wrist_y,
    )


def _sample_states(path: str, times: list[float]) -> list[_VisualState]:
    cv2 = importlib.import_module("cv2")
    mp = importlib.import_module("mediapipe")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    states: list[_VisualState] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                states.append(_VisualState(False, None, None, None, None))
                continue
            states.append(_visual_state(frame, face_mesh, pose))
    finally:
        face_mesh.close()
        pose.close()
        cap.release()
    return states


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _visual_reset_onset(path: str, left_word_end: float, safe_start: float, safe_end: float) -> tuple[float | None, dict[str, Any]]:
    frame_step = 1.0 / 30.0
    baseline_times = [
        max(0.0, left_word_end - 0.12),
        max(0.0, left_word_end - 0.08),
        max(0.0, left_word_end - 0.04),
        left_word_end,
    ]
    candidate_times: list[float] = []
    t = safe_start
    while t <= safe_end + 1e-6:
        candidate_times.append(t)
        t += frame_step
    times = baseline_times + candidate_times
    states = _sample_states(path, times)
    if len(states) != len(times):
        return None, {"reason": "visual_sampling_unavailable"}

    baseline_states = states[: len(baseline_times)]
    face_baseline = [
        s for s in baseline_states
        if s.face_visible and s.face_center_y is not None and s.face_height is not None
    ]
    base_y = _median([float(s.face_center_y) for s in face_baseline if s.face_center_y is not None])
    base_h = _median([float(s.face_height) for s in face_baseline if s.face_height is not None])
    base_lw = _median([float(s.left_wrist_y) for s in baseline_states if s.left_wrist_y is not None])
    base_rw = _median([float(s.right_wrist_y) for s in baseline_states if s.right_wrist_y is not None])

    if base_y is None and base_lw is None and base_rw is None:
        return None, {"reason": "baseline_visual_state_unavailable"}

    hits: list[bool] = []
    metrics: list[dict[str, float | bool | None | str]] = []
    for state in states[len(baseline_times):]:
        face_reset = False
        dy: float | None = None
        scale_delta: float | None = None
        if base_y is not None and base_h is not None:
            if not state.face_visible or state.face_center_y is None or state.face_height is None:
                face_reset = True
            else:
                dy = float(state.face_center_y) - base_y
                scale_delta = abs(float(state.face_height) - base_h) / max(1e-6, base_h)
                face_reset = dy >= 0.016 or abs(dy) >= 0.028 or scale_delta >= 0.045

        left_wrist_drop: float | None = None
        right_wrist_drop: float | None = None
        if base_lw is not None and state.left_wrist_y is not None:
            left_wrist_drop = float(state.left_wrist_y) - base_lw
        if base_rw is not None and state.right_wrist_y is not None:
            right_wrist_drop = float(state.right_wrist_y) - base_rw

        gesture_reset = (
            (left_wrist_drop is not None and left_wrist_drop >= 0.035)
            or (right_wrist_drop is not None and right_wrist_drop >= 0.035)
        )
        reset = face_reset or gesture_reset
        hits.append(reset)
        metrics.append({
            "face_visible": state.face_visible,
            "dy": None if dy is None else round(dy, 4),
            "scale_delta": None if scale_delta is None else round(scale_delta, 4),
            "left_wrist_drop": None if left_wrist_drop is None else round(left_wrist_drop, 4),
            "right_wrist_drop": None if right_wrist_drop is None else round(right_wrist_drop, 4),
            "channel": "face" if face_reset else ("gesture" if gesture_reset else "none"),
        })

    for index in range(max(0, len(hits) - 1)):
        if hits[index] and hits[index + 1]:
            onset = candidate_times[index]
            channels = {str(metrics[index].get("channel")), str(metrics[index + 1].get("channel"))}
            reason = (
                "persistent_silent_gesture_reset_after_word"
                if "gesture" in channels and "face" not in channels
                else "persistent_face_or_gesture_reset_after_word"
            )
            return onset, {
                "reason": reason,
                "frame_step_sec": round(frame_step, 4),
                "baseline_center_y": None if base_y is None else round(base_y, 4),
                "baseline_face_height": None if base_h is None else round(base_h, 4),
                "baseline_left_wrist_y": None if base_lw is None else round(base_lw, 4),
                "baseline_right_wrist_y": None if base_rw is None else round(base_rw, 4),
                "onset_metrics": metrics[index:index + 2],
            }
    return None, {"reason": "no_persistent_visual_reset"}


def _minimum_quiet_ratio(raw_gap: float) -> float:
    """Require stronger acoustic evidence as the candidate reset gap gets longer."""
    if raw_gap <= 0.62:
        return 0.68
    if raw_gap <= 0.90:
        return 0.76
    return 0.82


def detect_speech_safe_visual_microtrims(
    path: str,
    *,
    asr_model: str = "medium",
    language_hint: str | None = None,
    max_total_trim_sec: float = 4.0,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Find reset slack strictly between spoken-word envelopes.

    Human edits often remove a creator's visible post-take reset even when that reset is
    longer than a classic sub-second micro-pause. We therefore allow gaps up to 1.25 s,
    but only when the removal is physically speech-safe, acoustically quiet, and backed
    by a persistent face/body reset. Longer candidates require stronger quiet evidence.
    """
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
        if raw_gap < 0.12 or raw_gap > 1.25:
            continue
        safe_start = left_end + 0.035
        safe_end = right_start - 0.045
        if safe_end - safe_start < 0.075:
            continue
        quiet_ratio = _quiet_ratio(silences, safe_start, safe_end)
        minimum_quiet = _minimum_quiet_ratio(raw_gap)
        if quiet_ratio < minimum_quiet:
            continue
        candidates += 1
        onset, visual = _visual_reset_onset(path, left_end, safe_start, safe_end)
        if onset is None:
            continue
        cut_start = max(safe_start, float(onset))
        cut_end = safe_end
        cut_duration = cut_end - cut_start
        if cut_duration < 0.075 or cut_duration > 0.90:
            continue
        if total_trim + cut_duration > max_total_trim_sec:
            break
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
            "minimum_quiet_ratio": round(minimum_quiet, 3),
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
        "visual_channels": ["face_head", "pose_wrist_gesture"],
        "max_reset_gap_sec": 1.25,
        "max_single_trim_sec": 0.90,
        "rule": "word_end_plus_acoustic_guard_then_persistent_visual_reset_until_next_word_guard",
    }
