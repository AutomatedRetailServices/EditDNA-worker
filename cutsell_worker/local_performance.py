"""Dense local face/body/hand/motion evidence for the CutSell temporal brain.

This module measures trajectories; it does not decide CUT/KEEP. Abrupt changes are
emitted as ``*_candidate`` events so semantic/retry context remains authoritative.
OpenCV/MediaPipe are dynamically imported at runtime to preserve the clean worker's
light import boundary in API/test processes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import math
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals
from .providers import ProviderStatus
from .whole_video_analysis import TemporalEvent, WholeVideoContext


@dataclass(frozen=True)
class PerformanceFrame:
    source_asset_id: str
    timestamp: float
    face_visible: float
    pose_visible: float
    eye_contact_proxy: float
    motion: float
    expression: float | None = None
    body_x: float | None = None
    body_y: float | None = None
    left_wrist_x: float | None = None
    left_wrist_y: float | None = None
    right_wrist_x: float | None = None
    right_wrist_y: float | None = None


@dataclass(frozen=True)
class LocalPerformanceTimeline:
    source_asset_id: str
    observations: Tuple[PerformanceFrame, ...]
    events: Tuple[TemporalEvent, ...]
    sampled_fps: float
    source_fps: float
    status: ProviderStatus


@dataclass(frozen=True)
class LocalPerformanceResult:
    timelines: Tuple[LocalPerformanceTimeline, ...]
    status: ProviderStatus


def _pt(landmarks, index: int):
    try:
        p = landmarks.landmark[index]
        return float(p.x), float(p.y)
    except Exception:
        return None


def _mid(*points):
    valid = [p for p in points if p is not None]
    if not valid:
        return None
    return sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid)


def _centroid(landmarks):
    if landmarks is None:
        return None
    try:
        pts = [(float(p.x), float(p.y)) for p in landmarks.landmark]
        return _mid(*pts)
    except Exception:
        return None


def _dist(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _face(face):
    if face is None:
        return 0.0, None
    le, re, nose = _pt(face, 33), _pt(face, 263), _pt(face, 1)
    mt, mb = _pt(face, 13), _pt(face, 14)
    if le is None or re is None or nose is None:
        return 0.5, None
    span = max(0.02, _dist(le, re))
    eye_mid = _mid(le, re)
    yaw = abs(nose[0] - eye_mid[0]) / span if eye_mid else 0.0
    engagement = max(0.0, min(1.0, 1.0 - yaw / 0.75))
    expression = min(1.0, _dist(mt, mb) / span) if mt and mb else None
    return engagement, expression


def _body(pose):
    if pose is None:
        return None
    shoulders = _mid(_pt(pose, 11), _pt(pose, 12))
    hips = _mid(_pt(pose, 23), _pt(pose, 24))
    return _mid(shoulders, hips)


def detect_candidate_events(observations: Iterable[PerformanceFrame]) -> Tuple[TemporalEvent, ...]:
    frames = tuple(observations)
    output = []
    last: dict[str, float] = {}

    def emit(kind, a, b, confidence, description):
        if b.timestamp - last.get(kind, -999.0) < 0.18:
            return
        output.append(TemporalEvent(
            b.source_asset_id, max(0.0, a.timestamp), max(a.timestamp, b.timestamp),
            kind, max(0.0, min(1.0, confidence)), description,
        ))
        last[kind] = b.timestamp

    for a, b in zip(frames, frames[1:]):
        if a.face_visible and b.face_visible:
            drop = a.eye_contact_proxy - b.eye_contact_proxy
            if a.eye_contact_proxy >= 0.62 and b.eye_contact_proxy <= 0.38 and drop >= 0.28:
                emit("camera_disengagement_candidate", a, b, 0.58 + drop * 0.45,
                     "abrupt head/camera-engagement geometry change")
            if a.expression is not None and b.expression is not None:
                shift = abs(b.expression - a.expression)
                if shift >= 0.085:
                    emit("facial_expression_shift_candidate", a, b, 0.55 + shift * 2.0,
                         "abrupt facial-geometry change")

        pa = (a.body_x, a.body_y) if a.body_x is not None and a.body_y is not None else None
        pb = (b.body_x, b.body_y) if b.body_x is not None and b.body_y is not None else None
        body_delta = _dist(pa, pb)
        if body_delta >= 0.055 and b.motion >= 0.025:
            emit("body_reset_candidate", a, b, 0.55 + body_delta * 3.5 + b.motion,
                 "abrupt body-center movement; requires speech/retry context")

        lwa = (a.left_wrist_x, a.left_wrist_y) if a.left_wrist_x is not None else None
        lwb = (b.left_wrist_x, b.left_wrist_y) if b.left_wrist_x is not None else None
        rwa = (a.right_wrist_x, a.right_wrist_y) if a.right_wrist_x is not None else None
        rwb = (b.right_wrist_x, b.right_wrist_y) if b.right_wrist_x is not None else None
        hand_delta = max(_dist(lwa, lwb), _dist(rwa, rwb))
        if hand_delta >= 0.10 and b.motion >= 0.02:
            emit("hand_motion_reset_candidate", a, b, 0.52 + hand_delta * 2.4 + b.motion,
                 "abrupt hand trajectory change; may be gesture or reset")
    return tuple(output)


def analyze_source_local_performance(path: str, *, source_asset_id: str,
                                     target_fps: float = 12.0,
                                     max_frames: int = 9000) -> LocalPerformanceTimeline:
    try:
        cv2 = importlib.import_module("cv2")
        mp = importlib.import_module("mediapipe")
        holistic_api = mp.solutions.holistic
    except Exception as exc:
        return LocalPerformanceTimeline(source_asset_id, (), (), 0.0, 0.0,
            ProviderStatus("local_performance", True, False, "provider_unavailable", exc.__class__.__name__))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return LocalPerformanceTimeline(source_asset_id, (), (), 0.0, 0.0,
            ProviderStatus("local_performance", True, False, "provider_error", "VideoCaptureError"))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 0 or not math.isfinite(source_fps):
        source_fps = 30.0
    step = max(1, int(round(source_fps / max(2.0, min(target_fps, source_fps)))))
    effective_fps = source_fps / step
    frames, previous_gray, index, sampled = [], None, 0, 0

    try:
        with holistic_api.Holistic(static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, refine_face_landmarks=True,
                min_detection_confidence=0.45, min_tracking_confidence=0.45) as holistic:
            while sampled < max_frames:
                ok, image = cap.read()
                if not ok:
                    break
                if index % step:
                    index += 1
                    continue
                timestamp = index / source_fps
                h, w = image.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    image = cv2.resize(image, (640, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                raw_motion = 0.0 if previous_gray is None else float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
                previous_gray = gray
                result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                face = getattr(result, "face_landmarks", None)
                pose = getattr(result, "pose_landmarks", None)
                engagement, expression = _face(face)
                body = _body(pose)
                # Prefer actual Holistic hand landmarks; pose wrists are fallback.
                left = _centroid(getattr(result, "left_hand_landmarks", None)) or _pt(pose, 15)
                right = _centroid(getattr(result, "right_hand_landmarks", None)) or _pt(pose, 16)
                frames.append(PerformanceFrame(
                    source_asset_id, timestamp, 1.0 if face else 0.0, 1.0 if pose else 0.0,
                    engagement, min(1.0, raw_motion * 8.0), expression,
                    body[0] if body else None, body[1] if body else None,
                    left[0] if left else None, left[1] if left else None,
                    right[0] if right else None, right[1] if right else None,
                ))
                sampled += 1
                index += 1
    except Exception as exc:
        return LocalPerformanceTimeline(source_asset_id, tuple(frames), detect_candidate_events(frames),
            effective_fps, source_fps,
            ProviderStatus("local_performance", True, False, "provider_error", exc.__class__.__name__))
    finally:
        cap.release()

    return LocalPerformanceTimeline(source_asset_id, tuple(frames), detect_candidate_events(frames),
        effective_fps, source_fps, ProviderStatus("local_performance", True, True, "applied"))


def analyze_local_performance(local_paths: Mapping[str, str], *, target_fps: float = 12.0) -> LocalPerformanceResult:
    timelines = tuple(analyze_source_local_performance(path, source_asset_id=sid, target_fps=target_fps)
                      for sid, path in sorted(local_paths.items()))
    available = any(t.status.available for t in timelines)
    status = "applied" if available else (timelines[0].status.status if timelines else "not_requested")
    reason = None if available else (timelines[0].status.reason if timelines else "no_sources")
    return LocalPerformanceResult(timelines,
        ProviderStatus("local_performance", bool(timelines), available, status, reason))


def merge_local_events_into_context(context: WholeVideoContext,
                                    timelines: Iterable[LocalPerformanceTimeline]) -> WholeVideoContext:
    by_source = {t.source_asset_id: t for t in timelines}
    merged = []
    for source in context.sources:
        timeline = by_source.get(source.source_asset_id)
        if timeline is None:
            merged.append(source)
            continue
        known = {(e.kind, round(e.start, 3), round(e.end, 3)) for e in source.events}
        additions = tuple(e for e in timeline.events
                          if (e.kind, round(e.start, 3), round(e.end, 3)) not in known)
        merged.append(replace(source, events=tuple(sorted(source.events + additions,
                                                          key=lambda e: (e.start, e.end, e.kind)))))
    return replace(context, sources=tuple(merged))


def _mean(values, default=0.5):
    values = tuple(values)
    return sum(values) / len(values) if values else default


def apply_local_performance_to_takes(takes: Iterable[CandidateTake],
                                     timelines: Iterable[LocalPerformanceTimeline]) -> Tuple[CandidateTake, ...]:
    by_source = {t.source_asset_id: t for t in timelines}
    output = []
    for take in takes:
        timeline = by_source.get(take.source_asset_id)
        if timeline is None:
            output.append(take)
            continue
        frames = tuple(f for f in timeline.observations if take.start <= f.timestamp <= take.end)
        if not frames:
            output.append(take)
            continue
        events = tuple(e for e in timeline.events if e.end > take.start and e.start < take.end)
        duration = max(0.25, take.duration_sec)
        face = _mean((f.face_visible for f in frames), 0.0)
        eye = _mean((f.eye_contact_proxy for f in frames if f.face_visible), 0.5)
        motions = tuple(f.motion for f in frames)
        avg_motion = _mean(motions, 0.0)
        variance = _mean(((m - avg_motion) ** 2 for m in motions), 0.0)
        stability = max(0.0, min(1.0, 1.0 - variance * 18.0))
        body_n = sum(e.kind in {"body_reset_candidate", "hand_motion_reset_candidate"} for e in events)
        face_n = sum(e.kind == "facial_expression_shift_candidate" for e in events)
        disengage_n = sum(e.kind == "camera_disengagement_candidate" for e in events)
        local_fumble = min(0.85, (body_n + 0.6 * face_n) / max(1.0, duration * 1.5))
        local_expression = max(0.15, 1.0 - min(0.85, face_n / max(1.0, duration * 1.7)))
        local_gesture = max(0.15, 1.0 - min(0.85, body_n / max(1.0, duration * 1.4)))
        local_distraction = min(0.9, disengage_n / max(1.0, duration))
        base = take.signals or MediaSignals(take.source_asset_id, take.start, take.end)
        signals = replace(base,
            face_visibility=0.45 * base.face_visibility + 0.55 * face,
            eye_contact=0.45 * base.eye_contact + 0.55 * eye,
            motion_stability=0.45 * base.motion_stability + 0.55 * stability,
            visual_fumble=max(base.visual_fumble, local_fumble),
            expression_naturalness=0.55 * base.expression_naturalness + 0.45 * local_expression,
            gesture_naturalness=0.55 * base.gesture_naturalness + 0.45 * local_gesture,
            distraction_risk=max(base.distraction_risk, local_distraction),
        )
        output.append(replace(take, signals=signals))
    return tuple(output)
