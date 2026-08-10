"""Dense local performance tracking for CutSell.

This layer is deliberately *evidence*, not an editor.  It decodes source video at
an adjustable cadence, derives face/body/hand/motion trajectories locally, and
turns abrupt changes into conservative candidate events.  Candidate events never
mean ``CUT`` by themselves; downstream temporal/editorial reasoning combines them
with speech, retry and whole-video context.

OpenCV and MediaPipe are imported lazily so API-only/test environments can use the
rest of CutSell without those worker-only dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals
from .providers import ProviderStatus
from .whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


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


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point(landmarks, index: int) -> tuple[float, float] | None:
    try:
        item = landmarks.landmark[index]
        return float(item.x), float(item.y)
    except Exception:
        return None


def _midpoint(*points: tuple[float, float] | None) -> tuple[float, float] | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return (
        sum(item[0] for item in valid) / len(valid),
        sum(item[1] for item in valid) / len(valid),
    )


def _face_features(face_landmarks) -> tuple[float, float | None]:
    """Return head-on engagement proxy and normalized mouth opening.

    These are intentionally simple geometric measurements, not claims about gaze,
    emotion or intent.  The reasoning layers decide what the trajectory means.
    """
    if face_landmarks is None:
        return 0.0, None
    left_eye = _point(face_landmarks, 33)
    right_eye = _point(face_landmarks, 263)
    nose = _point(face_landmarks, 1)
    mouth_top = _point(face_landmarks, 13)
    mouth_bottom = _point(face_landmarks, 14)
    if left_eye is None or right_eye is None or nose is None:
        return 0.5, None
    eye_span = max(0.02, _distance(left_eye, right_eye))
    eye_mid = _midpoint(left_eye, right_eye)
    yaw_proxy = abs(nose[0] - eye_mid[0]) / eye_span if eye_mid else 0.0
    engagement = max(0.0, min(1.0, 1.0 - yaw_proxy / 0.75))
    expression = None
    if mouth_top is not None and mouth_bottom is not None:
        expression = min(1.0, _distance(mouth_top, mouth_bottom) / eye_span)
    return engagement, expression


def _pose_features(pose_landmarks) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None]:
    if pose_landmarks is None:
        return None, None, None
    # MediaPipe Pose indices: shoulders 11/12, wrists 15/16, hips 23/24.
    shoulder = _midpoint(_point(pose_landmarks, 11), _point(pose_landmarks, 12))
    hip = _midpoint(_point(pose_landmarks, 23), _point(pose_landmarks, 24))
    body = _midpoint(shoulder, hip)
    left_wrist = _point(pose_landmarks, 15)
    right_wrist = _point(pose_landmarks, 16)
    return body, left_wrist, right_wrist


def detect_candidate_events(observations: Iterable[PerformanceFrame]) -> Tuple[TemporalEvent, ...]:
    """Find abrupt local changes without assigning editorial meaning.

    Event names deliberately end in ``_candidate``.  The destructive temporal
    editor does not treat these as automatic bad-performance events.
    """
    frames = tuple(observations)
    events = []
    cooldown: dict[str, float] = {}

    def emit(kind: str, previous: PerformanceFrame, current: PerformanceFrame, confidence: float, description: str) -> None:
        last = cooldown.get(kind, -999.0)
        if current.timestamp - last < 0.18:
            return
        events.append(TemporalEvent(
            source_asset_id=current.source_asset_id,
            start=max(0.0, previous.timestamp),
            end=max(previous.timestamp, current.timestamp),
            kind=kind,
            confidence=max(0.0, min(1.0, confidence)),
            description=description,
        ))
        cooldown[kind] = current.timestamp

    for previous, current in zip(frames, frames[1:]):
        if previous.face_visible and current.face_visible:
            drop = previous.eye_contact_proxy - current.eye_contact_proxy
            if previous.eye_contact_proxy >= 0.62 and current.eye_contact_proxy <= 0.38 and drop >= 0.28:
                emit(
                    "camera_disengagement_candidate", previous, current,
                    min(0.92, 0.58 + drop * 0.45),
                    "abrupt head/camera-engagement geometry change",
                )
            if previous.expression is not None and current.expression is not None:
                shift = abs(current.expression - previous.expression)
                if shift >= 0.085:
                    emit(
                        "facial_expression_shift_candidate", previous, current,
                        min(0.90, 0.55 + shift * 2.0),
                        "abrupt facial-geometry change",
                    )

        body_before = (previous.body_x, previous.body_y) if previous.body_x is not None and previous.body_y is not None else None
        body_after = (current.body_x, current.body_y) if current.body_x is not None and current.body_y is not None else None
        body_delta = _distance(body_before, body_after)
        if body_delta >= 0.055 and current.motion >= 0.025:
            emit(
                "body_reset_candidate", previous, current,
                min(0.94, 0.55 + body_delta * 3.5 + current.motion),
                "abrupt body-center movement; requires speech/retry context",
            )

        wrist_delta = max(
            _distance(
                (previous.left_wrist_x, previous.left_wrist_y) if previous.left_wrist_x is not None and previous.left_wrist_y is not None else None,
                (current.left_wrist_x, current.left_wrist_y) if current.left_wrist_x is not None and current.left_wrist_y is not None else None,
            ),
            _distance(
                (previous.right_wrist_x, previous.right_wrist_y) if previous.right_wrist_x is not None and previous.right_wrist_y is not None else None,
                (current.right_wrist_x, current.right_wrist_y) if current.right_wrist_x is not None and current.right_wrist_y is not None else None,
            ),
        )
        if wrist_delta >= 0.10 and current.motion >= 0.02:
            emit(
                "hand_motion_reset_candidate", previous, current,
                min(0.92, 0.52 + wrist_delta * 2.4 + current.motion),
                "abrupt wrist/hand trajectory change; may be gesture or reset",
            )

    return tuple(events)


def analyze_source_local_performance(
    path: str,
    *,
    source_asset_id: str,
    target_fps: float = 12.0,
    max_frames: int = 9000,
) -> LocalPerformanceTimeline:
    """Decode one source and build dense local face/body/hand/motion evidence."""
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except Exception as exc:
        return LocalPerformanceTimeline(
            source_asset_id, (), (), 0.0, 0.0,
            ProviderStatus("local_performance", True, False, "provider_unavailable", exc.__class__.__name__),
        )

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return LocalPerformanceTimeline(
            source_asset_id, (), (), 0.0, 0.0,
            ProviderStatus("local_performance", True, False, "provider_error", "VideoCaptureError"),
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0 or not math.isfinite(source_fps):
        source_fps = 30.0
    sample_fps = max(2.0, min(float(target_fps), source_fps))
    frame_step = max(1, int(round(source_fps / sample_fps)))
    effective_fps = source_fps / frame_step
    observations = []
    previous_gray = None
    frame_index = 0
    sampled = 0

    try:
        holistic_api = mp.solutions.holistic
        with holistic_api.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        ) as holistic:
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step:
                    frame_index += 1
                    continue

                timestamp = frame_index / source_fps
                small = frame
                height, width = frame.shape[:2]
                max_width = 640
                if width > max_width:
                    scale = max_width / float(width)
                    small = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                motion = 0.0
                if previous_gray is not None and previous_gray.shape == gray.shape:
                    motion = float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
                previous_gray = gray

                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                face = getattr(result, "face_landmarks", None)
                pose = getattr(result, "pose_landmarks", None)
                engagement, expression = _face_features(face)
                body, left_wrist, right_wrist = _pose_features(pose)
                observations.append(PerformanceFrame(
                    source_asset_id=source_asset_id,
                    timestamp=timestamp,
                    face_visible=1.0 if face is not None else 0.0,
                    pose_visible=1.0 if pose is not None else 0.0,
                    eye_contact_proxy=engagement,
                    motion=min(1.0, motion * 8.0),
                    expression=expression,
                    body_x=body[0] if body else None,
                    body_y=body[1] if body else None,
                    left_wrist_x=left_wrist[0] if left_wrist else None,
                    left_wrist_y=left_wrist[1] if left_wrist else None,
                    right_wrist_x=right_wrist[0] if right_wrist else None,
                    right_wrist_y=right_wrist[1] if right_wrist else None,
                ))
                sampled += 1
                frame_index += 1
    except Exception as exc:
        capture.release()
        return LocalPerformanceTimeline(
            source_asset_id, tuple(observations), detect_candidate_events(observations),
            effective_fps, source_fps,
            ProviderStatus("local_performance", True, False, "provider_error", exc.__class__.__name__),
        )
    finally:
        capture.release()

    return LocalPerformanceTimeline(
        source_asset_id=source_asset_id,
        observations=tuple(observations),
        events=detect_candidate_events(observations),
        sampled_fps=effective_fps,
        source_fps=source_fps,
        status=ProviderStatus("local_performance", True, True, "applied"),
    )


def analyze_local_performance(
    local_paths: Mapping[str, str],
    *,
    target_fps: float = 12.0,
) -> LocalPerformanceResult:
    timelines = tuple(
        analyze_source_local_performance(path, source_asset_id=source_id, target_fps=target_fps)
        for source_id, path in sorted(local_paths.items())
    )
    available = any(item.status.available for item in timelines)
    status = "applied" if available else (timelines[0].status.status if timelines else "not_requested")
    reason = None if available else (timelines[0].status.reason if timelines else "no_sources")
    return LocalPerformanceResult(
        timelines,
        ProviderStatus("local_performance", bool(timelines), available, status, reason),
    )


def merge_local_events_into_context(
    context: WholeVideoContext,
    timelines: Iterable[LocalPerformanceTimeline],
) -> WholeVideoContext:
    """Attach non-destructive dense candidate events to existing whole context."""
    by_source = {item.source_asset_id: item for item in timelines}
    if not context.sources:
        return context
    merged_sources = []
    for source in context.sources:
        timeline = by_source.get(source.source_asset_id)
        if timeline is None or not timeline.events:
            merged_sources.append(source)
            continue
        known = {(item.kind, round(item.start, 3), round(item.end, 3)) for item in source.events}
        additions = tuple(
            event for event in timeline.events
            if (event.kind, round(event.start, 3), round(event.end, 3)) not in known
        )
        merged_sources.append(replace(source, events=tuple(sorted(source.events + additions, key=lambda event: (event.start, event.end, event.kind)))))
    return replace(context, sources=tuple(merged_sources))


def _mean(values: Iterable[float], default: float = 0.5) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else default


def apply_local_performance_to_takes(
    takes: Iterable[CandidateTake],
    timelines: Iterable[LocalPerformanceTimeline],
) -> Tuple[CandidateTake, ...]:
    """Blend dense local measurements into take-level MediaSignals conservatively."""
    by_source = {item.source_asset_id: item for item in timelines}
    output = []
    for take in takes:
        timeline = by_source.get(take.source_asset_id)
        if timeline is None:
            output.append(take)
            continue
        frames = tuple(item for item in timeline.observations if take.start <= item.timestamp <= take.end)
        if not frames:
            output.append(take)
            continue
        events = tuple(item for item in timeline.events if item.end > take.start and item.start < take.end)
        duration = max(0.25, take.duration_sec)
        face_visibility = _mean((item.face_visible for item in frames), 0.0)
        eye_contact = _mean((item.eye_contact_proxy for item in frames if item.face_visible), 0.5)
        motions = tuple(item.motion for item in frames)
        motion_mean = _mean(motions, 0.0)
        motion_variance = _mean(((item - motion_mean) ** 2 for item in motions), 0.0)
        motion_stability = max(0.0, min(1.0, 1.0 - motion_variance * 18.0))
        body_events = sum(1 for item in events if item.kind in {"body_reset_candidate", "hand_motion_reset_candidate"})
        face_events = sum(1 for item in events if item.kind == "facial_expression_shift_candidate")
        disengagements = sum(1 for item in events if item.kind == "camera_disengagement_candidate")
        event_rate = min(1.0, (body_events + 0.6 * face_events) / max(1.0, duration * 1.5))
        local_fumble = min(0.85, event_rate)
        local_expression = max(0.15, 1.0 - min(0.85, face_events / max(1.0, duration * 1.7)))
        local_gesture = max(0.15, 1.0 - min(0.85, body_events / max(1.0, duration * 1.4)))
        local_distraction = min(0.9, disengagements / max(1.0, duration))

        base = take.signals or MediaSignals(take.source_asset_id, take.start, take.end)
        # Existing multimodal provider signals remain authoritative for semantics;
        # dense local measurements improve temporal coverage rather than replacing it.
        blended = replace(
            base,
            face_visibility=max(0.0, min(1.0, 0.45 * base.face_visibility + 0.55 * face_visibility)),
            eye_contact=max(0.0, min(1.0, 0.45 * base.eye_contact + 0.55 * eye_contact)),
            motion_stability=max(0.0, min(1.0, 0.45 * base.motion_stability + 0.55 * motion_stability)),
            visual_fumble=max(base.visual_fumble, local_fumble),
            expression_naturalness=max(0.0, min(1.0, 0.55 * base.expression_naturalness + 0.45 * local_expression)),
            gesture_naturalness=max(0.0, min(1.0, 0.55 * base.gesture_naturalness + 0.45 * local_gesture)),
            distraction_risk=max(base.distraction_risk, local_distraction),
        )
        output.append(replace(take, signals=blended))
    return tuple(output)
