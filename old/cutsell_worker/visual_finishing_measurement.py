"""Visual Finishing MEASUREMENT FOUNDATION (D-258).

D-257 audited CutSell's current visual capabilities and found real,
already-proven primitives (MediaPipe Holistic in `local_performance.py`,
MediaPipe FaceMesh/Pose in `speech_visual_microtrim.py`) alongside
genuine gaps: no face bbox, no face-position/scale cross-clip
measurement, no brightness/luma, no color statistics, no cross-clip
visual-join measurement. This module is the first gate that actually
builds those missing MEASUREMENTS -- and measurements ONLY.

    VISUAL MEASUREMENT (this module) -> FUTURE VISUAL POLICY ->
    FUTURE VISUAL PLAN -> RENDERER EXECUTION -> POST-RENDER VISUAL QC

## Scope discipline (binding, D-258's own scope banner)

NO CROP. NO REFRAME. NO PUNCH-IN. NO EXPOSURE CORRECTION. NO COLOR
CORRECTION. NO VISUAL POLICY. NO RENDER POLICY CHANGE. NO NEW VISUAL
THRESHOLD. NO NEW HEURISTIC. This module:

- never calls ffmpeg to MODIFY a frame (only ffprobe-equivalent reads
  via `probe_media`, and read-only ffmpeg filter passes it delegates to
  `post_render_media_qc.py`'s existing `check_frozen_frames`/
  `check_dead_black_frames` for reference evidence, never recomputing
  their logic itself);
- never decides GOOD/BAD, never emits a PASS/FAIL verdict on any
  measured value -- every number is reported as a fact, with no target,
  band, or threshold attached anywhere in this file;
- never mutates `render.py`, `local_performance.py`,
  `speech_visual_microtrim.py`, or `post_render_media_qc.py` -- it is a
  new, additive, standalone owner module, exactly as D-257 Stage 1 and
  D-258 Stage 1 both require;
- opens its OWN MediaPipe FaceMesh/Pose instances (matching the
  existing convention: `local_performance.py` and
  `speech_visual_microtrim.py` already each open and close their own
  instances independently -- there is no single shared "MediaPipe
  owner" in this codebase to violate) and never imports the private
  (underscore-prefixed) internals of either existing module;
- bounds every failure to an explicit `MEASUREMENT_STATUS_*` value --
  never raises an unbounded exception for a normal malformed-media case
  (mirrors `audio_finishing_measurement.py`'s own discipline exactly).

## Testability design

Every geometric/statistical computation is a pure function of already-
extracted primitives (a list of `(x, y)` landmark points, a decoded
numpy pixel array, plain floats) -- NONE of them import `cv2` or
`mediapipe` in their own signature or body. Only the top-level
`measure_visual_clip` integration function touches `cv2`/`mediapipe`
(both dynamically imported, matching this repo's existing lazy-import
convention), so the deterministic math is fully unit-testable without
either library installed -- exactly the pattern `local_performance.py`'s
own test suite already uses (`test_cutsell_local_performance.py` tests
`detect_candidate_events`/`apply_local_performance_to_takes` via
directly-constructed `PerformanceFrame` objects, never by actually
invoking the real cv2/mediapipe decode path).
"""
from __future__ import annotations

import importlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Sequence

from .media_probe import probe_media
from .post_render_media_qc import check_dead_black_frames, check_frozen_frames
from .post_render_watch_listen_qc import PostRenderQCResult

MEASUREMENT_VERSION = "visual_finishing_measurement.v1"

# ---------------------------------------------------------------------------
# STAGE 21: bounded status vocabulary. Mirrors D-247's
# `audio_finishing_measurement.py` MEASUREMENT_STATUS_* naming exactly.
# ---------------------------------------------------------------------------

MEASUREMENT_STATUS_COMPLETE = "COMPLETE"
MEASUREMENT_STATUS_PARTIAL = "PARTIAL"
MEASUREMENT_STATUS_NO_FACE = "NO_FACE"
MEASUREMENT_STATUS_UNAVAILABLE = "UNAVAILABLE"
MEASUREMENT_STATUS_DECODE_ERROR = "DECODE_ERROR"
MEASUREMENT_STATUS_MEASUREMENT_ERROR = "MEASUREMENT_ERROR"

# ---------------------------------------------------------------------------
# STAGE 18: orientation categories -- factual classification, no policy.
# ---------------------------------------------------------------------------

ORIENTATION_PORTRAIT = "PORTRAIT"
ORIENTATION_LANDSCAPE = "LANDSCAPE"
ORIENTATION_SQUARE = "SQUARE"
ORIENTATION_UNKNOWN = "UNKNOWN"

# STAGE 16: no product detector exists (D-257 finding) -- this status is
# the only value this module will ever report for product_bbox_status.
PRODUCT_BBOX_UNAVAILABLE = "UNAVAILABLE"

# STAGE 17: no caption-safe/UI-safe policy is established by this gate.
CAPTION_SAFE_NOT_ESTABLISHED = "NOT_ESTABLISHED"

_DEFAULT_SAMPLE_COUNT = 8


# ---------------------------------------------------------------------------
# STAGE 2: result types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualFrameMeasurement:
    """One sampled frame's factual visual state. Every field is either a
    real measured value or an explicit `None` -- never a fabricated
    default (mirrors `AudioFinishingMeasurement`'s own discipline)."""

    source_id: str | None
    clip_id: str | None
    frame_timestamp_sec: float

    frame_width: int | None
    frame_height: int | None

    face_detected: bool
    face_count: int
    multiple_faces_detected: bool
    # STAGE 4: MediaPipe FaceMesh's `process()` output (`multi_face_
    # landmarks`) carries NO per-face detection-confidence score --
    # `min_detection_confidence` is only an INPUT threshold to the
    # constructor, never an output value. Audited directly against the
    # actual MediaPipe FaceMesh API surface (confirmed: neither
    # `local_performance.py` nor `speech_visual_microtrim.py` stores or
    # returns any such value either). This field is therefore always
    # `None` in this version -- never a landmark-count proxy invented to
    # fill the gap, per this gate's own explicit instruction.
    face_confidence: float | None

    face_bbox_x_min: float | None
    face_bbox_y_min: float | None
    face_bbox_x_max: float | None
    face_bbox_y_max: float | None
    face_center_x: float | None
    face_center_y: float | None
    face_bbox_width: float | None
    face_bbox_height: float | None
    face_area_ratio: float | None
    headroom_ratio: float | None

    face_bbox_clipped_left: bool | None
    face_bbox_clipped_right: bool | None
    face_bbox_clipped_top: bool | None
    face_bbox_clipped_bottom: bool | None

    # STAGE 8: torso/pose center is a DISTINCT concept from face center --
    # never collapsed into it. `None` when Pose evidence is unavailable.
    torso_center_x: float | None
    torso_center_y: float | None

    luma_mean: float | None
    luma_std: float | None
    color_mean_r: float | None
    color_mean_g: float | None
    color_mean_b: float | None

    measurement_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualClipMeasurement:
    """Aggregated, factual clip-level visual state across sampled frames."""

    source_id: str | None
    clip_id: str | None

    frame_width: int | None
    frame_height: int | None
    aspect_ratio: float | None
    orientation: str
    rotation_degrees: float | None

    requested_frame_count: int
    valid_frame_count: int
    face_valid_frame_count: int
    face_detection_rate: float | None

    face_center_x_median: float | None
    face_center_y_median: float | None
    face_area_ratio_median: float | None
    headroom_ratio_median: float | None

    luma_mean_median: float | None
    # STAGE 13: two DISTINCT, both-real concepts, never collapsed --
    # `contrast_median` is the clip-typical INTRA-frame contrast (the
    # median of each sampled frame's own `luma_std`, STAGE 10's literal
    # metric); `luma_variability` is the CROSS-frame brightness stability
    # across the clip's own sampled frames (how much the overall scene
    # brightness itself drifts from frame to frame -- a temporal signal,
    # not a per-frame contrast measurement).
    contrast_median: float | None
    luma_variability: float | None
    color_mean_r_median: float | None
    color_mean_g_median: float | None
    color_mean_b_median: float | None

    product_bbox_status: str
    caption_safe_status: str

    freeze_frame_evidence: PostRenderQCResult | None
    black_frame_evidence: PostRenderQCResult | None

    frames: tuple[VisualFrameMeasurement, ...]

    measurement_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VisualJoinMeasurement:
    """A pure, measurement-only comparison of two explicit adjacent
    clips. No GOOD/BAD decision, no punch-in decision, no crop
    decision -- factual deltas only."""

    left_clip_id: str | None
    right_clip_id: str | None

    face_center_dx: float | None
    face_center_dy: float | None
    face_area_ratio_delta: float | None
    headroom_delta: float | None
    luma_delta: float | None
    contrast_delta: float | None
    color_delta: float | None

    measurement_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Pure geometry/statistics helpers -- no cv2/mediapipe import anywhere in
# this section, fully unit-testable without either library installed.
# ---------------------------------------------------------------------------

def _orientation_category(width: int | None, height: int | None) -> str:
    if not width or not height:
        return ORIENTATION_UNKNOWN
    if width == height:
        return ORIENTATION_SQUARE
    return ORIENTATION_PORTRAIT if height > width else ORIENTATION_LANDSCAPE


def face_bbox_from_landmark_points(points: Sequence[tuple[float, float]]) -> dict | None:
    """STAGE 3/6/15: derive a normalized face bbox + center + area ratio +
    headroom + edge-clipping flags from a plain sequence of `(x, y)`
    points already normalized to `[0, 1]` by the caller (MediaPipe's own
    landmark convention). Pure geometry -- no detector call, no policy.
    Returns `None` for an empty/degenerate input (never fabricates a
    bbox from nothing)."""
    pts = list(points)
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    return {
        "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
        "center_x": (x_min + x_max) / 2.0, "center_y": (y_min + y_max) / 2.0,
        "width": width, "height": height,
        # Both the bbox and the frame are normalized to [0, 1], so
        # width*height IS already the bbox's area as a fraction of the
        # whole frame -- no separate pixel-dimension conversion needed.
        "area_ratio": width * height,
        # STAGE 6: literal geometric distance from the top of the face
        # bbox to the top of the frame, normalized by frame height --
        # since y is already frame-height-normalized, this is y_min
        # itself. No good/bad threshold attached.
        "headroom_ratio": y_min,
        # STAGE 15: literal frame-boundary contact only -- no safe-margin
        # tolerance invented. A bbox coordinate at or beyond the [0, 1]
        # edge is "clipped"; MediaPipe often simply fails to detect a
        # face at all when too much of it is out of frame (face_detected
        # becomes False) rather than reporting an out-of-bounds bbox, so
        # this condition may rarely fire in practice -- recorded honestly,
        # not engineered to fire more often than the real geometry does.
        "clipped_left": x_min <= 0.0,
        "clipped_right": x_max >= 1.0,
        "clipped_top": y_min <= 0.0,
        "clipped_bottom": y_max >= 1.0,
    }


def torso_center_from_landmark_points(
    left_shoulder: tuple[float, float] | None,
    right_shoulder: tuple[float, float] | None,
    left_hip: tuple[float, float] | None,
    right_hip: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """STAGE 8: torso/pose centroid -- a DISTINCT concept from face
    center, never collapsed into it. Mirrors the midpoint-of-shoulders/
    midpoint-of-hips geometry `local_performance.py`'s own `_body()`
    helper uses (a public, well-known MediaPipe Pose landmark-index
    convention, not a reuse of that module's private function)."""
    points = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p is not None]
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def luma_stats_from_gray_array(gray) -> tuple[float, float]:
    """STAGE 9/10: mean + standard deviation of an already-converted
    grayscale (luma) pixel array. `gray` is any array-like numpy ndarray
    (BT.601-weighted luma if produced via `cv2.cvtColor(..., COLOR_BGR2GRAY)`,
    a standard, deterministic, well-documented conversion -- this
    function itself performs no color-space conversion, only the
    statistics, so it needs no cv2 import at all). Contrast metric
    (STAGE 10) is literally `luma_std`, the directive's own suggested
    example -- no new metric invented."""
    import numpy as np

    arr = np.asarray(gray, dtype="float64")
    return float(arr.mean()), float(arr.std())


def color_stats_from_bgr_array(bgr) -> tuple[float, float, float]:
    """STAGE 11: compact mean-per-channel color summary (mean R, mean G,
    mean B) of an already-decoded BGR pixel array (OpenCV's own native
    channel order) -- no white-balance correction, no "warm/cool is bad"
    judgment, just the three channel means."""
    import numpy as np

    arr = np.asarray(bgr, dtype="float64")
    mean_b = float(arr[..., 0].mean())
    mean_g = float(arr[..., 1].mean())
    mean_r = float(arr[..., 2].mean())
    return mean_r, mean_g, mean_b


def default_sample_timestamps(duration_sec: float, count: int = _DEFAULT_SAMPLE_COUNT) -> tuple[float, ...]:
    """STAGE 12: deterministic, bounded, evenly-spaced sample timestamps.
    This is measurement MECHANICS (how many frames to look at), never a
    product-quality policy -- `count` is an ordinary implementation
    parameter with a modest default, not a visual-quality threshold.
    Samples stay inside `[0.05, 0.95]` of the duration to avoid the
    exact first/last frame, which is more prone to encoder/decoder edge
    artifacts unrelated to genuine visual content."""
    count = max(1, int(count))
    duration = max(0.0, float(duration_sec))
    if duration <= 0.0:
        return (0.0,)
    if count == 1:
        return (duration / 2.0,)
    start, end = duration * 0.05, duration * 0.95
    step = (end - start) / (count - 1)
    return tuple(start + i * step for i in range(count))


def _face_confidence_placeholder() -> None:
    """STAGE 4: documents, in code, that no confidence value is ever
    invented here -- always `None`. Kept as a named function (rather
    than an inline `None`) so a future gate that DOES gain a real
    confidence signal has exactly one place to change."""
    return None


def build_frame_measurement(
    *,
    source_id: str | None,
    clip_id: str | None,
    frame_timestamp_sec: float,
    frame_width: int | None,
    frame_height: int | None,
    face_landmark_points_list: Sequence[Sequence[tuple[float, float]]] = (),
    torso_points: tuple[
        tuple[float, float] | None, tuple[float, float] | None,
        tuple[float, float] | None, tuple[float, float] | None,
    ] | None = None,
    luma_mean: float | None = None,
    luma_std: float | None = None,
    color_rgb: tuple[float, float, float] | None = None,
) -> VisualFrameMeasurement:
    """The pure builder every integration path funnels through. Accepts
    already-extracted primitives only -- `face_landmark_points_list` is
    zero-or-more faces, each a plain sequence of `(x, y)` points (this
    is exactly what a caller gets by reading `landmark.x`/`landmark.y`
    off MediaPipe's own `multi_face_landmarks`, done ONCE at the
    integration boundary in `measure_visual_clip`, never inside this
    function). STAGE 5: when more than one face is supplied, the
    LARGEST-area face (by bbox) becomes the reported "primary" face --
    a documented, deterministic tie-break, not an arbitrary pick."""
    faces = [face_bbox_from_landmark_points(pts) for pts in face_landmark_points_list]
    faces = [f for f in faces if f is not None]
    face_count = len(faces)
    primary = max(faces, key=lambda f: f["area_ratio"]) if faces else None

    torso = None
    if torso_points is not None:
        torso = torso_center_from_landmark_points(*torso_points)

    return VisualFrameMeasurement(
        source_id=source_id, clip_id=clip_id, frame_timestamp_sec=frame_timestamp_sec,
        frame_width=frame_width, frame_height=frame_height,
        face_detected=primary is not None, face_count=face_count,
        multiple_faces_detected=face_count > 1,
        face_confidence=_face_confidence_placeholder(),
        face_bbox_x_min=None if primary is None else primary["x_min"],
        face_bbox_y_min=None if primary is None else primary["y_min"],
        face_bbox_x_max=None if primary is None else primary["x_max"],
        face_bbox_y_max=None if primary is None else primary["y_max"],
        face_center_x=None if primary is None else primary["center_x"],
        face_center_y=None if primary is None else primary["center_y"],
        face_bbox_width=None if primary is None else primary["width"],
        face_bbox_height=None if primary is None else primary["height"],
        face_area_ratio=None if primary is None else primary["area_ratio"],
        headroom_ratio=None if primary is None else primary["headroom_ratio"],
        face_bbox_clipped_left=None if primary is None else primary["clipped_left"],
        face_bbox_clipped_right=None if primary is None else primary["clipped_right"],
        face_bbox_clipped_top=None if primary is None else primary["clipped_top"],
        face_bbox_clipped_bottom=None if primary is None else primary["clipped_bottom"],
        torso_center_x=None if torso is None else torso[0],
        torso_center_y=None if torso is None else torso[1],
        luma_mean=luma_mean, luma_std=luma_std,
        color_mean_r=None if color_rgb is None else color_rgb[0],
        color_mean_g=None if color_rgb is None else color_rgb[1],
        color_mean_b=None if color_rgb is None else color_rgb[2],
        measurement_status=MEASUREMENT_STATUS_COMPLETE,
        errors=(),
    )


# ---------------------------------------------------------------------------
# STAGE 13: clip-level aggregation -- pure, given already-built frames.
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_clip_measurement(
    frames: Sequence[VisualFrameMeasurement],
    *,
    source_id: str | None,
    clip_id: str | None,
    frame_width: int | None,
    frame_height: int | None,
    rotation_degrees: float | None,
    requested_frame_count: int,
    freeze_frame_evidence: PostRenderQCResult | None,
    black_frame_evidence: PostRenderQCResult | None,
    provenance: dict | None = None,
) -> VisualClipMeasurement:
    frames = tuple(frames)
    valid = [f for f in frames if f.measurement_status not in (MEASUREMENT_STATUS_DECODE_ERROR,)]
    valid_frame_count = len(valid)
    face_frames = [f for f in valid if f.face_detected]
    face_valid_frame_count = len(face_frames)

    face_detection_rate = face_valid_frame_count / valid_frame_count if valid_frame_count else None
    face_center_x_median = _median([f.face_center_x for f in face_frames if f.face_center_x is not None])
    face_center_y_median = _median([f.face_center_y for f in face_frames if f.face_center_y is not None])
    face_area_ratio_median = _median([f.face_area_ratio for f in face_frames if f.face_area_ratio is not None])
    headroom_ratio_median = _median([f.headroom_ratio for f in face_frames if f.headroom_ratio is not None])

    luma_values = [f.luma_mean for f in valid if f.luma_mean is not None]
    luma_mean_median = _median(luma_values)
    # STAGE 13: clip-level variability is the spread of per-frame means
    # ACROSS the sampled frames (distinct from a single frame's own
    # intra-frame `luma_std`, i.e. that frame's own contrast).
    luma_variability = statistics.pstdev(luma_values) if len(luma_values) > 1 else (0.0 if luma_values else None)
    # STAGE 10/13: clip-typical contrast -- the median of each sampled
    # frame's OWN intra-frame luma_std, distinct from `luma_variability`
    # above (which measures brightness drift ACROSS frames, not
    # within one).
    contrast_median = _median([f.luma_std for f in valid if f.luma_std is not None])

    color_r_median = _median([f.color_mean_r for f in valid if f.color_mean_r is not None])
    color_g_median = _median([f.color_mean_g for f in valid if f.color_mean_g is not None])
    color_b_median = _median([f.color_mean_b for f in valid if f.color_mean_b is not None])

    errors = tuple(e for f in frames for e in f.errors)

    # STAGE 21: deterministic status ladder, checked in a fixed, tested
    # order -- never ambiguous for the same inputs.
    if valid_frame_count == 0:
        status = MEASUREMENT_STATUS_DECODE_ERROR
    elif valid_frame_count == requested_frame_count and face_valid_frame_count == 0:
        status = MEASUREMENT_STATUS_NO_FACE
    elif valid_frame_count < requested_frame_count:
        status = MEASUREMENT_STATUS_PARTIAL
    else:
        status = MEASUREMENT_STATUS_COMPLETE

    return VisualClipMeasurement(
        source_id=source_id, clip_id=clip_id,
        frame_width=frame_width, frame_height=frame_height,
        aspect_ratio=(frame_width / frame_height) if frame_width and frame_height else None,
        orientation=_orientation_category(frame_width, frame_height),
        rotation_degrees=rotation_degrees,
        requested_frame_count=requested_frame_count,
        valid_frame_count=valid_frame_count,
        face_valid_frame_count=face_valid_frame_count,
        face_detection_rate=face_detection_rate,
        face_center_x_median=face_center_x_median,
        face_center_y_median=face_center_y_median,
        face_area_ratio_median=face_area_ratio_median,
        headroom_ratio_median=headroom_ratio_median,
        luma_mean_median=luma_mean_median,
        contrast_median=contrast_median,
        luma_variability=luma_variability,
        color_mean_r_median=color_r_median,
        color_mean_g_median=color_g_median,
        color_mean_b_median=color_b_median,
        product_bbox_status=PRODUCT_BBOX_UNAVAILABLE,
        caption_safe_status=CAPTION_SAFE_NOT_ESTABLISHED,
        freeze_frame_evidence=freeze_frame_evidence,
        black_frame_evidence=black_frame_evidence,
        frames=frames,
        measurement_status=status,
        errors=errors,
        provenance=dict(provenance) if provenance else {},
    )


# ---------------------------------------------------------------------------
# STAGE 14: visual join measurement -- pure, given two clip measurements.
# ---------------------------------------------------------------------------

def compute_visual_join_measurement(
    left: VisualClipMeasurement, right: VisualClipMeasurement,
) -> VisualJoinMeasurement:
    def _delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else b - a

    face_center_dx = _delta(left.face_center_x_median, right.face_center_x_median)
    face_center_dy = _delta(left.face_center_y_median, right.face_center_y_median)
    face_area_ratio_delta = _delta(left.face_area_ratio_median, right.face_area_ratio_median)
    headroom_delta = _delta(left.headroom_ratio_median, right.headroom_ratio_median)
    luma_delta = _delta(left.luma_mean_median, right.luma_mean_median)
    contrast_delta = _delta(left.contrast_median, right.contrast_median)

    color_delta = None
    left_rgb = (left.color_mean_r_median, left.color_mean_g_median, left.color_mean_b_median)
    right_rgb = (right.color_mean_r_median, right.color_mean_g_median, right.color_mean_b_median)
    if all(v is not None for v in left_rgb) and all(v is not None for v in right_rgb):
        color_delta = math.sqrt(sum((r - l) ** 2 for l, r in zip(left_rgb, right_rgb)))

    deltas = (face_center_dx, face_center_dy, face_area_ratio_delta, headroom_delta,
              luma_delta, contrast_delta, color_delta)
    computed = sum(1 for d in deltas if d is not None)
    if computed == 0:
        status = MEASUREMENT_STATUS_UNAVAILABLE
    elif computed < len(deltas):
        status = MEASUREMENT_STATUS_PARTIAL
    else:
        status = MEASUREMENT_STATUS_COMPLETE

    errors = tuple(left.errors) + tuple(right.errors)
    return VisualJoinMeasurement(
        left_clip_id=left.clip_id, right_clip_id=right.clip_id,
        face_center_dx=face_center_dx, face_center_dy=face_center_dy,
        face_area_ratio_delta=face_area_ratio_delta, headroom_delta=headroom_delta,
        luma_delta=luma_delta, contrast_delta=contrast_delta, color_delta=color_delta,
        measurement_status=status, errors=errors,
    )


# ---------------------------------------------------------------------------
# Integration boundary: the ONLY functions in this module that touch
# cv2/mediapipe/ffprobe subprocess I/O. Dynamically imported, bounded,
# never raise for a normal malformed-media case.
# ---------------------------------------------------------------------------

def _probe_rotation_degrees(path: str) -> float | None:
    """STAGE 18: read rotation metadata if present (old-style `rotate`
    stream tag or a `side_data_list` display-matrix rotation), via one
    bounded, list-arg ffprobe call. Returns `None` (never fabricated)
    when absent or unreadable -- most files this pipeline produces
    carry no rotation tag at all, which is a real, honestly-reported
    fact, not a measurement failure."""
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
                "-of", "json", path,
            ],
            capture_output=True, text=True, check=True, timeout=30.0,
        )
    except Exception:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return float(tags["rotate"])
        except (TypeError, ValueError):
            return None
    for entry in stream.get("side_data_list") or []:
        if "rotation" in entry:
            try:
                return float(entry["rotation"])
            except (TypeError, ValueError):
                return None
    return None


def _import_cv2_mediapipe():
    cv2 = importlib.import_module("cv2")
    mp = importlib.import_module("mediapipe")
    return cv2, mp


def _extract_face_points(face_landmarks) -> list[tuple[float, float]]:
    return [(float(p.x), float(p.y)) for p in face_landmarks.landmark]


def _extract_torso_points(pose_landmarks):
    if pose_landmarks is None:
        return None
    pts = pose_landmarks.landmark
    if len(pts) <= 24:
        return None

    def _visible(index: int):
        p = pts[index]
        if float(getattr(p, "visibility", 0.0) or 0.0) < 0.45:
            return None
        return (float(p.x), float(p.y))

    # Public MediaPipe Pose landmark indices (11/12 shoulders, 23/24
    # hips) -- a documented, well-known convention, independently
    # re-derived here, not a reuse of any other module's private code.
    return (_visible(11), _visible(12), _visible(23), _visible(24))


def measure_visual_clip(
    path: str,
    *,
    timestamps: Sequence[float] | None = None,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    source_id: str | None = None,
    clip_id: str | None = None,
    max_faces: int = 4,
    freeze_frame_evidence: PostRenderQCResult | None = None,
    black_frame_evidence: PostRenderQCResult | None = None,
    compute_frame_quality_evidence: bool = False,
) -> VisualClipMeasurement:
    """STAGE 1/12: the single public integration entry point. Opens its
    own bounded cv2 `VideoCapture` + MediaPipe `FaceMesh`/`Pose`
    instances (closed in a `finally` block; no shared global state, no
    cross-job cache -- STAGE 25). `compute_frame_quality_evidence`
    defaults to `False`: black/freeze-frame detection is a full-file
    ffmpeg decode pass (`post_render_media_qc.py`'s own existing,
    UNCHANGED authority) and this module never forces that extra cost
    on every measurement call -- a caller that already ran it elsewhere
    in the pipeline should pass the result in via `freeze_frame_
    evidence`/`black_frame_evidence` instead (STAGE 19's own "reference
    additively" instruction)."""
    try:
        probe = probe_media(path)
    except Exception as exc:
        return aggregate_clip_measurement(
            (), source_id=source_id, clip_id=clip_id, frame_width=None, frame_height=None,
            rotation_degrees=None, requested_frame_count=0,
            freeze_frame_evidence=freeze_frame_evidence, black_frame_evidence=black_frame_evidence,
            provenance={"error": f"probe_media failed: {exc.__class__.__name__}"},
        )

    rotation_degrees = _probe_rotation_degrees(path)
    resolved_timestamps = tuple(timestamps) if timestamps is not None else default_sample_timestamps(
        probe.duration_sec, sample_count,
    )
    requested_frame_count = len(resolved_timestamps)

    if compute_frame_quality_evidence:
        if freeze_frame_evidence is None:
            freeze_frame_evidence = check_frozen_frames(path)
        if black_frame_evidence is None:
            black_frame_evidence = check_dead_black_frames(path)

    try:
        cv2, mp = _import_cv2_mediapipe()
    except Exception as exc:
        return aggregate_clip_measurement(
            (), source_id=source_id, clip_id=clip_id,
            frame_width=probe.width or None, frame_height=probe.height or None,
            rotation_degrees=rotation_degrees, requested_frame_count=requested_frame_count,
            freeze_frame_evidence=freeze_frame_evidence, black_frame_evidence=black_frame_evidence,
            provenance={"status_reason": "cv2/mediapipe unavailable", "error": exc.__class__.__name__},
        )
    frames: list[VisualFrameMeasurement] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return aggregate_clip_measurement(
            (), source_id=source_id, clip_id=clip_id,
            frame_width=probe.width or None, frame_height=probe.height or None,
            rotation_degrees=rotation_degrees, requested_frame_count=requested_frame_count,
            freeze_frame_evidence=freeze_frame_evidence, black_frame_evidence=black_frame_evidence,
            provenance={"status_reason": "VideoCaptureError"},
        )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=max(1, int(max_faces)),
        refine_landmarks=False, min_detection_confidence=0.5,
    )
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    try:
        for timestamp in resolved_timestamps:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
                ok, bgr_frame = cap.read()
                if not ok or bgr_frame is None:
                    frames.append(_frame_with_status(
                        build_frame_measurement(
                            source_id=source_id, clip_id=clip_id, frame_timestamp_sec=timestamp,
                            frame_width=probe.width or None, frame_height=probe.height or None,
                        ),
                        MEASUREMENT_STATUS_DECODE_ERROR,
                        (f"frame_read_failed@{timestamp:.3f}",),
                    ))
                    continue
                h, w = bgr_frame.shape[:2]
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                luma_mean, luma_std = luma_stats_from_gray_array(gray)
                color_rgb = color_stats_from_bgr_array(bgr_frame)
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                face_result = face_mesh.process(rgb_frame)
                pose_result = pose.process(rgb_frame)
                face_landmarks_list = getattr(face_result, "multi_face_landmarks", None) or ()
                points_list = [_extract_face_points(fl) for fl in face_landmarks_list]
                torso_points = _extract_torso_points(getattr(pose_result, "pose_landmarks", None))
                frames.append(build_frame_measurement(
                    source_id=source_id, clip_id=clip_id, frame_timestamp_sec=timestamp,
                    frame_width=w, frame_height=h,
                    face_landmark_points_list=points_list, torso_points=torso_points,
                    luma_mean=luma_mean, luma_std=luma_std, color_rgb=color_rgb,
                ))
            except Exception as exc:
                frames.append(_frame_with_status(
                    build_frame_measurement(
                        source_id=source_id, clip_id=clip_id, frame_timestamp_sec=timestamp,
                        frame_width=probe.width or None, frame_height=probe.height or None,
                    ),
                    MEASUREMENT_STATUS_DECODE_ERROR,
                    (f"frame_measurement_error@{timestamp:.3f}:{exc.__class__.__name__}",),
                ))
    finally:
        face_mesh.close()
        pose.close()
        cap.release()

    return aggregate_clip_measurement(
        frames, source_id=source_id, clip_id=clip_id,
        frame_width=probe.width or None, frame_height=probe.height or None,
        rotation_degrees=rotation_degrees, requested_frame_count=requested_frame_count,
        freeze_frame_evidence=freeze_frame_evidence, black_frame_evidence=black_frame_evidence,
    )


def _frame_with_status(
    frame: VisualFrameMeasurement, status: str, extra_errors: tuple[str, ...],
) -> VisualFrameMeasurement:
    """Small immutable-update helper (frozen dataclasses use
    `dataclasses.replace`-style construction throughout this codebase;
    this local helper avoids importing `replace` twice for one field
    pair)."""
    from dataclasses import replace as _replace

    return _replace(frame, measurement_status=status, errors=frame.errors + extra_errors)
