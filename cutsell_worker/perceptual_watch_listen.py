"""D-097 §4 -- perceptual System Watch+Listen reviewer v1 (advisory, routing only).

D-096 root cause #5: the technical post-render QC catches silence / black /
frozen frames / join clicks and nothing perceptual, so every visual or
delivery defect passes straight to the human. This is the first version of
the perceptual reviewer the D-095 ladder places AFTER the technical QC:

    technical post-render QC -> perceptual SYSTEM WATCH + LISTEN -> HUMAN WATCH + LISTEN

Contract (PO adjustment §4):
- it DIAGNOSES and ROUTES; it never edits Selection, Boundary or the render;
- every capability reports one of EVALUATED_PASS / EVALUATED_FAIL /
  UNCERTAIN / NOT_IMPLEMENTED / ERROR; NOT_IMPLEMENTED, UNCERTAIN and ERROR
  never become PASS, and the review as a whole is PASS only when EVERY
  capability is EVALUATED_PASS -- so v1, which still carries
  NOT_IMPLEMENTED capabilities, can never auto-PASS a candidate;
- `gate_mode` is "advisory_v1": the technical QC remains the blocking
  delivery gate until the Product Owner approves this gate's acceptance
  criteria (D-096 Part 12 Step 4, escalation A). Until then a technically
  clean candidate is reported as DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN,
  never as an approved preview;
- measurements are made on the REAL rendered MP4 where the capability
  allows (dead air, cut-adjacent speech energy); capabilities that map
  source evidence (A-5 reset events) or transcript onto the render timeline
  say so in `method`, and an absent signal is UNCERTAIN, never clean.
No Video00 constant is referenced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Mapping, Sequence

from .take_grouping import retry_similarity

SCHEMA_VERSION = "cutsell.perceptual_watch_listen.v1"
GATE_MODE_ADVISORY_V1 = "advisory_v1"

EVALUATED_PASS = "EVALUATED_PASS"
EVALUATED_FAIL = "EVALUATED_FAIL"
UNCERTAIN = "UNCERTAIN"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
ERROR = "ERROR"

REVIEW_PASS = "PASS"
REVIEW_FAIL = "FAIL"
REVIEW_UNCERTAIN = "UNCERTAIN"

# Finding kinds (perceptual). Routing names are D-021 authorities.
INTERIOR_DEAD_AIR = "PERCEPTUAL_INTERIOR_DEAD_AIR"
LONG_PAUSE = "PERCEPTUAL_LONG_PAUSE"
SPEECH_ENERGY_AT_CUT = "PERCEPTUAL_SPEECH_ENERGY_AT_CUT"
RESET_DEBRIS_AT_EDGE = "PERCEPTUAL_RESET_DEBRIS_AT_EDGE"
REPEATED_AUDIENCE_CONTENT = "PERCEPTUAL_REPEATED_AUDIENCE_CONTENT"

ROUTE_BOUNDARY = "BoundaryEngine"
ROUTE_BEST_TAKE = "BestTakeResolver"
ROUTE_RENDERER = "Renderer"

DEAD_AIR_FAIL_SEC = 1.20
LONG_PAUSE_SEC = 0.80
CUT_ENERGY_WINDOW_SEC = 0.06
CUT_ENERGY_SPEECH_DBFS = -22.0
EDGE_DEBRIS_WINDOW_SEC = 0.35
EDGE_DEBRIS_MIN_CONFIDENCE = 0.85
REPEATED_CONTENT_SIMILARITY = 0.72
_RESET_KINDS = frozenset({
    "body_reset_candidate", "hand_motion_reset_candidate",
    "camera_disengagement_candidate", "facial_expression_shift_candidate",
    "retry_setup", "false_start", "wrong_take", "breaking_character",
})

NOT_IMPLEMENTED_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("facial_expression_post_line", "needs face/expression estimation on decoded frames"),
    ("gesture_continuity_across_cut", "needs pose tracking across the join on decoded frames"),
    ("clipped_phoneme_asr_realign", "needs ASR word/phone re-alignment on the rendered audio"),
    ("framing_and_eye_contact", "needs face/gaze estimation on decoded frames"),
)


@dataclass(frozen=True)
class PerceptualFinding:
    capability: str
    kind: str
    start: float
    end: float
    severity: str  # "FAIL" | "UNCERTAIN"
    routes_to: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityReport:
    capability: str
    status: str
    method: str  # "mp4_measured" | "source_evidence_mapped" | "transcript_derived" | "none"
    findings: tuple[PerceptualFinding, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class PerceptualReview:
    status: str
    gate_mode: str
    capabilities: tuple[CapabilityReport, ...]
    schema_version: str = SCHEMA_VERSION

    @property
    def findings(self) -> tuple[PerceptualFinding, ...]:
        return tuple(f for c in self.capabilities for f in c.findings)

    def as_dict(self) -> dict:
        routing: dict[str, int] = {}
        for finding in self.findings:
            routing[finding.routes_to] = routing.get(finding.routes_to, 0) + 1
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "gate_mode": self.gate_mode,
            "blocking": False,
            "human_watch_listen_required": True,
            "capabilities": [asdict(c) for c in self.capabilities],
            "capability_status_counts": {
                s: sum(1 for c in self.capabilities if c.status == s)
                for s in (EVALUATED_PASS, EVALUATED_FAIL, UNCERTAIN, NOT_IMPLEMENTED, ERROR)
            },
            "finding_count": len(self.findings),
            "routing": routing,
        }


def overall_status(capabilities: Iterable[CapabilityReport]) -> str:
    statuses = [c.status for c in capabilities]
    if any(s == EVALUATED_FAIL for s in statuses):
        return REVIEW_FAIL
    if all(s == EVALUATED_PASS for s in statuses) and statuses:
        return REVIEW_PASS
    return REVIEW_UNCERTAIN


# --- capabilities ----------------------------------------------------------------

def _dead_air_on_mp4(media_path: str) -> CapabilityReport:
    name = "interior_dead_air_mp4"
    try:
        from .post_render_media_qc import _detect_silence_intervals
        intervals = _detect_silence_intervals(media_path, noise_floor_db=-35.0, min_silence_sec=LONG_PAUSE_SEC)
    except Exception as exc:  # noqa: BLE001 -- reported, never hidden
        return CapabilityReport(name, ERROR, "mp4_measured", note=f"silencedetect failed: {str(exc)[:120]}")
    findings = []
    for start, end in intervals:
        duration = end - start
        if duration >= DEAD_AIR_FAIL_SEC:
            findings.append(PerceptualFinding(name, INTERIOR_DEAD_AIR, start, end, "FAIL", ROUTE_BOUNDARY, {"duration_sec": round(duration, 3)}))
        elif duration >= LONG_PAUSE_SEC:
            findings.append(PerceptualFinding(name, LONG_PAUSE, start, end, "UNCERTAIN", ROUTE_BOUNDARY, {"duration_sec": round(duration, 3)}))
    if any(f.severity == "FAIL" for f in findings):
        status = EVALUATED_FAIL
    elif findings:
        status = UNCERTAIN
    else:
        status = EVALUATED_PASS
    return CapabilityReport(name, status, "mp4_measured", tuple(findings))


def _rms_dbfs(samples) -> float:
    import numpy as np
    if samples.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return 20.0 * math.log10(max(rms, 1e-9) / 32768.0)


def _speech_energy_at_cuts(media_path: str, joins: Sequence[float]) -> CapabilityReport:
    name = "cut_adjacent_speech_energy_mp4"
    if not joins:
        return CapabilityReport(name, EVALUATED_PASS, "mp4_measured", note="single segment: no joins to evaluate")
    try:
        import numpy as np  # noqa: F401
        from .post_render_media_qc import _extract_pcm_window
    except Exception as exc:  # noqa: BLE001
        return CapabilityReport(name, ERROR, "mp4_measured", note=f"pcm decode unavailable: {str(exc)[:120]}")
    findings = []
    try:
        for join in joins:
            before = _extract_pcm_window(media_path, center_sec=max(0.0, join - CUT_ENERGY_WINDOW_SEC / 2), window_sec=CUT_ENERGY_WINDOW_SEC / 2)
            after = _extract_pcm_window(media_path, center_sec=join + CUT_ENERGY_WINDOW_SEC / 2, window_sec=CUT_ENERGY_WINDOW_SEC / 2)
            before_db, after_db = _rms_dbfs(before), _rms_dbfs(after)
            if before_db >= CUT_ENERGY_SPEECH_DBFS:
                findings.append(PerceptualFinding(
                    name, SPEECH_ENERGY_AT_CUT, join - CUT_ENERGY_WINDOW_SEC, join, "UNCERTAIN", ROUTE_BOUNDARY,
                    {"side": "outgoing", "rms_dbfs": round(before_db, 2)},
                ))
            if after_db >= CUT_ENERGY_SPEECH_DBFS:
                findings.append(PerceptualFinding(
                    name, SPEECH_ENERGY_AT_CUT, join, join + CUT_ENERGY_WINDOW_SEC, "UNCERTAIN", ROUTE_BOUNDARY,
                    {"side": "incoming", "rms_dbfs": round(after_db, 2)},
                ))
    except Exception as exc:  # noqa: BLE001
        return CapabilityReport(name, ERROR, "mp4_measured", tuple(findings), note=f"decode failed: {str(exc)[:120]}")
    return CapabilityReport(name, UNCERTAIN if findings else EVALUATED_PASS, "mp4_measured", tuple(findings),
                            note="speech-level energy right at a join may be a clipped word; needs ASR re-alignment to confirm")


def _events_for_source(diagnostics: Mapping, source_asset_id: str) -> tuple[dict, ...] | None:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(e for e in (source.get("events") or ()) if isinstance(e, dict))
    return None


def _reset_debris_at_edges(draft, segments: Sequence, output_windows: Sequence[tuple[float, float]]) -> CapabilityReport:
    name = "reset_debris_at_edges_source_evidence"
    diagnostics = dict(getattr(draft, "diagnostics", None) or {})
    findings = []
    evidence_seen = False
    for segment, (win_start, win_end) in zip(segments, output_windows):
        events = _events_for_source(diagnostics, segment.source_asset_id)
        if events is None:
            continue
        evidence_seen = True
        seg_start, seg_end = float(segment.start), float(segment.end)
        for event in events:
            kind = str(event.get("kind") or "").strip().lower()
            if kind not in _RESET_KINDS or float(event.get("confidence") or 0.0) < EDGE_DEBRIS_MIN_CONFIDENCE:
                continue
            e_start, e_end = float(event.get("start") or 0.0), float(event.get("end") or 0.0)
            at_entry = e_start < seg_start + EDGE_DEBRIS_WINDOW_SEC and e_end > seg_start
            at_exit = e_end > seg_end - EDGE_DEBRIS_WINDOW_SEC and e_start < seg_end
            if not (at_entry or at_exit):
                continue
            edge = "entry" if at_entry else "exit"
            out_start = win_start if at_entry else max(win_start, win_end - EDGE_DEBRIS_WINDOW_SEC)
            out_end = min(win_end, win_start + EDGE_DEBRIS_WINDOW_SEC) if at_entry else win_end
            findings.append(PerceptualFinding(
                name, RESET_DEBRIS_AT_EDGE, out_start, out_end, "FAIL", ROUTE_BOUNDARY,
                {"clip_id": segment.clip_id, "edge": edge, "event_kind": kind,
                 "confidence": round(float(event.get("confidence") or 0.0), 3),
                 "source_start": round(e_start, 3), "source_end": round(e_end, 3)},
            ))
    if not evidence_seen:
        return CapabilityReport(name, UNCERTAIN, "source_evidence_mapped", note="no local performance evidence available for the rendered sources")
    return CapabilityReport(name, EVALUATED_FAIL if findings else EVALUATED_PASS, "source_evidence_mapped", tuple(findings),
                            note="A-5 reset/break events mapped onto the render timeline; not re-measured on decoded frames")


def _repeated_audience_content(draft, segments: Sequence, output_windows: Sequence[tuple[float, float]]) -> CapabilityReport:
    name = "repeated_audience_content_transcript"
    text_by_clip = {c.clip_id: str(c.text or "") for c in getattr(draft, "selected", ()) or ()}
    parents = []
    for segment in segments:
        parent = getattr(segment, "parent_semantic_clip_id", None) or segment.clip_id
        parents.append((parent, text_by_clip.get(segment.clip_id) or text_by_clip.get(parent) or ""))
    if not any(text for _, text in parents):
        return CapabilityReport(name, UNCERTAIN, "transcript_derived", note="no clip text available for the rendered segments")
    findings = []
    for index in range(len(parents) - 1):
        left_parent, left_text = parents[index]
        right_parent, right_text = parents[index + 1]
        if left_parent == right_parent or not left_text or not right_text:
            continue
        similarity = retry_similarity(left_text, right_text)
        if similarity >= REPEATED_CONTENT_SIMILARITY:
            findings.append(PerceptualFinding(
                name, REPEATED_AUDIENCE_CONTENT, output_windows[index][0], output_windows[index + 1][1], "FAIL", ROUTE_BEST_TAKE,
                {"left_clip_id": segments[index].clip_id, "right_clip_id": segments[index + 1].clip_id, "similarity": round(similarity, 3)},
            ))
    return CapabilityReport(name, EVALUATED_FAIL if findings else EVALUATED_PASS, "transcript_derived", tuple(findings),
                            note="adjacent rendered pieces from different semantic clips saying the same thing")


def review_rendered_candidate(
    media_path: str,
    draft,
    segments: Sequence,
    output_windows: Sequence[tuple[float, float]],
) -> PerceptualReview:
    """Run every v1 capability against the rendered candidate and return the
    routed review. Never raises: a capability that cannot run reports ERROR."""
    joins = [w[1] for w in output_windows[:-1]]
    capabilities = [
        _dead_air_on_mp4(media_path),
        _speech_energy_at_cuts(media_path, joins),
        _reset_debris_at_edges(draft, segments, output_windows),
        _repeated_audience_content(draft, segments, output_windows),
        *(CapabilityReport(cap, NOT_IMPLEMENTED, "none", note=why) for cap, why in NOT_IMPLEMENTED_CAPABILITIES),
    ]
    return PerceptualReview(status=overall_status(capabilities), gate_mode=GATE_MODE_ADVISORY_V1, capabilities=tuple(capabilities))


def error_review(reason: str) -> PerceptualReview:
    return PerceptualReview(
        status=REVIEW_UNCERTAIN, gate_mode=GATE_MODE_ADVISORY_V1,
        capabilities=(CapabilityReport("reviewer", ERROR, "none", note=str(reason)[:200]),),
    )
