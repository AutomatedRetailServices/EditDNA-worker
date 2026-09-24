"""D-097 §4 -- perceptual System Watch+Listen reviewer v1 (routing, explicit
4-state delivery gate).

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
- D-154 (Gate 6 second correction, real RAW #118 audit, explicit Product
  Owner acceptance criterion, replacing D-153's binary `blocks_delivery`
  with an explicit 4-state status): `PerceptualReview.watch_listen_status`
  (and `as_dict()["watch_listen_status"]`) is always exactly one of
  `WATCH_LISTEN_BLOCKED` / `WATCH_LISTEN_HUMAN_REVIEW_REQUIRED` /
  `WATCH_LISTEN_SYSTEM_PASS` / `WATCH_LISTEN_HUMAN_APPROVED`:
    * `EVALUATED_FAIL` on any capability -> `BLOCKED` (a confirmed
      perceptual defect; the same condition `overall_status` already uses
      for `REVIEW_FAIL`, and the same condition `benchmarks/
      clean_raw_gate.py` already independently checked before this
      correction -- naming an existing downstream behavior honestly).
    * `ERROR` on any capability -> `BLOCKED` too (an error means the
      measurement itself did not run -- there is no reliable evidence at
      all for that capability, which is at least as unsafe as a confirmed
      FAIL, never merely "uncertain"). D-153 wrongly treated ERROR the
      same as UNCERTAIN/NOT_IMPLEMENTED; corrected here.
    * `UNCERTAIN` or `NOT_IMPLEMENTED` (with nothing BLOCKED) ->
      `HUMAN_REVIEW_REQUIRED`: the render is NOT deleted or withheld from
      a human reviewer (this module never mutates the render either way --
      "diagnoses and routes" holds unchanged), but `SYSTEM_PASS`,
      "Ready", and automatic delivery are withheld. With 4 of 8
      capabilities still NOT_IMPLEMENTED today, this is the ordinary
      status for a real review, not an edge case.
    * `SYSTEM_PASS` only when EVERY capability in the v1 acceptance set is
      implemented, evaluated, AND EVALUATED_PASS -- i.e. nothing BLOCKED
      and nothing needs human review. This exists so a future version with
      every capability actually implemented can reach it; v1's 4
      NOT_IMPLEMENTED capabilities mean no real v1 review reaches it yet,
      which is intentional, not a bug to route around.
    * `HUMAN_APPROVED` is NEVER computed by this module -- it requires an
      explicit human decision this code has no way to observe on its own.
      `apply_human_watch_listen_approval()` below takes that external
      decision as an explicit argument and is the only way to reach it,
      and only ever promotes from `HUMAN_REVIEW_REQUIRED` or
      `SYSTEM_PASS` -- a `BLOCKED` review is a root-authority defect to
      fix or a render to re-attempt, never something a human "approves
      away" through this gate (this module's own "never mutates" contract
      would otherwise be laundered through human sign-off instead of a
      real fix).
  Technical QC and System Watch+Listen remain two separate gates -- this
  status is perceptual-only and never substitutes for the technical QC's
  own PASS/FAIL;
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

from .audio_silence import AUDIO_SILENCE_EVENT_KIND
from .take_grouping import retry_similarity

SCHEMA_VERSION = "cutsell.perceptual_watch_listen.v1"
GATE_MODE_STATE_MACHINE_V1 = "state_machine_v1_blocked_human_review_system_pass"

# D-154: the explicit 4-state System Watch+Listen delivery status. See the
# module docstring's D-154 section for the full precedence/rationale.
WATCH_LISTEN_BLOCKED = "BLOCKED"
WATCH_LISTEN_HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
WATCH_LISTEN_SYSTEM_PASS = "SYSTEM_PASS"
WATCH_LISTEN_HUMAN_APPROVED = "HUMAN_APPROVED"
WATCH_LISTEN_STATUSES = (
    WATCH_LISTEN_BLOCKED, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS, WATCH_LISTEN_HUMAN_APPROVED,
)

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
# D-149 (Gate 6 correction, real RAW #118 audit): a real audit found the
# actual 14 findings on a real render were ALL `hand_motion_reset_candidate`
# -- normal expressive gesture and microphone repositioning while speaking,
# not recording-process resets. `local_performance.py`'s own docstring is
# explicit that this event kind is pure kinematic measurement ("abrupt
# changes are emitted as `*_candidate` events so semantic/retry context
# remains authoritative") -- its `confidence` field is `0.52 + hand_delta*
# 2.4 + b.motion`, a MAGNITUDE score, not a probability the movement is a
# genuine reset. A large, fast, natural hand gesture during animated speech
# scores just as "confident" as an actual reset; `EDGE_DEBRIS_MIN_CONFIDENCE`
# cannot tell them apart on its own. See this constant's use below.
RESET_CANDIDATE_PAUSE_PROXIMITY_SEC = 0.50
REPEATED_CONTENT_SIMILARITY = 0.72
_RESET_KINDS = frozenset({
    "body_reset_candidate", "hand_motion_reset_candidate",
    "camera_disengagement_candidate", "facial_expression_shift_candidate",
    "retry_setup", "false_start", "wrong_take", "breaking_character",
})
# D-145 (Gate 6, Gap C): `_RESET_KINDS` mixes two structurally different
# evidence classes. `_RESET_VISUAL_CANDIDATE_KINDS` is specific, positive
# evidence of a physical artifact (a stumble, a hand/body reset, a camera
# bump) -- if it bleeds into the kept window, it is genuinely likely to be
# visible debris, so it stays a hard FAIL. `_RESET_EXPLICIT_MARKER_KINDS`
# is the SAME evidence class `attempt_reconstruction.py`'s own
# `_EXPLICIT_ATTEMPT_BREAK_KINDS` already treats elsewhere as the reason a
# cut boundary was correctly placed at that exact point -- a marker sitting
# inside the 0.35s edge window is at least as consistent with "this is
# exactly why the cut is here, and the timestamp has ordinary measurement
# slop" as it is with "residue leaked into the render", and this capability
# never decodes a rendered frame to tell the two apart (see its own `note`
# below). Reporting it as an unconditional FAIL misrepresents an unverified
# inference as a confirmed defect.
_RESET_VISUAL_CANDIDATE_KINDS = frozenset({
    "body_reset_candidate", "hand_motion_reset_candidate",
    "camera_disengagement_candidate", "facial_expression_shift_candidate",
})
_RESET_EXPLICIT_MARKER_KINDS = frozenset({
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

    @property
    def watch_listen_status(self) -> str:
        """D-154 (corrected by D-155 -- independent audit, empty-
        capabilities gap): the explicit, primary delivery-gating status --
        always exactly one of `WATCH_LISTEN_BLOCKED` / `WATCH_LISTEN_HUMAN_
        REVIEW_REQUIRED` / `WATCH_LISTEN_SYSTEM_PASS`. Never `WATCH_LISTEN_
        HUMAN_APPROVED` -- this computation has no way to observe a human
        decision; see `apply_human_watch_listen_approval()` for that.
        Precedence: BLOCKED (EVALUATED_FAIL or ERROR on any capability)
        outranks HUMAN_REVIEW_REQUIRED (UNCERTAIN or NOT_IMPLEMENTED on any
        capability, nothing BLOCKED) outranks SYSTEM_PASS (every capability
        EVALUATED_PASS). See the module docstring's D-154 section for the
        full rationale, especially why ERROR is BLOCKED, not merely
        uncertain -- an error means the measurement itself never ran.

        D-155: a review with NO capabilities at all (nothing was ever
        evaluated) must never fall through to SYSTEM_PASS -- `all(...)` over
        an empty sequence is vacuously True, which would otherwise silently
        treat "no measurement happened" as "everything passed". `status`
        (`overall_status`) already avoids this trap by requiring the
        `statuses` list to be non-empty; this property now matches that
        same fail-safe direction explicitly."""
        if not self.capabilities:
            return WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
        if any(c.status in (EVALUATED_FAIL, ERROR) for c in self.capabilities):
            return WATCH_LISTEN_BLOCKED
        if any(c.status in (UNCERTAIN, NOT_IMPLEMENTED) for c in self.capabilities):
            return WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
        return WATCH_LISTEN_SYSTEM_PASS

    @property
    def has_confirmed_blocking_defect(self) -> bool:
        """D-155 (independent audit correction, replacing D-154's
        ambiguously-named `blocks_delivery`): True only for
        `WATCH_LISTEN_BLOCKED` -- a confirmed EVALUATED_FAIL/ERROR defect.
        This is deliberately NOT the inverse of `allows_automatic_delivery`
        below: `HUMAN_REVIEW_REQUIRED` is neither a confirmed defect nor
        something that may auto-deliver -- collapsing the two into one
        boolean is exactly the bug an independent audit found (a
        downstream gate read a `blocks_delivery=False` for HUMAN_REVIEW_
        REQUIRED as "safe to pass"). Always check `watch_listen_status`
        directly, or both booleans together, never this one alone."""
        return self.watch_listen_status == WATCH_LISTEN_BLOCKED

    @property
    def allows_automatic_delivery(self) -> bool:
        """D-155: True only for `SYSTEM_PASS` or `HUMAN_APPROVED` -- the
        only two statuses where automatic delivery/"Ready" is permitted.
        False for BOTH `BLOCKED` and `HUMAN_REVIEW_REQUIRED` -- a caller
        that only checks `has_confirmed_blocking_defect is False` and
        treats that as "safe to deliver" reproduces the exact bug this
        property exists to prevent."""
        return self.watch_listen_status in (WATCH_LISTEN_SYSTEM_PASS, WATCH_LISTEN_HUMAN_APPROVED)

    def as_dict(self) -> dict:
        routing: dict[str, int] = {}
        for finding in self.findings:
            routing[finding.routes_to] = routing.get(finding.routes_to, 0) + 1
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "gate_mode": self.gate_mode,
            "watch_listen_status": self.watch_listen_status,
            "has_confirmed_blocking_defect": self.has_confirmed_blocking_defect,
            "allows_automatic_delivery": self.allows_automatic_delivery,
            "human_watch_listen_required": self.watch_listen_status != WATCH_LISTEN_HUMAN_APPROVED,
            "capabilities": [asdict(c) for c in self.capabilities],
            "capability_status_counts": {
                s: sum(1 for c in self.capabilities if c.status == s)
                for s in (EVALUATED_PASS, EVALUATED_FAIL, UNCERTAIN, NOT_IMPLEMENTED, ERROR)
            },
            "finding_count": len(self.findings),
            "routing": routing,
        }


def apply_human_watch_listen_approval(review: PerceptualReview, *, approved: bool) -> str:
    """D-154: the final HUMAN WATCH + LISTEN gate in the D-095 ladder
    (`... -> perceptual SYSTEM WATCH + LISTEN -> HUMAN WATCH + LISTEN`).
    Takes the human reviewer's explicit decision as an argument -- this
    module has no other way to observe one. Returns `WATCH_LISTEN_HUMAN_
    APPROVED` only when `approved` is True AND the review's own automated
    `watch_listen_status` is `HUMAN_REVIEW_REQUIRED` or `SYSTEM_PASS`;
    otherwise returns the review's unchanged automated status. A `BLOCKED`
    review is deliberately NEVER promotable here, approved or not -- a
    confirmed EVALUATED_FAIL/ERROR is a root-authority fix or a render
    re-attempt, never something a human sign-off launders through this
    gate (this module's own "diagnoses and routes, never mutates"
    contract)."""
    automated = review.watch_listen_status
    if approved and automated in (WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, WATCH_LISTEN_SYSTEM_PASS):
        return WATCH_LISTEN_HUMAN_APPROVED
    return automated


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


def _measured_pause_near(
    events: Sequence[dict], e_start: float, e_end: float, *, tolerance_sec: float,
) -> bool:
    """D-149: real, already-measured source silence (`audio_silence.py`'s
    `AUDIO_SILENCE_EVENT_KIND`, ffmpeg `silencedetect` -- the same
    `mp4_measured`-grade signal `_dead_air_on_mp4` above already trusts)
    overlapping `[e_start, e_end]` within `tolerance_sec` on either side.
    A genuine recording-process reset/retry is a creator stopping and
    restarting -- that pattern virtually always has at least a brief pause
    around it. Continuous, unbroken speech through the exact moment of an
    abrupt hand/body/face measurement is the signature of natural
    expressive gesture or a mid-sentence mic adjustment instead."""
    lo, hi = e_start - tolerance_sec, e_end + tolerance_sec
    for event in events:
        if str(event.get("kind") or "").strip().lower() != AUDIO_SILENCE_EVENT_KIND:
            continue
        p_start, p_end = float(event.get("start") or 0.0), float(event.get("end") or 0.0)
        if p_end >= lo and p_start <= hi:
            return True
    return False


def _spoken_words_for_segment(draft, segment) -> tuple[tuple[float, float], ...]:
    """D-291.10: the frozen draft's own word timings for this segment's clip
    (by clip id, then by parent semantic clip id for a physical fragment),
    in source seconds. Empty when the draft carries no words for it."""
    wanted = {str(segment.clip_id), str(getattr(segment, "parent_semantic_clip_id", "") or "")}
    for clip in getattr(draft, "selected", ()) or ():
        if str(clip.clip_id) in wanted:
            return tuple(
                (float(w.start), float(w.end)) for w in (getattr(clip, "words", ()) or ())
                if float(getattr(w, "end", 0.0)) > float(getattr(w, "start", 0.0))
            )
    return ()


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
        spoken = _spoken_words_for_segment(draft, segment)
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
            detail = {"clip_id": segment.clip_id, "edge": edge, "event_kind": kind,
                      "confidence": round(float(event.get("confidence") or 0.0), 3),
                      "source_start": round(e_start, 3), "source_end": round(e_end, 3)}
            if kind in _RESET_EXPLICIT_MARKER_KINDS:
                # D-145: an explicit recording-process-break marker at this
                # edge is unverified evidence of visible residue (this
                # capability never decodes rendered frames) -- it stays a
                # reported, BoundaryEngine-routed finding, but UNCERTAIN,
                # never a confirmed FAIL.
                severity = "UNCERTAIN"
            else:
                # D-149: a visual/motion reset CANDIDATE's `confidence` is a
                # kinematic magnitude score, not a genuine-reset probability
                # -- see `_measured_pause_near`'s docstring. Only a
                # candidate co-occurring with a REAL measured pause keeps
                # its hard FAIL; one landing mid continuous, unbroken speech
                # is far more consistent with natural expressive gesture or
                # a mic adjustment and downgrades to UNCERTAIN instead.
                pause_nearby = _measured_pause_near(
                    events, e_start, e_end, tolerance_sec=RESET_CANDIDATE_PAUSE_PROXIMITY_SEC,
                )
                detail["measured_pause_nearby"] = pause_nearby
                # D-291.10 (RAW #126: six FAILs, all 67 ms hand-motion
                # candidates at an edge with the clip's own pre/post-speech
                # pause "nearby", every one during a spoken word on the
                # frames -- the mic hand moving as speech starts or ends).
                # An edge event that overlaps a word the frozen clip itself
                # carries is happening DURING speech: a gesture, never
                # confirmed debris. UNCERTAIN, still reported and routed.
                during_word = any(w_start < e_end and w_end > e_start for w_start, w_end in spoken)
                detail["during_spoken_word"] = during_word
                severity = "FAIL" if (pause_nearby and not during_word) else "UNCERTAIN"
            findings.append(PerceptualFinding(
                name, RESET_DEBRIS_AT_EDGE, out_start, out_end, severity, ROUTE_BOUNDARY, detail,
            ))
    if not evidence_seen:
        return CapabilityReport(name, UNCERTAIN, "source_evidence_mapped", note="no local performance evidence available for the rendered sources")
    if any(f.severity == "FAIL" for f in findings):
        status = EVALUATED_FAIL
    elif findings:
        status = UNCERTAIN
    else:
        status = EVALUATED_PASS
    return CapabilityReport(name, status, "source_evidence_mapped", tuple(findings),
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
    return PerceptualReview(status=overall_status(capabilities), gate_mode=GATE_MODE_STATE_MACHINE_V1, capabilities=tuple(capabilities))


def error_review(reason: str) -> PerceptualReview:
    return PerceptualReview(
        status=REVIEW_UNCERTAIN, gate_mode=GATE_MODE_STATE_MACHINE_V1,
        capabilities=(CapabilityReport("reviewer", ERROR, "none", note=str(reason)[:200]),),
    )
