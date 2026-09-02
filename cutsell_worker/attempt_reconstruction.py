"""Reconstruct creator delivery attempts before Clean Cut deletion.

Whisper/ASR boundaries are transcription boundaries, not editorial takes.  This module
combines nearby speech fragments into the larger delivery attempt a human editor would
judge, while keeping retry/reset boundaries separate so Best Take can compare complete
attempts rather than isolated fragments.

The policy is deliberately conservative: uncertain neighboring speech stays together.
A boundary is created only from a real pause, a hard creator/session discontinuity,
strong multi-family recording-process evidence, or clear lexical restart evidence.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals
from .session_boundaries import infer_session_boundaries
from .source_identity import stable_clip_id
from .whole_video_analysis import TemporalEvent, WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_TERMINAL_PUNCT_RE = re.compile(r"[.!?…][\"')\]]*$")
_STOP = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your",
    # Spanish
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "es", "esta",
    "este", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "o", "para",
    "pero", "por", "porque", "que", "se", "si", "sin", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})

_RESET_KINDS = frozenset({
    "body_reset", "body_reset_candidate", "hand_reset", "hand_motion_reset_candidate",
})
_CAMERA_KINDS = frozenset({
    "camera_disengagement", "camera_disengagement_candidate", "camera_adjustment",
})
_FACE_KINDS = frozenset({
    "facial_expression_shift", "facial_expression_shift_candidate", "breaking_character",
})
_EXPLICIT_ATTEMPT_BREAK_KINDS = frozenset({
    "retry_setup", "wrong_take", "false_start", "frustration", "recording_joke",
    "accidental_laughter", "searching_for_words", "verbal_fumble", "visual_fumble",
    "product_handling_mistake",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 3 and token not in _STOP)


def _source_events(context: WholeVideoContext | None, source_asset_id: str) -> tuple[TemporalEvent, ...]:
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _events_near_transition(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    padding_sec: float = 0.36,
) -> tuple[TemporalEvent, ...]:
    start = min(left.end, right.start) - padding_sec
    end = max(left.end, right.start) + padding_sec
    return tuple(
        event
        for event in _source_events(context, left.source_asset_id)
        if event.end >= start and event.start <= end
    )


def _hard_session_boundary_between(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    tolerance_sec: float = 0.14,
) -> bool:
    if context is None:
        return False
    for boundary in infer_session_boundaries(context, left.source_asset_id):
        if left.end - tolerance_sec <= boundary.at <= right.start + tolerance_sec:
            return True
    return False


def _restart_evidence(left_text: str, right_text: str) -> bool:
    """Detect an immediate verbal restart without classifying broad topic similarity."""
    left = _content_tokens(left_text)
    right = _content_tokens(right_text)
    if len(left) < 2 or len(right) < 2:
        return False

    # The same meaningful opening is the strongest common creator retry pattern.
    if left[:2] == right[:2]:
        return True

    # A creator can restart the last phrase of a longer attempt rather than the whole
    # paragraph. Require a three-token ordered phrase to avoid topic-only collisions.
    if len(right) >= 3:
        needle = right[:3]
        tail = left[-10:]
        for index in range(max(0, len(tail) - 2)):
            if tail[index : index + 3] == needle:
                return True
    return False


def _short_incomplete_suffix(left: CandidateTake, right: CandidateTake) -> bool:
    """Keep a tiny unfinished tail separate from an already usable delivery.

    ASR can fluctuate between runs and attach the last broken three-second self-correction
    directly to the preceding good paragraph. Once those pieces are merged, Hybrid sees
    one long winner and can no longer remove only the bad tail. Preserve the boundary
    when the preceding candidate is already complete and the new candidate is a short,
    explicitly incomplete suffix. If more speech follows, that suffix can still merge
    forward into its continuation on the next reconstruction step.
    """
    if not left.complete_idea or right.complete_idea:
        return False
    tokens = _tokens(right.text)
    if not tokens or len(tokens) > 10 or right.duration_sec > 4.5:
        return False
    gap = max(0.0, right.start - left.end)
    return gap <= 0.35


def _tiny_nonterminal_continuation(
    left: CandidateTake,
    right: CandidateTake,
    *,
    max_continuation_gap_sec: float,
) -> bool:
    """Protect a tiny lexical tail from a false visual-reset boundary.

    Dense body/hand reset detectors often fire during a normal breath at the end of a
    sentence. If Whisper split the final one or two words into a new candidate while the
    preceding text is visibly non-terminal, Best Take must judge the completed delivery,
    not the truncated prefix and suffix as competing takes. Hard session walls, lexical
    restarts and explicit recording-process breaks are still evaluated separately.
    """
    if left.source_asset_id != right.source_asset_id:
        return False
    gap = max(0.0, right.start - left.end)
    if gap > min(max_continuation_gap_sec, 1.20):
        return False
    right_tokens = _tokens(right.text)
    if not 1 <= len(right_tokens) <= 2 or right.duration_sec > 1.80:
        return False
    if len(_tokens(left.text)) < 4:
        return False
    left_text = str(left.text or "").rstrip()
    if not left_text or _TERMINAL_PUNCT_RE.search(left_text):
        return False
    return True


def _attempt_boundary_reason(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    max_continuation_gap_sec: float,
) -> str | None:
    if left.source_asset_id != right.source_asset_id:
        return "source_change"

    gap = max(0.0, right.start - left.end)
    if _hard_session_boundary_between(context, left, right):
        return "hard_session_boundary"

    # Lexical restarts must remain separate candidates so retry grouping/Best Take can
    # compare them. Require a non-zero word boundary so overlapping ASR duplicates are
    # not joined into one self-repeating sentence either.
    if right.start >= left.end - 0.03 and _restart_evidence(left.text, right.text):
        return "lexical_restart"

    # Do not let a short unfinished final self-correction poison a preceding coherent
    # paragraph merely because Whisper made the two chunks nearly contiguous.  This is
    # deliberately directional: complete -> short incomplete.  An incomplete fragment
    # can still merge forward into later continuation speech.
    if _short_incomplete_suffix(left, right):
        return "short_incomplete_suffix"

    # A one/two-word tail after non-terminal text is more reliable speech-continuation
    # evidence than an isolated body/hand reset candidate. Keep it attached before local
    # performance evidence is allowed to split the attempt.
    if _tiny_nonterminal_continuation(
        left,
        right,
        max_continuation_gap_sec=max_continuation_gap_sec,
    ):
        return None

    nearby = _events_near_transition(context, left, right)
    strong = tuple(event for event in nearby if float(event.confidence) >= 0.80)
    kinds = {_kind(event.kind) for event in strong}
    has_reset = bool(kinds & _RESET_KINDS)
    has_camera = bool(kinds & _CAMERA_KINDS)
    has_face = bool(kinds & _FACE_KINDS)

    if any(
        _kind(event.kind) in _EXPLICIT_ATTEMPT_BREAK_KINDS and float(event.confidence) >= 0.90
        for event in nearby
    ):
        return "explicit_recording_process_break"

    # A reset plus an independent disengagement/face change is the talking-head pattern
    # for "finished this try, preparing to say it again". One reset alone is not enough.
    if has_reset and (has_camera or has_face):
        return "multi_family_delivery_reset"

    # With an actual pause, one exceptionally strong reset is sufficient corroboration.
    if gap >= 0.65 and any(
        _kind(event.kind) in _RESET_KINDS and float(event.confidence) >= 0.95
        for event in nearby
    ):
        return "pause_plus_strong_reset"

    if gap > max_continuation_gap_sec:
        return "real_speech_pause"
    return None


def _merge_signals(members: tuple[CandidateTake, ...]) -> MediaSignals | None:
    signaled = [member for member in members if member.signals is not None]
    if not signaled:
        return None
    total = sum(max(0.001, member.duration_sec) for member in members)

    def weighted(name: str, default: float) -> float:
        numerator = 0.0
        for member in members:
            duration = max(0.001, member.duration_sec)
            value = getattr(member.signals, name) if member.signals is not None else default
            numerator += float(value) * duration
        return numerator / total

    return MediaSignals(
        source_asset_id=members[0].source_asset_id,
        start=members[0].start,
        end=members[-1].end,
        silence_ratio=weighted("silence_ratio", 0.0),
        audio_quality=weighted("audio_quality", 0.5),
        face_visibility=weighted("face_visibility", 0.5),
        eye_contact=weighted("eye_contact", 0.5),
        framing_quality=weighted("framing_quality", 0.5),
        product_visibility=weighted("product_visibility", 0.0),
        motion_stability=weighted("motion_stability", 0.5),
        continuity=weighted("continuity", 0.5),
        visual_fumble=max(
            (member.signals.visual_fumble if member.signals is not None else 0.0)
            for member in members
        ),
        expression_naturalness=weighted("expression_naturalness", 0.5),
        gesture_naturalness=weighted("gesture_naturalness", 0.5),
        delivery_energy=weighted("delivery_energy", 0.5),
        distraction_risk=max(
            (member.signals.distraction_risk if member.signals is not None else 0.0)
            for member in members
        ),
    )


def _merge_attempt(members: tuple[CandidateTake, ...]) -> CandidateTake:
    if len(members) == 1:
        return members[0]
    text = " ".join(member.text.strip() for member in members if member.text.strip()).strip()
    start = members[0].start
    end = members[-1].end
    return CandidateTake(
        clip_id=stable_clip_id(members[0].source_asset_id, start, end, text),
        source_asset_id=members[0].source_asset_id,
        source_order=members[0].source_order,
        start=start,
        end=end,
        text=text,
        words=tuple(word for member in members for word in member.words),
        signals=_merge_signals(members),
        # Completeness belongs to the tail of the delivery. A complete earlier sentence
        # must not hide an unfinished final fragment.
        complete_idea=members[-1].complete_idea,
    )


# D-046 FIX B (Invariant 2 -- good subspan preservation): a merge decision is
# correct on average (`max_continuation_gap_sec`'s job), but D-045 Case B
# showed a borderline internal gap can fuse two ALREADY-independently-
# complete deliveries into one physical attempt. When that fused attempt
# later loses its Best Take contest as a whole, any semantically-usable
# content that existed only in one half is destroyed even though nothing
# downstream did anything wrong -- BestTake/DeliveryScorer correctly picked
# the stronger candidate; they were just never shown the good half on its
# own. See D-045's docs/CUTSELL_DECISIONS.md entry for the full trace
# (0.76s gap, same audio, separate attempts in the prior passing run).
#
# This does not change the merge decision itself (Option A -- tightening
# the boundary rule -- was evaluated and rejected: "both sides already
# complete_idea" is common in ordinary multi-sentence delivery, so gating a
# real boundary on it alone would re-split large amounts of currently-
# correct continuous speech; a global fragment explosion, not a targeted
# fix). Instead (Option B) it ADDS, never replaces: when a merged bucket
# contains an internal gap that is a real pause (at least
# ``subspan_preservation_min_gap_sec``) between two members BOTH already
# independently judged ``complete_idea`` -- the same asymmetry-free
# condition that makes a gap "borderline" rather than an obvious same-
# breath continuation -- the two halves are additionally reconstructed as
# their own standalone candidates and appended to the returned pool. The
# fused attempt is always still returned too, unchanged, so a genuinely
# correct merge (the fused attempt wins) is completely unaffected; only a
# LOSING fused attempt's independently-valid subspan gets a second chance
# at survival, through the existing IdeaClusterer/BestTake/ClaimCoverage
# machinery -- no new selection logic, no Video00-specific text, and at
# most one extra split per merged bucket (the single widest qualifying
# internal gap), so this cannot fragment-explode a long multi-member
# bucket with several ordinary internal pauses.
_SUBSPAN_MIN_CONTENT_TOKENS = 3


def _borderline_split_index(
    bucket: tuple[CandidateTake, ...], *, min_gap_sec: float, max_gap_sec: float,
) -> int | None:
    """The single widest internal gap in `bucket` that is a real pause
    (`min_gap_sec` <= gap < `max_gap_sec` -- already below the merge
    threshold, or `_attempt_boundary_reason` would have split it) between
    two members that were each already independently judged a complete
    idea, or None if no member pair qualifies. Both halves must also carry
    enough of their own content to be worth preserving as a standalone
    candidate -- guards against recovering a bare filler word."""
    best_index: int | None = None
    best_gap = -1.0
    for index in range(1, len(bucket)):
        left = bucket[index - 1]
        right = bucket[index]
        if not (left.complete_idea and right.complete_idea):
            continue
        gap = max(0.0, float(right.start) - float(left.end))
        if not (min_gap_sec <= gap < max_gap_sec):
            continue
        if len(_content_tokens(left.text)) < _SUBSPAN_MIN_CONTENT_TOKENS:
            continue
        if len(_content_tokens(right.text)) < _SUBSPAN_MIN_CONTENT_TOKENS:
            continue
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return best_index


def _preserve_borderline_subspans(
    buckets: list[list[CandidateTake]],
    *,
    subspan_preservation_min_gap_sec: float,
    max_continuation_gap_sec: float,
) -> tuple[list[CandidateTake], list[dict]]:
    preserved: list[CandidateTake] = []
    audit: list[dict] = []
    for bucket in buckets:
        if len(bucket) < 2:
            continue
        members = tuple(bucket)
        split_index = _borderline_split_index(
            members,
            min_gap_sec=subspan_preservation_min_gap_sec,
            max_gap_sec=max_continuation_gap_sec,
        )
        if split_index is None:
            continue
        prefix = _merge_attempt(members[:split_index])
        suffix = _merge_attempt(members[split_index:])
        parent = _merge_attempt(members)
        preserved.extend((prefix, suffix))
        gap = max(0.0, float(members[split_index].start) - float(members[split_index - 1].end))
        audit.append({
            "authority": "attempt_reconstruction_subspan_preservation",
            "decision": "preserve_borderline_subspans",
            "parent_clip_id": parent.clip_id,
            "parent_text": parent.text,
            "prefix_clip_id": prefix.clip_id,
            "prefix_text": prefix.text,
            "suffix_clip_id": suffix.clip_id,
            "suffix_text": suffix.text,
            "gap_sec": round(gap, 3),
        })
    return preserved, audit


def reconstruct_delivery_attempts(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    max_continuation_gap_sec: float = 1.20,
    subspan_preservation_min_gap_sec: float = 0.55,
) -> tuple[Tuple[CandidateTake, ...], dict]:
    """Build paragraph/delivery-level candidates from ASR-level candidates.

    The function never deletes speech. It only changes the unit that later Clean Cut and
    Best Take judge. This is intentionally fail-open: absent a clear boundary, nearby
    fragments remain one complete creator delivery attempt.

    D-046 FIX B: the returned pool may also include a small number of extra
    "preserved subspan" candidates -- see `_preserve_borderline_subspans`'s
    docstring above. These are additive only; every attempt the merge logic
    itself produces is still returned unchanged.
    """
    ordered = tuple(sorted(
        takes,
        key=lambda take: (take.source_order, take.start, take.end, take.clip_id),
    ))
    if not ordered:
        return (), {
            "input_take_count": 0,
            "attempt_count": 0,
            "merged_fragment_count": 0,
            "boundaries": [],
            "attempts": [],
        }

    buckets: list[list[CandidateTake]] = []
    boundaries: list[dict] = []
    for take in ordered:
        if not buckets:
            buckets.append([take])
            continue

        left = buckets[-1][-1]
        reason = _attempt_boundary_reason(
            context,
            left,
            take,
            max_continuation_gap_sec=max_continuation_gap_sec,
        )
        if reason is not None:
            boundaries.append({
                "after_clip_id": left.clip_id,
                "before_clip_id": take.clip_id,
                "at": round((float(left.end) + float(take.start)) / 2.0, 3),
                "gap_sec": round(max(0.0, float(take.start) - float(left.end)), 3),
                "reason": reason,
            })
            buckets.append([take])
            continue
        buckets[-1].append(take)

    attempts = tuple(_merge_attempt(tuple(bucket)) for bucket in buckets)
    # D-046 FIX B: audit-only preview of what preserved_subspan_candidates()
    # below would recover -- kept purely informational here (JSON-safe text,
    # no live CandidateTake objects) so `attempts`/`boundaries` stay BYTE-
    # IDENTICAL to their pre-FIX-B shape. That matters concretely:
    # attempt_boundary_integrity.py's terminal-connector-guard monkeypatch
    # wraps this exact function and indexes `attempts[-1]`/`attempts[-2]`
    # assuming one entry per merged bucket in bucket order -- mixing extra
    # candidates into this tuple would silently corrupt that indexing. The
    # actual extra candidates are produced by the separate, explicit
    # `preserved_subspan_candidates()` call below, which every caller opts
    # into deliberately (flow_b.py does, right after this call).
    _, preserved_audit = _preserve_borderline_subspans(
        buckets,
        subspan_preservation_min_gap_sec=subspan_preservation_min_gap_sec,
        max_continuation_gap_sec=max_continuation_gap_sec,
    )
    diagnostics = {
        "input_take_count": len(ordered),
        "attempt_count": len(attempts),
        "merged_fragment_count": max(0, len(ordered) - len(attempts)),
        "boundaries": boundaries[:300],
        "attempts": [
            {
                "clip_id": attempt.clip_id,
                "source_asset_id": attempt.source_asset_id,
                "start": round(attempt.start, 3),
                "end": round(attempt.end, 3),
                "duration_sec": round(attempt.duration_sec, 3),
                "member_clip_ids": [member.clip_id for member in bucket],
                "member_count": len(bucket),
                "complete_idea": attempt.complete_idea,
            }
            for attempt, bucket in zip(attempts, buckets)
        ][:300],
        # D-046 FIX B: additive-only borderline-merge subspan recovery --
        # empty whenever no bucket contains a qualifying internal gap, so
        # this key is a no-op for every currently-passing fixture.
        "preserved_borderline_subspans": preserved_audit[:300],
    }
    return attempts, diagnostics


def preserved_subspan_candidates(
    takes: Iterable[CandidateTake],
    diagnostics: Mapping[str, object],
    *,
    subspan_preservation_min_gap_sec: float = 0.55,
    max_continuation_gap_sec: float = 1.20,
) -> tuple[Tuple[CandidateTake, ...], list[dict]]:
    """D-046 FIX B integration point: given the ORIGINAL pre-reconstruction
    takes and the diagnostics `reconstruct_delivery_attempts` already
    produced for them (specifically `attempts[*].member_clip_ids`),
    reconstruct the same merged buckets and return the extra "preserved
    subspan" candidates (and their audit rows) any borderline internal gap
    qualifies for -- see `_preserve_borderline_subspans`'s docstring.

    Deliberately a separate call from `reconstruct_delivery_attempts`
    itself (not merged into its own `attempts` return value) so every
    existing caller of that function -- most notably attempt_boundary_
    integrity.py's terminal-connector-guard monkeypatch, which indexes
    `attempts[-1]`/`attempts[-2]` assuming exactly one entry per merged
    bucket -- is completely unaffected. A caller that wants Invariant 2
    (good subspan preservation) opts in explicitly by calling this too and
    appending its result to its own candidate pool, exactly as flow_b.py
    does right after calling `reconstruct_delivery_attempts`."""
    by_id = {take.clip_id: take for take in takes}
    preserved: list[CandidateTake] = []
    audit: list[dict] = []
    for row in diagnostics.get("attempts") or ():
        member_ids = tuple(row.get("member_clip_ids") or ())
        if len(member_ids) < 2:
            continue
        bucket = tuple(by_id[cid] for cid in member_ids if cid in by_id)
        if len(bucket) != len(member_ids):
            # Some member is missing from `takes` (a stale/mismatched
            # diagnostics dict) -- fail closed rather than guess.
            continue
        bucket_preserved, bucket_audit = _preserve_borderline_subspans(
            [list(bucket)],
            subspan_preservation_min_gap_sec=subspan_preservation_min_gap_sec,
            max_continuation_gap_sec=max_continuation_gap_sec,
        )
        preserved.extend(bucket_preserved)
        audit.extend(bucket_audit)
    return tuple(preserved), audit
