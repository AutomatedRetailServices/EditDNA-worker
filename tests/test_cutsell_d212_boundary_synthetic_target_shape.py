"""D-212 -- D-177 SYNTHETIC TARGET-SHAPE QUALIFICATION.

D-211 established that D-177's partial-edge trim mechanism is
OFFLINE_PROVEN and real-media SAFETY_PROVEN, but its own TRUE target shape
(a real straddling event with genuine debris outside required DELIVERY
that D-177 actually trims) has never yet occurred in a real Video00
dispatch. Product Owner directive D-212 authorizes ONE controlled,
deliberately-constructed exercise of that exact geometry instead of
waiting on more paid RAWs.

## Exact target geometry (derived from `boundary_engine_pass.py` and
`positioned_performance_evidence.py::classify_event_zone`, NOT assumed
from any prompt)

`classify_event_zone` returns `overlaps = event_start < d_end and
event_end > d_start`, `starts_before = event_start < d_start`, `ends_after
= event_end > d_end`, and zone=DELIVERY whenever `overlaps` is true (even
when the event also straddles a boundary). `tighten_selected_visual_edges`
treats a DELIVERY-zone event with `before=True, after=False` as an ENTRY
straddle, and one with `before=False, after=True` as an EXIT straddle.
Substituting:

  ENTRY straddle:  event_start < delivery_start < event_end <= delivery_end
  EXIT  straddle:  delivery_start <= event_start < delivery_end < event_end

D-177's own eligibility test (`inside_overlap <=
AUDIO_EDGE_OVERLAP_TOLERANCE_SEC`, no new constant) is:

  ENTRY:  (event_end - delivery_start)  <= AUDIO_EDGE_OVERLAP_TOLERANCE_SEC
  EXIT:   (delivery_end - event_start)  <= AUDIO_EDGE_OVERLAP_TOLERANCE_SEC

Both directions are proven implemented symmetrically in
`tighten_selected_visual_edges` (the ENTRY and EXIT partial-edge-trim
loops), so both are exercised here as mirrored variants
(`PRE_EDGE_STRADDLE` / `POST_EDGE_STRADDLE`).

## Boundary input seam (Step 2 -- not guessed)

`tighten_selected_visual_edges` (and the orchestrator that calls it,
`apply_post_freeze_boundary_pass`) consumes exactly two things: a
`DraftClip`'s own already-aligned `words` (used by `compute_delivery_span`)
and structured event dicts under
`diagnostics["whole_video_context"]["sources"][].events` (matching D-114's
`TemporalEvent` schema). It never touches raw pixels or audio samples --
those are D-114/`local_performance.py`'s and `audio_silence.py`'s own,
separately-proven upstream responsibility, out of D-212's authorized
scope. This fixture therefore enters at exactly that seam: hand-authored
`Word`/event evidence in the EXACT shape Boundary already reads, run
through the REAL `apply_post_freeze_boundary_pass` orchestrator (not an
isolated call to `tighten_selected_visual_edges` alone) -- the lowest
honest seam that still exercises D-177's actual decision logic, without
re-deriving ASR, a provider call, BestTake, P1, P2, or Ordering (none of
which D-177 itself consumes). No raw/synthetic MP4 is generated: D-177
never consumes pixels, so an MP4 would not exercise this mechanism any
more than the structured evidence already does, and re-running the actual
CV/audio detectors that WOULD consume such an MP4 is D-116's own
already-separately-tested upstream dependency, not this task's to rebuild.
Achieved proof level is therefore honestly
`STRUCTURED_INTEGRATION_FIXTURE_PROVEN` (real `apply_post_freeze_
boundary_pass` orchestration, not a bare unit call), never
`SYNTHETIC_MEDIA_INTEGRATION_PROVEN` and never `REAL_MEDIA_TARGET_SHAPE_
PROVEN`.

No new constant is introduced anywhere in this file -- every eligibility
value reuses `AUDIO_EDGE_OVERLAP_TOLERANCE_SEC` imported directly from
`boundary_engine_pass`. No Video00 timestamps/regions/labels appear
anywhere; the synthetic source id is `synthetic_d177_target` and the
transcript text is a generic five-token placeholder.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from cutsell_worker.boundary_engine_pass import (
    AUDIO_EDGE_OVERLAP_TOLERANCE_SEC,
    BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM,
    BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM,
    BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM,
    apply_post_freeze_boundary_pass,
)
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult,
    SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.positioned_performance_evidence import classify_event_zone
from cutsell_worker.selection_boundary_contract import (
    enforce_selection_contract, freeze_selection_contract,
)

SYNTHETIC_SOURCE_ID = "synthetic_d177_target"
DELIVERY_WORDS_TEXT = "one two three four five"
DELIVERY_START = 10.0
DELIVERY_END = 19.0


def _words(text: str, start: float, end: float) -> tuple[Word, ...]:
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


DELIVERY_WORDS = _words(DELIVERY_WORDS_TEXT, DELIVERY_START, DELIVERY_END)


def _clip(clip_id: str, start: float, end: float, *, words, source: str = SYNTHETIC_SOURCE_ID, source_order: int = 0) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=source_order, start=start, end=end,
        text=DELIVERY_WORDS_TEXT, caption_text=DELIVERY_WORDS_TEXT, words=words,
        semantic_role=SemanticRole.STORY, selected=True,
    )


def _event(kind: str, start: float, end: float, confidence: float = 0.9) -> dict:
    return {"kind": kind, "start": start, "end": end, "confidence": confidence}


def _diag(events_by_source: dict) -> dict:
    return {
        "whole_video_context": {
            "sources": [
                {"source_asset_id": source, "events": list(events)}
                for source, events in events_by_source.items()
            ]
        },
    }


def _draft(clips, diagnostics) -> DraftTimeline:
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="d212", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics,
    )


def _result(draft) -> ProcessingResult:
    return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="d212", state=JobState.DRAFT_READY, draft=draft, stage_status={})


def _control_clip() -> DraftClip:
    """An untouched second clip, far away in the same synthetic source,
    with no events at all -- proves the pass never touches anything it
    has no evidence for (item 12: no unrelated boundary change)."""
    return _clip("control", 30.0, 39.5, words=_words(DELIVERY_WORDS_TEXT, 30.0, 39.0), source_order=1)


def _run_boundary_pass(clips, diag) -> ProcessingResult:
    """Freeze -> apply_post_freeze_boundary_pass -> enforce_selection_contract,
    the REAL live sequence (`universal_clean_cut.py`), not a bare call to
    `tighten_selected_visual_edges` in isolation."""
    frozen_draft = freeze_selection_contract(_draft(clips, diag))
    result = apply_post_freeze_boundary_pass(_result(frozen_draft))
    verified_draft = enforce_selection_contract(result.draft)
    return replace(result, draft=verified_draft)


def _clip_by_id(result: ProcessingResult, clip_id: str) -> DraftClip:
    return next(c for c in result.draft.selected if c.clip_id == clip_id)


def _partial_rows(result: ProcessingResult, clip_id: str) -> list[dict]:
    rows = result.draft.diagnostics["boundary_engine_pass"]["partial_edge_trim_rows"]
    return [r for r in rows if r["clip_id"] == clip_id]


# --- fixtures --------------------------------------------------------------

def _entry_positive_fixture():
    clip_start = DELIVERY_START - 0.5  # 0.5s of leading debris entirely OUTSIDE delivery
    clip_end = 20.0
    target = _clip("target", clip_start, clip_end, words=DELIVERY_WORDS)
    event_start = clip_start  # touches the clip's own current leading edge
    event_end = DELIVERY_START + 0.04  # inside-delivery portion 0.04s <= 0.08 tolerance
    diag = _diag({SYNTHETIC_SOURCE_ID: [_event("camera_disengagement_candidate", event_start, event_end)]})
    return target, diag, event_start, event_end, clip_start, clip_end


def _exit_positive_fixture():
    clip_start = DELIVERY_START
    clip_end = DELIVERY_END + 0.5  # 0.5s of trailing debris entirely OUTSIDE delivery
    target = _clip("target", clip_start, clip_end, words=DELIVERY_WORDS)
    event_end = clip_end  # touches the clip's own current trailing edge
    event_start = DELIVERY_END - 0.04  # inside-delivery portion 0.04s <= 0.08 tolerance
    diag = _diag({SYNTHETIC_SOURCE_ID: [_event("facial_expression_shift_candidate", event_start, event_end)]})
    return target, diag, event_start, event_end, clip_start, clip_end


def _negative_inside_delivery_fixture():
    """Event wholly inside DELIVERY -- no straddle at all."""
    target = _clip("target", DELIVERY_START, 20.0, words=DELIVERY_WORDS)
    diag = _diag({SYNTHETIC_SOURCE_ID: [_event("body_reset_candidate", 12.0, 12.5)]})
    return target, diag


def _negative_material_overlap_fixture():
    """A real straddle, but the inside-delivery portion exceeds tolerance --
    a genuine DELIVERY defect, must never be trimmed here."""
    clip_start = DELIVERY_START - 0.5
    target = _clip("target", clip_start, 20.0, words=DELIVERY_WORDS)
    diag = _diag({SYNTHETIC_SOURCE_ID: [_event("camera_disengagement_candidate", clip_start, DELIVERY_START + 0.5)]})
    return target, diag


# --- Step 1: exact target-shape geometry proven (items 1-3) ----------------

def test_01_entry_geometry_matches_derived_inequality():
    _, _, event_start, event_end, _, _ = _entry_positive_fixture()
    assert event_start < DELIVERY_START < event_end <= DELIVERY_END
    assert (event_end - DELIVERY_START) <= AUDIO_EDGE_OVERLAP_TOLERANCE_SEC


def test_02_exit_geometry_matches_derived_inequality():
    _, _, event_start, event_end, _, _ = _exit_positive_fixture()
    assert DELIVERY_START <= event_start < DELIVERY_END < event_end
    assert (DELIVERY_END - event_start) <= AUDIO_EDGE_OVERLAP_TOLERANCE_SEC


def test_03_entry_fixture_truly_straddles_via_canonical_classifier():
    from cutsell_worker.positioned_performance_evidence import compute_delivery_span
    target, _, event_start, event_end, _, _ = _entry_positive_fixture()
    span = compute_delivery_span(target.words)
    zone, overlaps, before, after = classify_event_zone(event_start, event_end, span)
    assert zone == "DELIVERY" and overlaps is True and before is True and after is False


def test_04_exit_fixture_truly_straddles_via_canonical_classifier():
    from cutsell_worker.positioned_performance_evidence import compute_delivery_span
    target, _, event_start, event_end, _, _ = _exit_positive_fixture()
    span = compute_delivery_span(target.words)
    zone, overlaps, before, after = classify_event_zone(event_start, event_end, span)
    assert zone == "DELIVERY" and overlaps is True and before is False and after is True


def test_05_entry_external_debris_exists_outside_delivery():
    _, _, event_start, _, clip_start, _ = _entry_positive_fixture()
    debris_span = DELIVERY_START - event_start
    assert debris_span > 0.0
    assert event_start < DELIVERY_START  # debris lies strictly outside DELIVERY


def test_06_exit_external_debris_exists_outside_delivery():
    _, _, _, event_end, _, clip_end = _exit_positive_fixture()
    debris_span = event_end - DELIVERY_END
    assert debris_span > 0.0
    assert event_end > DELIVERY_END  # debris lies strictly outside DELIVERY


# --- positive integration proof (items 4-12, 25-27) -------------------------

def test_07_entry_evaluated_and_applied_through_real_boundary_pass():
    target, diag, *_ = _entry_positive_fixture()
    control = _control_clip()
    result = _run_boundary_pass((target, control), diag)
    rows = _partial_rows(result, "target")
    assert len(rows) == 1
    row = rows[0]
    assert row["partial_edge_trim_evaluated"] is True
    assert row["partial_edge_trim_applied"] is True
    assert row["entry_partial_edge_trim_applied"] is True
    assert row["exit_partial_edge_trim_applied"] is False


def test_08_exit_evaluated_and_applied_through_real_boundary_pass():
    target, diag, *_ = _exit_positive_fixture()
    control = _control_clip()
    result = _run_boundary_pass((target, control), diag)
    rows = _partial_rows(result, "target")
    assert len(rows) == 1
    row = rows[0]
    assert row["partial_edge_trim_evaluated"] is True
    assert row["partial_edge_trim_applied"] is True
    assert row["exit_partial_edge_trim_applied"] is True
    assert row["entry_partial_edge_trim_applied"] is False


def test_09_entry_physical_boundary_actually_changes():
    target, diag, _, _, clip_start, clip_end = _entry_positive_fixture()
    control = _control_clip()
    result = _run_boundary_pass((target, control), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start != clip_start  # a REAL trim occurred, not evaluated=true/applied=false
    assert after_clip.start == pytest.approx(DELIVERY_START)
    assert after_clip.end == pytest.approx(clip_end)  # untouched side unaffected


def test_10_exit_physical_boundary_actually_changes():
    target, diag, _, _, clip_start, clip_end = _exit_positive_fixture()
    control = _control_clip()
    result = _run_boundary_pass((target, control), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.end != clip_end
    assert after_clip.end == pytest.approx(DELIVERY_END)
    assert after_clip.start == pytest.approx(clip_start)


def test_11_entry_clamps_exactly_at_delivery_floor_never_past_it():
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start == pytest.approx(DELIVERY_START)
    assert after_clip.start >= DELIVERY_START - 1e-9  # never enters required DELIVERY


def test_12_exit_clamps_exactly_at_delivery_ceiling_never_past_it():
    target, diag, *_ = _exit_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.end == pytest.approx(DELIVERY_END)
    assert after_clip.end <= DELIVERY_END + 1e-9  # never enters required DELIVERY


def test_13_entry_required_delivery_fully_preserved():
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start <= DELIVERY_START and after_clip.end >= DELIVERY_END


def test_14_exit_required_delivery_fully_preserved():
    target, diag, *_ = _exit_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start <= DELIVERY_START and after_clip.end >= DELIVERY_END


def test_15_entry_no_word_cut_every_kept_word_fully_inside_retained_span():
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    for word in after_clip.words:
        assert word.start >= after_clip.start - 1e-9
        assert word.end <= after_clip.end + 1e-9


def test_16_exit_no_word_cut_every_kept_word_fully_inside_retained_span():
    target, diag, *_ = _exit_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    for word in after_clip.words:
        assert word.start >= after_clip.start - 1e-9
        assert word.end <= after_clip.end + 1e-9


def test_17_selection_contract_verified_no_semantic_content_change():
    """`enforce_selection_contract` succeeding (no RuntimeError) proves the
    ordered spoken token stream survived the trim untouched -- membership
    and text content, the selection contract's own authority, never moved."""
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "verified"


def test_18_entry_diagnostic_reason_is_exact_canonical_reason():
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    row = _partial_rows(result, "target")[0]
    assert row["partial_edge_trim_reason"] == BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM


def test_19_exit_diagnostic_reason_is_exact_canonical_reason():
    target, diag, *_ = _exit_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    row = _partial_rows(result, "target")[0]
    assert row["partial_edge_trim_reason"] == BOUNDARY_REASON_VISUAL_EXIT_PARTIAL_EDGE_TRIM


def test_20_source_identity_preserved_before_and_after():
    target, diag, *_ = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.clip_id == "target"
    assert after_clip.source_asset_id == SYNTHETIC_SOURCE_ID == target.source_asset_id


def test_21_before_after_diagnostic_boundaries_recorded_correctly():
    target, diag, _, _, clip_start, clip_end = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    row = _partial_rows(result, "target")[0]
    assert row["boundary_before_start"] == pytest.approx(clip_start)
    assert row["boundary_before_end"] == pytest.approx(clip_end)
    assert row["boundary_after_start"] == pytest.approx(DELIVERY_START)
    assert row["boundary_after_end"] == pytest.approx(clip_end)


def test_22_no_unrelated_clip_changed_by_the_positive_trim():
    target, diag, *_ = _entry_positive_fixture()
    control = _control_clip()
    result = _run_boundary_pass((target, control), diag)
    after_control = _clip_by_id(result, "control")
    assert after_control.start == control.start
    assert after_control.end == control.end
    assert after_control.words == control.words


# --- negative controls (items 13-15) ----------------------------------------

def test_23_negative_control_event_inside_delivery_not_trimmed():
    target, diag = _negative_inside_delivery_fixture()
    result = _run_boundary_pass((target,), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start == target.start
    assert after_clip.end == target.end
    audit_rows = result.draft.diagnostics["boundary_engine_pass"]["visual_edge_rows"]
    matching = [r for r in audit_rows if r["clip_id"] == "target"]
    assert matching and matching[0]["reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM
    assert matching[0]["trim_applied"] is False
    # A non-straddling embedded event never even enters the D-177 partial
    # summary (inside_delivery_overlap_sec stays None) -- correctly filtered.
    assert _partial_rows(result, "target") == []


def test_24_negative_control_material_overlap_not_trimmed():
    target, diag = _negative_material_overlap_fixture()
    result = _run_boundary_pass((target,), diag)
    after_clip = _clip_by_id(result, "target")
    assert after_clip.start == target.start  # untouched: real DELIVERY defect, D-177 declines
    row = _partial_rows(result, "target")[0]
    assert row["partial_edge_trim_applied"] is False
    assert row["partial_edge_trim_reason"] == BOUNDARY_REASON_VISUAL_DELIVERY_OVERLAP_NO_TRIM


# --- determinism (item 16) ---------------------------------------------------

def test_25_entry_positive_result_is_deterministic_across_two_runs():
    target1, diag1, *_ = _entry_positive_fixture()
    target2, diag2, *_ = _entry_positive_fixture()
    result1 = _run_boundary_pass((target1, _control_clip()), diag1)
    result2 = _run_boundary_pass((target2, _control_clip()), diag2)
    c1, c2 = _clip_by_id(result1, "target"), _clip_by_id(result2, "target")
    assert c1.start == c2.start and c1.end == c2.end
    r1, r2 = _partial_rows(result1, "target")[0], _partial_rows(result2, "target")[0]
    assert r1 == r2


def test_26_exit_positive_result_is_deterministic_across_two_runs():
    target1, diag1, *_ = _exit_positive_fixture()
    target2, diag2, *_ = _exit_positive_fixture()
    result1 = _run_boundary_pass((target1, _control_clip()), diag1)
    result2 = _run_boundary_pass((target2, _control_clip()), diag2)
    c1, c2 = _clip_by_id(result1, "target"), _clip_by_id(result2, "target")
    assert c1.start == c2.start and c1.end == c2.end
    r1, r2 = _partial_rows(result1, "target")[0], _partial_rows(result2, "target")[0]
    assert r1 == r2


# --- discipline checks (items 17-22, 26-27) ---------------------------------

def test_27_no_new_boundary_threshold_introduced():
    # Every eligibility computation in this file reuses the existing
    # constant; this asserts its value was not silently redefined anywhere
    # this file imports from.
    assert AUDIO_EDGE_OVERLAP_TOLERANCE_SEC == pytest.approx(0.08)


def test_28_fixture_is_source_generic_no_video00_identifiers():
    forbidden = ("pimples", "gyneco", "video00", "cut.ai", "human gold", "stomach")
    haystack = (SYNTHETIC_SOURCE_ID + " " + DELIVERY_WORDS_TEXT).lower()
    for needle in forbidden:
        assert needle not in haystack


def test_29_module_imports_no_network_provider_or_upstream_authorities():
    import ast
    import pathlib
    forbidden_modules = {
        "requests", "httpx", "urllib", "openai", "google", "genai",
        "modal", "runpod",
        "ordering_realization_plan", "ordering_composer_adapter",
        "ordering_live_diagnostics_integration",
        "editorial_moment_sequence", "whole_video_editorial_reasoning",
        "dialogue_pacing_transition", "render", "take_judge",
    }
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    hit = imported & forbidden_modules
    assert not hit, f"D-212 fixture imports an out-of-scope upstream module: {hit}"


def test_30_full_positive_scenario_summary_matches_deliverable_shape():
    """One consolidated end-to-end assertion mirroring the D-212 directive's
    own deliverable checklist for the ENTRY variant, in a single place."""
    target, diag, event_start, event_end, clip_start, clip_end = _entry_positive_fixture()
    result = _run_boundary_pass((target, _control_clip()), diag)
    after_clip = _clip_by_id(result, "target")
    row = _partial_rows(result, "target")[0]
    assert row["partial_edge_trim_evaluated"] is True
    assert row["partial_edge_trim_applied"] is True
    assert row["partial_edge_trim_reason"] == BOUNDARY_REASON_VISUAL_ENTRY_PARTIAL_EDGE_TRIM
    assert row["boundary_before_start"] == pytest.approx(clip_start)
    assert row["boundary_after_start"] == pytest.approx(DELIVERY_START)
    assert after_clip.start <= DELIVERY_START and after_clip.end >= DELIVERY_END
    for word in after_clip.words:
        assert word.start >= after_clip.start and word.end <= after_clip.end
    assert after_clip.source_asset_id == SYNTHETIC_SOURCE_ID
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "verified"
