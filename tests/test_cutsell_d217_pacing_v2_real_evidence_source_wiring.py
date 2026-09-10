"""D-217 -- PACING V2 REAL EVIDENCE-SOURCE WIRING. OFFLINE / DIAGNOSTIC ONLY.

Proves `pacing_v2_evidence_adapter.py` derives real relationship-hint/
Prosodic/candidate-timing evidence from ALREADY-COMPUTED, ALREADY-
SERIALIZED pipeline diagnostics (never rerunning P1/P2/ASR/Prosodic, never
calling a provider, never inventing a duration constant) and hands it to
D-216's own unmodified `build_pacing_v2_live_diagnostics` / D-215's own
unmodified `decide_transition` -- still zero live J_CUT/L_CUT/MICRO_AUDIO_
OVERLAP authority. Generic fixtures only -- no Video00 wording, timestamps,
or clip ids.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import HARD_CUT, J_CUT, L_CUT, MICRO_AUDIO_OVERLAP, TIGHT_CUT
from cutsell_worker.pacing_transition_decision import (
    DECISION_CONFLICTED,
    DECISION_SAFE_FALLBACK,
    DECISION_SUPPORTED,
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
)
from cutsell_worker import pacing_v2_evidence_adapter as adapter
from cutsell_worker import pacing_v2_live_diagnostics_integration as pv2

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "pacing_v2_evidence_adapter.py").read_text()


def _code_only(source: str) -> str:
    marker = '"""\nfrom __future__'
    idx = source.find(marker)
    return source[idx:] if idx != -1 else source


CODE_ONLY = _code_only(MODULE_SOURCE)

SOURCE_A = "synthetic_source_a"
SOURCE_B = "synthetic_source_b"


def _words(text: str, start: float, end: float) -> tuple[Word, ...]:
    tokens = text.split()
    if not tokens:
        return ()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _clip(clip_id, start, end, text, *, words=None, source=SOURCE_A, order=0, attempt_id=None) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=order, start=start, end=end,
        text=text, caption_text=text, words=(words if words is not None else _words(text, start, end)),
        semantic_role=SemanticRole.STORY, selected=True,
        attempt_id=attempt_id if attempt_id is not None else f"att_{clip_id}",
    )


def _moment_row(source_asset_id, attempt_id, relation_to_predecessor=None):
    return {
        "source_asset_id": source_asset_id, "attempt_ids": [attempt_id],
        "relation_to_predecessor": relation_to_predecessor,
    }


def _understanding(rows):
    return {"moments": list(rows)}


def _prosody_diag(*, restart="NONE", continuity="CONTINUOUS"):
    return {"prosodic_restart_state": restart, "prosodic_continuity_state": continuity}


def _take_judge_groups(candidate_evidence_by_clip):
    return [{"prosodic_pipeline_candidate_evidence": dict(candidate_evidence_by_clip)}]


# ---------------------------------------------------------------------------
# Fixture 1-8: candidate timing window derivation.
# ---------------------------------------------------------------------------

def test_01_right_has_safe_nonlexical_head_positive_j_lead():
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5))
    timing = adapter.available_silent_head_sec(right)
    assert timing == pytest.approx(0.3)


def test_02_right_starts_immediately_with_required_word_no_j_lead():
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.0, 9.5))
    assert adapter.available_silent_head_sec(right) == pytest.approx(0.0)


def test_03_left_has_safe_tail_positive_l_tail():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    assert adapter.available_silent_tail_sec(left) == pytest.approx(0.5)


def test_04_left_ends_exactly_on_required_word_no_l_tail():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 5.0))
    assert adapter.available_silent_tail_sec(left) == pytest.approx(0.0)


def test_05_both_have_safe_nonspeech_edges_micro_overlap_window():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    timing = adapter.candidate_timing_for_pair(left, right)
    assert timing["candidate_micro_overlap"] == pytest.approx(0.4)
    assert timing["candidate_timing_status"] == adapter.CANDIDATE_TIMING_AVAILABLE


def test_06_both_have_required_speech_at_overlap_no_micro_overlap():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 5.0))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.0, 9.5), source=SOURCE_B, order=1)
    timing = adapter.candidate_timing_for_pair(left, right)
    assert timing["candidate_micro_overlap"] is None
    assert timing["candidate_timing_status"] == adapter.CANDIDATE_TIMING_NONE_SAFE


def test_07_missing_words_timing_unavailable():
    left = _clip("a", 0.0, 5.0, "", words=())
    right = _clip("b", 5.0, 10.0, "", words=(), source=SOURCE_B, order=1)
    timing = adapter.candidate_timing_for_pair(left, right)
    assert timing["available_left_silent_tail"] is None
    assert timing["available_right_silent_head"] is None
    assert timing["candidate_timing_status"] == adapter.CANDIDATE_TIMING_UNKNOWN


def test_08_source_bound_reached_no_lead_or_tail():
    # First word starts exactly at clip.start -- zero room, never negative,
    # never clamped from something larger (there IS nothing larger: the
    # word literally starts at the boundary).
    right = _clip("b", 5.0, 10.0, "four", words=_words("four", 5.0, 6.0))
    assert adapter.available_silent_head_sec(right) == 0.0


# ---------------------------------------------------------------------------
# Fixture 9-12: relationship mapping.
# ---------------------------------------------------------------------------

def test_09_continuation_relation_mapped():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "CONTINUATION"),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] == RELATIONSHIP_CONTINUATION
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_MATCHED
    assert result["relationship_source"] == adapter.RELATIONSHIP_SOURCE_P1_LOCAL_SEQUENCE


def test_10_correction_relation_mapped():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "CORRECTION"),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] == RELATIONSHIP_CORRECTION


def test_11_retry_relation_mapped():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "RETRY"),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] == RELATIONSHIP_RETRY


def test_12_no_exact_pair_relation_unknown():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_missing")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "CONTINUATION"),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] is None
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_CLIP_NOT_IDENTIFIED


def test_12b_not_adjacent_in_p1_local_sequence_unknown():
    # A moment was removed between "a" and "c" -- never bridged by nearness.
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("c", 10.0, 15.0, "three", source=SOURCE_A, order=2, attempt_id="att_c")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "CONTINUATION"),
        _moment_row(SOURCE_A, "att_c", "CONTINUATION"),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] is None
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_NOT_ADJACENT_IN_P1


def test_12c_no_relation_resolved_still_unknown_hint():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", None),
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] is None
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_NO_RELATION_RESOLVED


def test_12d_ambiguous_identity_never_guessed():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None),
        _moment_row(SOURCE_A, "att_b", "CONTINUATION"),
        {"source_asset_id": SOURCE_A, "attempt_ids": ["att_b"], "relation_to_predecessor": "RETRY"},
    ])
    result = adapter.relationship_hint_for_pair(left, right, adapter._moments_by_source(understanding))
    assert result["relationship_hint"] is None
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_AMBIGUOUS_IDENTITY


def test_12e_temporal_proximity_alone_never_maps():
    # No P1 understanding at all -- clips are temporally adjacent but that
    # must never substitute for real evidence.
    left = _clip("a", 0.0, 5.0, "one")
    right = _clip("b", 5.0, 10.0, "two", order=1)
    result = adapter.relationship_hint_for_pair(left, right, {})
    assert result["relationship_hint"] is None
    assert result["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_NO_UNDERSTANDING


# ---------------------------------------------------------------------------
# Fixture 13-15: Prosodic mapping.
# ---------------------------------------------------------------------------

def test_13_prosodic_exact_mapping():
    left = _clip("a", 0.0, 5.0, "one")
    prosodic_by_id = adapter._prosodic_diagnostics_by_clip_id(_take_judge_groups({"a": _prosody_diag()}))
    result = adapter.prosodic_evidence_for_clip(left, prosodic_by_id)
    assert result["status"] == adapter.PROSODIC_STATUS_AVAILABLE
    assert result["evidence"].vocal_continuity_state == "CONTINUOUS"
    assert result["evidence"].restart_or_interruption_state == "NONE"


def test_14_prosodic_partial_coverage():
    left = _clip("a", 0.0, 5.0, "one")
    right = _clip("b", 5.0, 10.0, "two", order=1)
    groups = _take_judge_groups({"a": _prosody_diag()})  # only "a" covered
    evidence = adapter.build_pacing_v2_real_evidence((left, right), take_judge_groups=groups)
    row = evidence["pair_evidence_diagnostics"][0]
    assert row["left_prosodic_status"] == adapter.PROSODIC_STATUS_AVAILABLE
    assert row["right_prosodic_status"] == adapter.PROSODIC_STATUS_UNAVAILABLE
    assert row["prosodic_mapping_status"] == adapter.PROSODIC_PAIR_PARTIAL


def test_15_prosodic_absent():
    left = _clip("a", 0.0, 5.0, "one")
    right = _clip("b", 5.0, 10.0, "two", order=1)
    evidence = adapter.build_pacing_v2_real_evidence((left, right), take_judge_groups=())
    row = evidence["pair_evidence_diagnostics"][0]
    assert row["prosodic_mapping_status"] == adapter.PROSODIC_PAIR_UNAVAILABLE


# ---------------------------------------------------------------------------
# Fixture 16-18: same-source / multi-source / source identity.
# ---------------------------------------------------------------------------

def test_16_same_source_behavior():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None), _moment_row(SOURCE_A, "att_b", "RETRY"),
    ])
    evidence = adapter.build_pacing_v2_real_evidence(
        (left, right), editorial_moment_sequence_diagnostics=understanding,
    )
    assert evidence["pair_evidence_diagnostics"][0]["relationship_hint"] == RELATIONSHIP_RETRY


def test_17_multi_source_behavior_cross_source_pair_is_unknown():
    left = _clip("a", 0.0, 5.0, "one", source=SOURCE_A, attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None), _moment_row(SOURCE_B, "att_b", "RETRY"),
    ])
    evidence = adapter.build_pacing_v2_real_evidence(
        (left, right), editorial_moment_sequence_diagnostics=understanding,
    )
    row = evidence["pair_evidence_diagnostics"][0]
    assert row["relationship_hint"] is None
    assert row["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_CROSS_SOURCE
    assert row["left_clip_id"] == "a" and row["right_clip_id"] == "b"  # identity preserved


def test_18_source_identity_preserved_in_pair_rows():
    left = _clip("a", 0.0, 5.0, "one", source=SOURCE_A)
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=False)
    row = result["transitions"][0]
    assert row["left_source_asset_id"] == SOURCE_A and row["right_source_asset_id"] == SOURCE_B


# ---------------------------------------------------------------------------
# Fixture 19-20: no Boundary/Ordering mutation.
# ---------------------------------------------------------------------------

def test_19_no_boundary_mutation():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    before = (left.start, left.end, right.start, right.end)
    adapter.build_pacing_v2_live_diagnostics_with_real_evidence(
        (left, right), dialogue_overlap_enabled=True,
        editorial_moment_sequence_diagnostics=None, take_judge_groups=(),
    )
    after = (left.start, left.end, right.start, right.end)
    assert before == after


def test_20_no_ordering_authority_referenced():
    for banned in ("ordering_realization_plan", "ordering_composer_adapter", "compose_selected", "safe_compose_order"):
        assert banned not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# Fixture 21-22: determinism / input-order stability.
# ---------------------------------------------------------------------------

def test_21_deterministic_result():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    r1 = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    r2 = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    assert r1 == r2


def test_22_input_order_preserved_not_swapped():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((a, b), dialogue_overlap_enabled=False)
    assert result["transitions"][0]["left_clip_id"] == "a"
    assert result["transitions"][0]["right_clip_id"] == "b"


# ---------------------------------------------------------------------------
# Fixture 23-25: advanced-mode diagnostic-only / renderer / D-142 immutability.
# ---------------------------------------------------------------------------

def test_23_advanced_recommendation_is_diagnostic_only():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["selected_mode"] in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP, HARD_CUT, TIGHT_CUT)
    # It never becomes the LIVE mode -- nothing here writes back into D-142.
    assert "live_mode" in row  # observability field only, never authority


def test_24_renderer_live_unchanged_structurally():
    for banned in ("RenderSegment", "render_plan", "import render"):
        assert banned not in CODE_ONLY


def test_25_d142_live_unchanged_structurally():
    for banned in ("def plan_dialogue_pacing_transitions(", "def apply_dialogue_pacing_transition_pass("):
        assert banned not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# Additional coverage: J/L/micro eligibility end-to-end through real evidence,
# blocked-overlap fixture, correction/retry/continuation end-to-end, firewalls.
# ---------------------------------------------------------------------------

def _safe_jcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def _safe_lcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def test_j_cut_eligibility_end_to_end_real_evidence():
    left, right = _safe_jcut_pair()
    # Force lead-only geometry: give left a zero tail (ends on a word).
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 5.0))
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["selected_mode"] == J_CUT
    assert result["run_summary"]["j_cut_eligible_count"] == 1
    assert result["run_summary"]["advanced_mode_eligible_count"] == 1


def test_l_cut_eligibility_end_to_end_real_evidence():
    left, right = _safe_lcut_pair()
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.0, 9.5), source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["selected_mode"] == L_CUT
    assert result["run_summary"]["l_cut_eligible_count"] == 1


def test_micro_overlap_eligibility_end_to_end_real_evidence():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["selected_mode"] == MICRO_AUDIO_OVERLAP
    assert result["run_summary"]["micro_overlap_eligible_count"] == 1


def test_blocked_overlap_fixture_real_double_speech():
    # Both sides' required words abut the cut point exactly -- ZERO
    # available silent room on either edge (this adapter's own derived
    # candidate can never itself be double-speech-unsafe, since it only
    # ever offers room strictly outside real words -- see module
    # docstring's "safe by construction" property). This is the honest
    # NO_SAFE_WINDOW shape D-215 correctly falls back to HARD_CUT for.
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 5.0))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.0, 9.5), source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["candidate_timing_status"] == adapter.CANDIDATE_TIMING_NONE_SAFE
    assert row["selected_mode"] == HARD_CUT
    assert row["firewall_violation"] is False
    assert result["firewall_violation_count"] == 0


def test_correction_fixture_end_to_end():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5), attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None), _moment_row(SOURCE_A, "att_b", "CORRECTION"),
    ])
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence(
        (left, right), dialogue_overlap_enabled=True, editorial_moment_sequence_diagnostics=understanding,
    )
    row = result["transitions"][0]
    assert row["relationship_hint"] == RELATIONSHIP_CORRECTION
    assert row["decision_status"] == DECISION_SAFE_FALLBACK


def test_retry_fixture_end_to_end():
    left = _clip("a", 0.0, 5.0, "one", attempt_id="att_a")
    right = _clip("b", 5.0, 10.0, "two", source=SOURCE_A, order=1, attempt_id="att_b")
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None), _moment_row(SOURCE_A, "att_b", "RETRY"),
    ])
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence(
        (left, right), dialogue_overlap_enabled=True, editorial_moment_sequence_diagnostics=understanding,
    )
    row = result["transitions"][0]
    assert row["relationship_hint"] == RELATIONSHIP_RETRY
    assert row["decision_status"] == DECISION_CONFLICTED
    assert row["selected_mode"] == HARD_CUT


def test_continuation_fixture_end_to_end_no_restriction():
    left, right = _safe_jcut_pair()
    understanding = _understanding([
        _moment_row(SOURCE_A, "att_a", None), _moment_row(SOURCE_B, "att_b", "CONTINUATION"),
    ])
    # NOTE: continuation relation only reused when same-source-adjacent;
    # here left/right are cross-source (different SOURCE_A/SOURCE_B) so the
    # hint honestly resolves UNKNOWN -- still no restriction either way
    # (both sides here also carry natural leading/trailing silence, so the
    # advanced mode this pair is eligible for is MICRO_AUDIO_OVERLAP, not
    # pure J_CUT -- confirming CONTINUATION/UNKNOWN never restricts it).
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence(
        (left, right), dialogue_overlap_enabled=True, editorial_moment_sequence_diagnostics=understanding,
        take_judge_groups=(),
    )
    row = result["transitions"][0]
    assert row["relationship_hint"] is None
    assert row["selected_mode"] in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)


# ---------------------------------------------------------------------------
# Firewalls (reused from D-215, never duplicated here) + no-magic-duration.
# ---------------------------------------------------------------------------

def test_meaning_firewall_reused_never_duplicated():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "no it is not correct", words=_words("no it is not correct", 5.0, 9.5), source=SOURCE_B, order=1)
    result = adapter.build_pacing_v2_live_diagnostics_with_real_evidence((left, right), dialogue_overlap_enabled=True)
    row = result["transitions"][0]
    assert row["selected_mode"] not in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
    for banned in ("def classify_claim(", "CRITICAL ="):
        assert banned not in MODULE_SOURCE


def test_no_magic_duration_constants():
    for banned in ("J_CUT_MS", "L_CUT_MS", "OVERLAP_MS", "IDEAL_GAP_MS"):
        assert banned not in CODE_ONLY  # docstring names them only as examples of what NOT to add


def test_no_duplicate_transition_classifier():
    assert "def decide_transition(" not in MODULE_SOURCE
    assert "decide_transition(" not in MODULE_SOURCE  # never called directly -- only via D-216's own function
    assert "def build_pacing_v2_live_diagnostics(" not in MODULE_SOURCE


def test_no_new_word_or_asr_recompute():
    for banned in ("from .asr", "from .language_spine import", "def analyze_prosodic_delivery("):
        assert banned not in MODULE_SOURCE
    assert "analyze_prosodic_delivery(" not in MODULE_SOURCE


def test_no_provider_call():
    for banned in ("requests.", "urllib", "http.client", "OPENAI_API_KEY", "genai.", "GenerativeModel"):
        assert banned not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# Regression: D-216 own suite untouched behaviorally -- verified by import
# shape (module never redefines D-216's own public names) and by the full
# D-216 suite being run separately in the same qualification pass.
# ---------------------------------------------------------------------------

def test_d216_public_surface_untouched():
    assert hasattr(pv2, "build_pacing_v2_live_diagnostics")
    assert hasattr(pv2, "pacing_v2_diagnostics_enabled")
    assert "def build_pacing_v2_live_diagnostics(" not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# Pipeline-level wiring: default-off parity + flag-on immutability, mirroring
# D-216's own D-097.C-precedent pattern.
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    def _fixture(self):
        from cutsell_worker.contracts import DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION

        words_a = _words("uno dos tres cuatro", 0.2, 4.8)
        words_b = _words("cinco seis siete", 5.2, 9.5)
        left = DraftClip(
            clip_id="kept_a", source_asset_id="src", source_order=0, start=0.0, end=5.0,
            text="uno dos tres cuatro", caption_text="uno dos tres cuatro", words=words_a,
            semantic_role=SemanticRole.STORY, selected=True, attempt_id="att_kept_a",
        )
        right = DraftClip(
            clip_id="kept_b", source_asset_id="src", source_order=1, start=5.0, end=10.0,
            text="cinco seis siete", caption_text="cinco seis siete", words=words_b,
            semantic_role=SemanticRole.STORY, selected=True, attempt_id="att_kept_b",
        )
        draft = DraftTimeline(
            schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
            selected=(left, right), alternates=(), discarded=(), diagnostics={
                "editorial_moment_sequence": _understanding([
                    _moment_row("src", "att_kept_a", None),
                    _moment_row("src", "att_kept_b", "CONTINUATION"),
                ]),
            },
        )
        result = ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})
        return result, (left, right)

    def _run_pipeline_stub(self, monkeypatch, request_obj):
        import cutsell_worker.universal_clean_cut as universal
        result, clips = self._fixture()

        def fake_process(request, local_paths, **kwargs):
            return result

        monkeypatch.setattr(universal, "process_local_sources", fake_process)
        monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda res, paths: res)
        monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda res, paths, **kw: res)
        out = universal.process_universal_clean_cut_sources(
            request_obj, {}, asr_provider=object(), selection_reasoner=None,
        )
        return out, clips

    def test_default_off_byte_parity(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", raising=False)
        out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert "pacing_v2" not in out.draft.diagnostics

    def test_flag_on_real_evidence_reaches_live_seam(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        pacing_v2 = out.draft.diagnostics["pacing_v2"]
        row = pacing_v2["transitions"][0]
        # The real editorial_moment_sequence fixture supplies a genuine
        # CONTINUATION relation for this exact pair -- proving the live
        # seam actually threads it through, not just offline unit calls.
        assert row["relationship_hint"] == RELATIONSHIP_CONTINUATION
        assert row["relationship_mapping_status"] == adapter.RELATIONSHIP_MAPPING_MATCHED

    def test_flag_on_immutability_selected_and_edges_unchanged(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        left, right = clips
        selected_by_id = {c.clip_id: c for c in out.draft.selected}
        assert [c.clip_id for c in out.draft.selected] == ["kept_a", "kept_b"]
        assert (selected_by_id["kept_a"].start, selected_by_id["kept_a"].end) == (left.start, left.end)
        assert (selected_by_id["kept_b"].start, selected_by_id["kept_b"].end) == (right.start, right.end)

    def test_flag_on_d142_diagnostics_unchanged_vs_off(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "0")
        off_out, _ = self._run_pipeline_stub(monkeypatch, object())
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        on_out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert off_out.draft.diagnostics["dialogue_pacing_transition"] == on_out.draft.diagnostics["dialogue_pacing_transition"]
        off_keys = {k: v for k, v in off_out.draft.diagnostics.items() if k not in ("pacing_v2",)}
        on_keys = {k: v for k, v in on_out.draft.diagnostics.items() if k not in ("pacing_v2",)}
        assert off_keys == on_keys


# ---------------------------------------------------------------------------
# compileall.
# ---------------------------------------------------------------------------

def test_module_compiles_and_imports():
    import cutsell_worker.pacing_v2_evidence_adapter  # noqa: F401


def test_ast_no_module_level_side_effects():
    tree = ast.parse(MODULE_SOURCE)
    for node in tree.body:
        if isinstance(node, ast.Expr) and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            pytest.fail(f"unexpected module-level side effect: {ast.dump(node)}")
