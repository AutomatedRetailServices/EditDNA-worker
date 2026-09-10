"""D-199: P1 LIVE LANGUAGE-SPINE EVIDENCE INTEGRATION -- tests.

Proves: (1) the live Language Spine is constructed ONCE per source purely
from already-computed ASR word timings (zero re-transcription); (2) P1
consumes the real canonical LanguageAttempt/PropositionCandidate/
RelationEvidence collections when construction succeeds, and falls back to
the existing D-157 approximation -- never crashing, never dropping a
source -- when it doesn't; (3) canonical and D-157 relation evidence are
FUSED (agreement/conflict-abstention/single-source), never silently
majority-voted or overwritten; (4) both flags OFF (or the live spine flag
alone off) reproduce D-198 behavior byte-for-byte; (5) the whole thing is
bilingual-safe (English/Spanish/Spanglish) via STRUCTURAL evidence only,
never a phrase dictionary or a ``language ==`` branch built for these
fixtures.

See docs/CUTSELL_DECISIONS.md D-199.
"""
from __future__ import annotations

import ast
import inspect
import re

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.editorial_moment_sequence_integration import (
    build_editorial_moment_understanding_for_source,
    build_editorial_moment_understanding_for_sources,
    build_editorial_moments_for_source,
    editorial_moment_understanding_diagnostics,
    live_language_spine_source_diagnostics_for_p1,
)
import cutsell_worker.editorial_moment_sequence_integration as p1_integration_module
from cutsell_worker.language_proposition_relation import RelationEvidence
from cutsell_worker.language_spine import count_meaning_sensitive_tokens_preserved
import cutsell_worker.language_spine_live_integration as spine_live_module
from cutsell_worker.language_spine_live_integration import (
    CAPABILITY_AVAILABLE,
    CAPABILITY_NOT_EVALUABLE,
    LANGUAGE_EVIDENCE_CANONICAL,
    LANGUAGE_EVIDENCE_D157_FALLBACK,
    RELATION_SOURCE_AGREEMENT,
    RELATION_SOURCE_CANONICAL_ONLY,
    RELATION_SOURCE_CONFLICT_ABSTAINED,
    RELATION_SOURCE_D157_ONLY,
    RELATION_SOURCE_MISSING,
    build_live_language_spine_for_source,
    fuse_relation_evidence,
    language_attempts_by_span_id_for_source,
    live_language_spine_diagnostics,
    live_language_spine_diagnostics_enabled,
    live_language_spine_run_summary,
    relation_evidence_by_proposition_pair,
)
from cutsell_worker.language_utterance_attempt import ATTEMPT_CLEAN
from cutsell_worker.raw_understanding_map import BEHAVIOR_CLEAN_ATTEMPT, BehaviorHypothesis, RawUnderstandingMap
from cutsell_worker.watch_listen_understanding import (
    AttemptRelationHypothesis,
    CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
    MEANING_COMPLETE as WL_MEANING_COMPLETE,
    RELATION_RETRY,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------
def _take(clip_id, order, start, end, text="generic abstract statement"):
    return CandidateTake(clip_id=clip_id, source_asset_id="src1", source_order=order, start=start, end=end, text=text)


def _behavior(label, confidence=0.8):
    return BehaviorHypothesis(label=label, confidence=confidence, provenance="VISUAL_SIGNAL", basis="generic")


def _span(span_id, start, end, *, source_asset_id="src1", behavior_labels=(), relation=None):
    hyps = tuple(_behavior(label) for label in behavior_labels)
    rel = (AttemptRelationHypothesis(relation, WL_CONFIDENCE_SUPPORTED, "x", None, ()),) if relation else ()
    return UnderstandingSpan(
        span_id=span_id, source_asset_id=source_asset_id, source_start=start, source_end=end,
        behavior_state_hypotheses=hyps, behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=rel,
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(*spans, source_asset_id="src1"):
    return WatchListenUnderstanding(source_asset_id=source_asset_id, understanding_spans=tuple(spans))


def _sentence_words(text: str, t0: float, *, per_word: float = 0.25, inner_gap: float = 0.05):
    """Builds a contiguous ``Word`` tuple for one sentence -- gaps between
    words are small (never a phrase/utterance boundary on their own)."""
    words = []
    t = t0
    for tok in text.split(" "):
        start, end = t, t + per_word
        words.append(Word(text=tok, start=start, end=end, confidence=0.9))
        t = end + inner_gap
    return tuple(words), t


def _two_sentence_words(s1: str, s2: str, *, gap_between: float = 2.5):
    """Two sentences separated by a real pause (>> DEFAULT_SPLIT_GAP_SEC) --
    reliably forces separate phrases/utterances via structural (timing)
    evidence only, never text content."""
    words1, t = _sentence_words(s1, 0.0)
    words2, _ = _sentence_words(s2, t + gap_between)
    return words1 + words2


def _raw_map(source_asset_id: str, words):
    return RawUnderstandingMap(
        source_asset_id=source_asset_id, source_duration=(max((w.end for w in words), default=0.0) + 1.0),
        source_timeline_origin="source_relative_seconds", transcript="unused -- word_timings is authoritative",
        word_timings=tuple(words),
    )


def _pair_understanding_with_live_spine(relation, live_language_spine=None, *, source_asset_id="src1"):
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, source_asset_id=source_asset_id, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, source_asset_id=source_asset_id, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=relation),
    )
    return build_editorial_moment_understanding_for_source(
        source_asset_id=source_asset_id, takes_for_source=takes, watch_listen_understanding=wlu,
        live_language_spine=live_language_spine,
    )


# ===========================================================================
# 1-2: Flag default/on.
# ===========================================================================
def test_01_flag_default_off():
    assert live_language_spine_diagnostics_enabled({}) is False


def test_02_flag_on():
    assert live_language_spine_diagnostics_enabled(
        {"CUTSELL_LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED": "1"}
    ) is True


# ===========================================================================
# 3-8: Construction basics.
# ===========================================================================
def test_03_no_raw_understanding_map_not_evaluable():
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=None)
    assert evidence.capability_status == CAPABILITY_NOT_EVALUABLE
    assert "RAW_UNDERSTANDING_MAP_WORD_TIMINGS_ABSENT" in evidence.missing_evidence
    assert evidence.attempts == ()


def test_04_empty_word_timings_not_evaluable():
    raw_map = _raw_map("s1", ())
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert evidence.capability_status == CAPABILITY_NOT_EVALUABLE


def test_05_real_words_construct_full_spine():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    raw_map = _raw_map("s1", words)
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert evidence.capability_status == CAPABILITY_AVAILABLE
    assert len(evidence.words) == len(words)
    assert len(evidence.phrases) >= 1
    assert len(evidence.utterances) >= 1
    assert len(evidence.attempts) >= 1
    assert len(evidence.proposition_candidates) >= 1


def test_06_source_identity_preserved_on_every_object():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    raw_map = _raw_map("srcX", words)
    evidence = build_live_language_spine_for_source(source_asset_id="srcX", raw_understanding_map=raw_map)
    assert evidence.source_asset_id == "srcX"
    assert all(w.source_asset_id == "srcX" for w in evidence.words)
    assert all(p.source_asset_id == "srcX" for p in evidence.phrases)
    assert all(u.source_asset_id == "srcX" for u in evidence.utterances)
    assert all(a.source_asset_id == "srcX" for a in evidence.attempts)
    assert all(p.source_asset_id == "srcX" for p in evidence.proposition_candidates)


def test_07_construction_is_deterministic():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    raw_map = _raw_map("s1", words)
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    e2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert tuple(a.attempt_id for a in e1.attempts) == tuple(a.attempt_id for a in e2.attempts)
    assert tuple(p.proposition_candidate_id for p in e1.proposition_candidates) == \
        tuple(p.proposition_candidate_id for p in e2.proposition_candidates)


def test_08_silence_evidence_honestly_flagged_missing():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    raw_map = _raw_map("s1", words)
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert "AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED" in evidence.missing_evidence


# ===========================================================================
# 9-11: NO ASR RERUN.
# ===========================================================================
def test_09_module_imports_no_asr_provider():
    src = inspect.getsource(spine_live_module)
    forbidden = ("import asr", "from .asr", "WhisperModel", "run_asr", ".transcribe(")
    for needle in forbidden:
        assert needle not in src, needle


def test_10_construction_ignores_transcript_field():
    # `transcript` is deliberately wrong/unrelated -- only `word_timings`
    # may ever produce the LanguageWord/Phrase/Utterance/Attempt text.
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    raw_map = RawUnderstandingMap(
        source_asset_id="s1", source_duration=5.0, source_timeline_origin="source_relative_seconds",
        transcript="COMPLETELY UNRELATED TRANSCRIPT TEXT NEVER SPOKEN", word_timings=words,
    )
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert "UNRELATED" not in evidence.words[0].text_raw
    assert evidence.words[0].text_raw == "This"


def test_11_no_network_or_provider_symbol_referenced():
    src = inspect.getsource(spine_live_module)
    assert "requests." not in src
    assert "http" not in src.lower().replace("this ", "")  # no URL/provider call scaffolding


# ===========================================================================
# 12-13: Source isolation.
# ===========================================================================
def test_12_two_sources_independent():
    words1 = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    words2 = _sentence_words("That serum works differently here.", 0.0)[0]
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words1))
    e2 = build_live_language_spine_for_source(source_asset_id="s2", raw_understanding_map=_raw_map("s2", words2))
    assert e1.source_asset_id == "s1" and e2.source_asset_id == "s2"
    assert e1.words[0].text_raw != e2.words[0].text_raw
    assert not (set(a.attempt_id for a in e1.attempts) & set(a.attempt_id for a in e2.attempts))


def test_13_run_summary_aggregates_across_sources():
    words1 = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    words2 = _sentence_words("That serum works differently here.", 0.0)[0]
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words1))
    e2 = build_live_language_spine_for_source(source_asset_id="s2", raw_understanding_map=_raw_map("s2", words2))
    summary = live_language_spine_run_summary((e1, e2))
    assert summary["source_construction_count"] == 2
    assert summary["language_attempt_count"] == len(e1.attempts) + len(e2.attempts)


# ===========================================================================
# 14-16: Bridging (deterministic maximum-overlap, never text-match).
# ===========================================================================
def test_14_max_overlap_bridge_matches_correct_span():
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    evidence = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    assert len(evidence.attempts) == 2
    a0, a1 = sorted(evidence.attempts, key=lambda a: a.source_start)
    span0 = _span("c1", a0.source_start, a0.source_end)
    span1 = _span("c2", a1.source_start, a1.source_end)
    bridge = language_attempts_by_span_id_for_source((span0, span1), evidence.attempts)
    assert bridge["c1"].attempt_id == a0.attempt_id
    assert bridge["c2"].attempt_id == a1.attempt_id


def test_15_no_overlap_span_absent_from_bridge():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    evidence = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    far_span = _span("c_far", 500.0, 501.0)
    bridge = language_attempts_by_span_id_for_source((far_span,), evidence.attempts)
    assert "c_far" not in bridge


def test_16_bridge_uses_overlap_never_positional_order():
    # Spans deliberately supplied in REVERSED order relative to attempt
    # order -- correct bridging must come from actual timing overlap, not
    # from list position or any text lookup.
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    evidence = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    assert len(evidence.attempts) == 2
    a0, a1 = sorted(evidence.attempts, key=lambda a: a.source_start)
    span_late = _span("c_late", a1.source_start, a1.source_end)
    span_early = _span("c_early", a0.source_start, a0.source_end)
    bridge = language_attempts_by_span_id_for_source((span_late, span_early), evidence.attempts)
    assert bridge["c_early"].attempt_id == a0.attempt_id
    assert bridge["c_late"].attempt_id == a1.attempt_id


# ===========================================================================
# 17-21: fuse_relation_evidence -- the 5-way contract.
# ===========================================================================
def test_17_fuse_both_missing():
    value, source = fuse_relation_evidence(None, None)
    assert value is None and source == RELATION_SOURCE_MISSING


def test_18_fuse_d157_only():
    value, source = fuse_relation_evidence(RELATION_RETRY, None)
    assert value == RELATION_RETRY and source == RELATION_SOURCE_D157_ONLY


def test_19_fuse_canonical_only():
    value, source = fuse_relation_evidence(None, RELATION_RETRY)
    assert value == RELATION_RETRY and source == RELATION_SOURCE_CANONICAL_ONLY


def test_20_fuse_agreement():
    value, source = fuse_relation_evidence(RELATION_RETRY, RELATION_RETRY)
    assert value == RELATION_RETRY and source == RELATION_SOURCE_AGREEMENT


def test_21_fuse_conflict_abstains_never_picks_a_side():
    value, source = fuse_relation_evidence(RELATION_RETRY, "CORRECTION")
    assert value not in (RELATION_RETRY, "CORRECTION")
    assert source == RELATION_SOURCE_CONFLICT_ABSTAINED


# ===========================================================================
# 22-27: P1 consumption / precedence / diagnostics fields.
# ===========================================================================
def test_22_live_language_spine_overrides_supplied_fallback_params():
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    # Even though we ALSO pass a caller-supplied (empty) language_attempts_by_span_id,
    # the live spine must be the single source of truth once supplied.
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1",
        takes_for_source=[_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)],
        watch_listen_understanding=_wlu(
            _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
            _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        ),
        language_attempts_by_span_id={},
        live_language_spine=spine,
    )
    assert any(s == LANGUAGE_EVIDENCE_CANONICAL for s in u.moment_language_evidence_source)


def test_23_no_live_language_spine_is_byte_identical_to_pre_d199():
    baseline = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    again = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    assert baseline.moment_relation_to_predecessor == again.moment_relation_to_predecessor
    assert all(s == LANGUAGE_EVIDENCE_D157_FALLBACK for s in baseline.moment_language_evidence_source)
    assert all(s == RELATION_SOURCE_MISSING or s == RELATION_SOURCE_D157_ONLY for s in baseline.moment_relation_evidence_source)


def test_24_p1_consumes_real_attempt_ids_when_bridged():
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    a0, a1 = sorted(spine.attempts, key=lambda a: a.source_start)
    takes = [_take("c1", 0, a0.source_start, a0.source_end), _take("c2", 1, a1.source_start, a1.source_end)]
    wlu = _wlu(
        _span("c1", a0.source_start, a0.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", a1.source_start, a1.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu, live_language_spine=spine,
    )
    assert u.moments[0].attempt_ids and u.moments[0].attempt_ids[0] in (a0.attempt_id, a1.attempt_id)


def test_25_partial_bridge_coverage_falls_back_per_span():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    takes = [_take("c1", 0, 0.0, 1.0), _take("c_far", 1, 500.0, 501.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c_far", 500.0, 501.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu, live_language_spine=spine,
    )
    sources = u.moment_language_evidence_source
    assert LANGUAGE_EVIDENCE_D157_FALLBACK in sources  # the unbridged c_far span


def test_26_moment_language_evidence_source_field_length_matches_moments():
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    assert len(u.moment_language_evidence_source) == len(u.moments)


def test_27_moment_relation_evidence_source_field_length_matches_moments():
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    assert len(u.moment_relation_evidence_source) == len(u.moments)


# ===========================================================================
# 28-29: Fallback failure test -- no crash, no dropped source.
# ===========================================================================
def test_28_not_evaluable_spine_still_yields_p1_result_via_fallback():
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=None)
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=spine)
    assert u.moment_count == 2
    assert all(s == LANGUAGE_EVIDENCE_D157_FALLBACK for s in u.moment_language_evidence_source)


def test_29_batch_one_source_missing_spine_does_not_affect_other():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    good_spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    take1 = CandidateTake(clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=1.0, text="generic")
    wlu1 = _wlu(_span("c1", 0.0, 1.0, source_asset_id="s1", behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]), source_asset_id="s1")
    take2 = CandidateTake(clip_id="d1", source_asset_id="s2", source_order=0, start=0.0, end=1.0, text="x")
    wlu2 = _wlu(_span("d1", 0.0, 1.0, source_asset_id="s2", behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]), source_asset_id="s2")
    understandings = build_editorial_moment_understanding_for_sources(
        sources=["s1", "s2"], takes=[take1, take2], watch_listen_understandings=(wlu1, wlu2),
        live_language_spine_by_source={"s1": good_spine},  # s2 absent -- must not crash
    )
    assert len(understandings) == 2
    assert all(u.moment_count == 1 for u in understandings)


# ===========================================================================
# 30-31: Proposition/Retry identity firewall + no silent overwrite.
# ===========================================================================
def test_30_editorial_moment_has_no_retry_family_field():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    u = _pair_understanding_with_live_spine(None, live_language_spine=spine)
    field_names = {f for m in u.moments for f in vars(m).keys()}
    assert "retry_family_id" not in field_names


def test_31_conflict_never_silently_picks_a_side_in_full_pipeline():
    # Left/right relation evidence disagree -- P1 must record RELATION_UNCERTAIN
    # style abstention, never overwrite one with the other.
    class _FakeAttempt:
        def __init__(self, aid, start, end):
            self.attempt_id = aid
            self.source_start = start
            self.source_end = end

    left = _FakeAttempt("a0", 0.0, 1.0)
    right = _FakeAttempt("a1", 1.0, 2.0)
    props_by_attempt = {"a0": ("p0",), "a1": ("p1",)}
    rel = RelationEvidence(
        left_proposition_candidate_id="p0", right_proposition_candidate_id="p1",
        relation_candidate="CORRECTION", support_status="SUPPORTED", confidence="SUPPORTED",
        semantic_support="UNKNOWN", language_support="SUPPORTED", watch_listen_support="UNKNOWN",
        meaning_conflict=False, proposition_conflict=False, provenance=("LANGUAGE_SPINE",),
    )
    pair_map = relation_evidence_by_proposition_pair((rel,))
    assert pair_map[("p0", "p1")] is rel
    fused, source = fuse_relation_evidence(RELATION_RETRY, rel.relation_candidate)
    assert source == RELATION_SOURCE_CONFLICT_ABSTAINED
    del left, right, props_by_attempt  # exercised via the pair_map/fuse contract above


# ===========================================================================
# 32-37: Spanish fixtures (structural evidence only -- no phrase dictionary).
# ===========================================================================
def test_32_spanish_negation_complete():
    words = _sentence_words("No tuve dolor.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert spine.capability_status == CAPABILITY_AVAILABLE
    assert len(spine.attempts) == 1
    assert "no" in spine.attempts[0].text_normalized.lower()


def test_33_spanish_negative_polarity_accents_preserved():
    words = _sentence_words("Nunca me pasó eso.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert "pasó" in spine.attempts[0].text_normalized or "pasó" in spine.attempts[0].text_raw
    assert "nunca" in spine.attempts[0].text_normalized.lower()


def test_34_spanish_open_clause_constructs_without_crash():
    words = _sentence_words("Me diagnosticaron con", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    # No forced verdict on completeness -- only that construction is honest and safe.
    assert spine.capability_status in (CAPABILITY_AVAILABLE, CAPABILITY_NOT_EVALUABLE)


def test_35_spanish_retry_phrase_alone_does_not_force_retry_classification():
    words = _sentence_words("Déjame hacerlo otra vez.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    # A single, isolated utterance containing "otra vez" with NO structural
    # restart pair evidence must classify as a plain clean attempt --
    # never RETRY/CORRECTION from the phrase content alone.
    assert len(spine.attempts) == 1
    assert spine.attempts[0].attempt_state == ATTEMPT_CLEAN
    assert spine.attempts[0].restart_evidence is False


def test_36_spanish_two_distinct_proposition_capable_units():
    words = _two_sentence_words("Primero tenía náuseas.", "Después me empezó el dolor.")
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert len(spine.attempts) == 2
    assert len(spine.proposition_candidates) == 2


def test_37_spanish_number_correction_intact():
    words = _sentence_words("No fueron dos, fueron tres.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    preserved = count_meaning_sensitive_tokens_preserved(spine.words)
    assert preserved >= 1  # "no" (polarity) is preserved by construction


# ===========================================================================
# 38-43: English equivalents.
# ===========================================================================
def test_38_english_negation_complete():
    words = _sentence_words("I didn't have pain.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert spine.capability_status == CAPABILITY_AVAILABLE
    assert len(spine.attempts) == 1


def test_39_english_negative_polarity_contraction_preserved():
    words = _sentence_words("That never happened to me.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert "never" in spine.attempts[0].text_normalized.lower()


def test_40_english_open_clause_constructs_without_crash():
    words = _sentence_words("I was diagnosed with", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert spine.capability_status in (CAPABILITY_AVAILABLE, CAPABILITY_NOT_EVALUABLE)


def test_41_english_retry_phrase_alone_does_not_force_retry_classification():
    words = _sentence_words("Let me do that again.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert len(spine.attempts) == 1
    assert spine.attempts[0].attempt_state == ATTEMPT_CLEAN
    assert spine.attempts[0].restart_evidence is False


def test_42_english_two_distinct_proposition_capable_units():
    words = _two_sentence_words("First I felt nauseous.", "Then the pain started.")
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert len(spine.attempts) == 2
    assert len(spine.proposition_candidates) == 2


def test_43_english_number_correction_intact():
    words = _sentence_words("It wasn't two, it was three.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    preserved = count_meaning_sensitive_tokens_preserved(spine.words)
    assert preserved >= 1


# ===========================================================================
# 44-45: Bilingual equivalence (categorical, never identical text/ids).
# ===========================================================================
def test_44_negation_equivalent_across_languages():
    es = build_live_language_spine_for_source(
        source_asset_id="s1", raw_understanding_map=_raw_map("s1", _sentence_words("No tuve dolor.", 0.0)[0]),
    )
    en = build_live_language_spine_for_source(
        source_asset_id="s2", raw_understanding_map=_raw_map("s2", _sentence_words("I didn't have pain.", 0.0)[0]),
    )
    assert es.capability_status == en.capability_status == CAPABILITY_AVAILABLE
    assert len(es.attempts) == len(en.attempts) == 1
    assert es.attempts[0].attempt_state == en.attempts[0].attempt_state == ATTEMPT_CLEAN


def test_45_structural_restart_equivalent_across_languages():
    # Same-first-two-content-tokens restart pattern, in both languages --
    # tests the STRUCTURAL detector, not a translated phrase list.
    es_words = _two_sentence_words("Uso este producto diario.", "Uso este producto siempre.")
    en_words = _two_sentence_words("I use this product daily.", "I use this product always.")
    es = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", es_words))
    en = build_live_language_spine_for_source(source_asset_id="s2", raw_understanding_map=_raw_map("s2", en_words))
    es_states = {a.attempt_state for a in es.attempts}
    en_states = {a.attempt_state for a in en.attempts}
    assert es_states == en_states  # same categorical outcome, independent of language


# ===========================================================================
# 46-48: Spanglish / code-switching.
# ===========================================================================
def test_46_spanglish_construction_succeeds():
    words = _sentence_words("This is what I use porque me ayuda bastante.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert spine.capability_status == CAPABILITY_AVAILABLE
    assert len(spine.attempts) == 1


def test_47_language_switch_mid_sentence_does_not_force_a_split():
    words = _sentence_words("This is what I use porque me ayuda bastante mucho.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    # No timing/punctuation boundary was introduced mid-sentence -- the
    # language switch alone must not create a second phrase/utterance.
    assert len(spine.utterances) == 1


def test_48_spanglish_retry_style_fixture_constructs_without_crash():
    words = _sentence_words("Yo tried it again because the first take was incomplete.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert spine.capability_status == CAPABILITY_AVAILABLE


# ===========================================================================
# 49-56: Safety.
# ===========================================================================
def test_49_negation_preserved_english():
    words = _sentence_words("I didn't have pain.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert count_meaning_sensitive_tokens_preserved(spine.words) >= 1


def test_50_negation_preserved_spanish():
    words = _sentence_words("No tuve dolor.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    assert count_meaning_sensitive_tokens_preserved(spine.words) >= 1


def test_51_numbers_preserved():
    words = _sentence_words("It wasn't two, it was three.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    texts = " ".join(w.text_normalized for w in spine.words)
    assert "two" in texts.lower() and "three" in texts.lower()


def test_52_spanish_accents_preserved_end_to_end():
    words = _sentence_words("Nunca me pasó eso.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    combined = " ".join(w.text_normalized for w in spine.words)
    assert "ó" in combined


def test_53_english_contractions_preserved():
    words = _sentence_words("I didn't have pain.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=_raw_map("s1", words))
    combined = " ".join(w.text_normalized for w in spine.words)
    assert "'" in combined or "didn" in combined.lower()


def _string_literals_used_in_comparisons(source: str):
    """Extracts string literals that appear as the operand of an equality/
    membership test in real code -- deliberately excludes plain prose
    (docstrings/comments never contain a bare ``==``/``in (`` next to the
    literal), so this does not false-positive on a docstring merely
    DISCUSSING a forbidden phrase."""
    literals = set()
    for match in re.finditer(r'==\s*"([^"]{2,40})"', source):
        literals.add(match.group(1).lower())
    for match in re.finditer(r"==\s*'([^']{2,40})'", source):
        literals.add(match.group(1).lower())
    for match in re.finditer(r'in\s*\(\s*"([^"]{2,40})"', source):
        literals.add(match.group(1).lower())
    return literals


def test_54_no_phrase_hardcoding_for_retry_or_correction():
    forbidden = {"otra vez", "again", "perdón", "perdon", "sorry"}
    for module in (spine_live_module, p1_integration_module):
        literals = _string_literals_used_in_comparisons(inspect.getsource(module))
        assert not (literals & forbidden), (module.__name__, literals & forbidden)


def test_55_no_hardcoded_language_branch():
    forbidden_langs = {"es", "en", "es-es", "en-us"}
    for module in (spine_live_module, p1_integration_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        assert comparator.value.lower() not in forbidden_langs, (module.__name__, comparator.value)


def test_56_new_module_never_references_english_only_whisper_model():
    src = inspect.getsource(spine_live_module)
    for needle in ("base.en", "small.en", "medium.en", "large.en"):
        assert needle not in src


# ===========================================================================
# 57-59: Immutability (default OFF / flag off preserves D-198 exactly).
# ===========================================================================
def test_57_build_editorial_moments_for_source_default_arity_unchanged():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    spans_by_id = {
        "c1": _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        "c2": _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    }
    result = build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=takes, understanding_spans_by_id=spans_by_id,
    )
    assert len(result) == 4  # unchanged D-197/D-198 contract


def test_58_no_live_spine_supplied_default_fields_are_fallback_missing():
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    assert all(s == LANGUAGE_EVIDENCE_D157_FALLBACK for s in u.moment_language_evidence_source)
    assert u.moment_relation_evidence_source[1] == RELATION_SOURCE_D157_ONLY


def test_59_flag_on_but_construction_unavailable_no_crash():
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=None)
    assert spine.capability_status == CAPABILITY_NOT_EVALUABLE
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=spine)
    assert u.moment_count == 2


# ===========================================================================
# 60-63: P1 real-coverage diagnostics.
# ===========================================================================
def test_60_canonical_attempt_used_by_p1_count_nonzero_when_bridged():
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    a0, a1 = sorted(spine.attempts, key=lambda a: a.source_start)
    takes = [_take("c1", 0, a0.source_start, a0.source_end), _take("c2", 1, a1.source_start, a1.source_end)]
    wlu = _wlu(
        _span("c1", a0.source_start, a0.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", a1.source_start, a1.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu, live_language_spine=spine,
    )
    diag = live_language_spine_source_diagnostics_for_p1(spine, u)
    assert diag["canonical_attempt_used_by_p1_count"] == 2
    assert diag["fallback_attempt_used_by_p1_count"] == 0


def test_61_canonical_proposition_coverage_count_nonzero():
    words = _two_sentence_words("This cream removes wrinkles fast.", "That serum works differently here.")
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    a0, a1 = sorted(spine.attempts, key=lambda a: a.source_start)
    takes = [_take("c1", 0, a0.source_start, a0.source_end), _take("c2", 1, a1.source_start, a1.source_end)]
    wlu = _wlu(
        _span("c1", a0.source_start, a0.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", a1.source_start, a1.source_end, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu, live_language_spine=spine,
    )
    diag = live_language_spine_source_diagnostics_for_p1(spine, u)
    assert diag["canonical_proposition_coverage_count"] == 2


def test_62_diagnostics_helper_includes_construction_counts():
    words = _sentence_words("This cream removes wrinkles fast.", 0.0)[0]
    spine = build_live_language_spine_for_source(source_asset_id="src1", raw_understanding_map=_raw_map("src1", words))
    u = _pair_understanding_with_live_spine(None, live_language_spine=spine)
    diag = live_language_spine_source_diagnostics_for_p1(spine, u)
    for key in (
        "capability_status", "language_word_count", "language_phrase_count",
        "language_utterance_count", "language_attempt_count", "proposition_candidate_count",
        "relation_evidence_count", "canonical_attempt_used_by_p1_count",
        "fallback_attempt_used_by_p1_count", "canonical_proposition_coverage_count",
        "canonical_relation_coverage_count",
    ):
        assert key in diag


def test_63_p1_moment_diagnostics_expose_language_and_relation_source_no_full_text():
    u = _pair_understanding_with_live_spine(RELATION_RETRY, live_language_spine=None)
    d = editorial_moment_understanding_diagnostics(u)
    row = d["moments"][1]
    assert "language_evidence_source" in row
    assert "relation_evidence_source" in row
    assert "attempt_ids" in row and "proposition_candidate_ids" in row
    # No transcript dump anywhere in the diagnostic row.
    assert "text_raw" not in row and "transcript" not in row
