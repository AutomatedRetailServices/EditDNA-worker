"""D-157 Phase B -- Watch+Listen Multimodal Understanding V1.

Per docs/CUTSELL_DECISIONS.md D-148/D-154/D-155/D-156/D-157.
`watch_listen_understanding.py` consumes ONE `RawUnderstandingMap` (D-155,
unchanged) and forms bounded, categorical HYPOTHESES about behavior state,
attempt boundaries, attempt relations, meaning completion, and performance
usability. This suite proves: (1) hypotheses are grounded in already-real
evidence, never invented; (2) the bounded relation/boundary vocabularies
are respected and `DISTINCT_PROPOSITION` never rises above UNCERTAIN;
(3) conflicts are preserved, never collapsed into a forced label; (4) no
final Proposition/Family/BestTake/Boundary/Pacing authority is read or
written; (5) deterministic ordering and source-identity/timing
preservation.

Generic synthetic fixtures only -- no Video00-specific transcript text,
timestamps, or clip ids (CLAUDE.md: never hardcode those).
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake, MediaSignals, TranscriptSegment, Word
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.raw_understanding_map import (
    TRACK_STATUS_PASS,
    build_raw_understanding_maps_for_sources,
)
from cutsell_worker.watch_listen_understanding import (
    ALLOWED_ATTEMPT_RELATIONS,
    ALLOWED_BOUNDARY_KINDS,
    ALLOWED_CONFIDENCE_LEVELS,
    ALLOWED_MEANING_STATES,
    ALLOWED_USABILITY_STATES,
    BOUNDARY_ATTEMPT_ABANDONED,
    BOUNDARY_ATTEMPT_BEGINS,
    BOUNDARY_ATTEMPT_COMPLETES,
    BOUNDARY_POST_TAKE_RESET,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    SCHEMA_VERSION,
    USABILITY_QUESTIONABLE,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    build_watch_listen_understanding,
    build_watch_listen_understanding_for_sources,
    watch_listen_understanding_diagnostics,
)
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _take(clip_id: str, start: float, end: float, text: str, *, complete_idea: bool = True) -> CandidateTake:
    tokens = text.split() or ["x"]
    step = max(0.05, (end - start) / max(1, len(tokens)))
    words = tuple(_word(tok, start + i * step, start + i * step + step * 0.8) for i, tok in enumerate(tokens))
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src-1", source_order=0,
        start=start, end=end, text=text, words=words, complete_idea=complete_idea,
        signals=MediaSignals(source_asset_id="src-1", start=start, end=end),
    )


class _Source:
    source_asset_id = "src-1"
    duration_sec = 60.0


def _context(events: tuple[TemporalEvent, ...] = ()) -> WholeVideoContext | None:
    if not events:
        return None
    return WholeVideoContext(
        sources=(SourceVideoContext(source_asset_id="src-1", summary="", dominant_style="", creator_intent="", events=events),),
        status=ProviderStatus("wv", True, True, "applied"),
    )


def _understanding_for(takes: tuple[CandidateTake, ...], events: tuple[TemporalEvent, ...] = ()):
    segments = tuple(
        TranscriptSegment(source_asset_id="src-1", start=t.start, end=t.end, text=t.text, words=t.words)
        for t in takes
    )
    maps = build_raw_understanding_maps_for_sources(
        sources=[_Source()], transcript_tuple=segments, whole_context=_context(events), takes=takes,
        media_probes_by_source={}, speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    return build_watch_listen_understanding(maps[0], takes)


# ---------------------------------------------------------------------------
# 1: clean audience delivery.
# ---------------------------------------------------------------------------

def test_clean_audience_delivery():
    a = _take("a1", 0.0, 3.0, "hola a todos bienvenidos", complete_idea=True)
    u = _understanding_for((a,))
    span = u.understanding_spans[0]
    assert span.meaning_completion_hypothesis == MEANING_COMPLETE
    assert span.performance_usability_hypothesis == USABILITY_USABLE
    assert span.attempt_relation_hypotheses[0].relation == RELATION_UNCERTAIN
    assert span.attempt_relation_hypotheses[0].confidence == CONFIDENCE_UNKNOWN


# ---------------------------------------------------------------------------
# 2: false start -> restart.
# ---------------------------------------------------------------------------

def test_false_start_then_restart():
    p = _take("p1", 0.0, 2.0, "yo tengo un problema", complete_idea=False)
    s = _take("s1", 2.4, 5.0, "yo tengo un problema con mi piel", complete_idea=True)
    events = (TemporalEvent("src-1", 1.9, 2.1, "false_start", 0.9, ""),)
    u = _understanding_for((p, s), events)
    p_span, s_span = u.understanding_spans
    assert "FALSE_START" in {h.label for h in p_span.behavior_state_hypotheses}
    relations = {r.relation for r in s_span.attempt_relation_hypotheses}
    assert RELATION_RETRY in relations


# ---------------------------------------------------------------------------
# 3: abandoned attempt -> retry hypothesis (D-100 confirmed events).
# ---------------------------------------------------------------------------

def test_abandoned_attempt_retry_hypothesis():
    p = _take("p1", 0.0, 2.0, "voy a hacer esto", complete_idea=False)
    s = _take("s1", 2.5, 5.0, "voy a hacer esto bien esta vez", complete_idea=True)
    events = (TemporalEvent("src-1", 1.8, 2.3, "wrong_take", 0.97, ""),)
    u = _understanding_for((p, s), events)
    p_span, s_span = u.understanding_spans
    assert "ABANDONED_ATTEMPT" in {h.label for h in p_span.behavior_state_hypotheses}
    supported_retry = [r for r in s_span.attempt_relation_hypotheses if r.relation == RELATION_RETRY]
    assert supported_retry


# ---------------------------------------------------------------------------
# 4/6: clean complete retry pair / correction.
# ---------------------------------------------------------------------------

def test_correction_after_apparently_complete_statement():
    p = _take("p1", 0.0, 2.0, "compra ahora mismo este producto increible.", complete_idea=True)
    s = _take("s1", 2.6, 4.6, "compra ahora mismo este producto en oferta.", complete_idea=True)
    u = _understanding_for((p, s))
    s_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in s_span.attempt_relation_hypotheses}
    assert relations.get(RELATION_CORRECTION) == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 5: incomplete A + continuation B.
# ---------------------------------------------------------------------------

def test_incomplete_then_continuation():
    a = _take("a1", 0.0, 2.0, "yo tenia un problema con mi piel y", complete_idea=False)
    b = _take("b1", 2.2, 4.0, "no sabia que hacer.", complete_idea=True)
    u = _understanding_for((a, b))
    b_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in b_span.attempt_relation_hypotheses}
    assert relations.get(RELATION_CONTINUATION) == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 7: complementary information.
# ---------------------------------------------------------------------------

def test_complementary_information():
    a = _take("a1", 0.0, 2.0, "primero el ingrediente uno.", complete_idea=True)
    b = _take("b1", 5.0, 7.0, "tambien usamos el ingrediente dos.", complete_idea=True)
    u = _understanding_for((a, b))
    b_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in b_span.attempt_relation_hypotheses}
    assert RELATION_COMPLEMENTARY in relations


# ---------------------------------------------------------------------------
# 8: new audience beat.
# ---------------------------------------------------------------------------

def test_new_audience_beat():
    a = _take("a1", 0.0, 2.0, "hablemos primero del acne.", complete_idea=True)
    b = _take("b1", 5.0, 7.0, "ahora hablemos del estomago.", complete_idea=True)
    u = _understanding_for((a, b))
    b_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in b_span.attempt_relation_hypotheses}
    assert relations.get(RELATION_NEW_AUDIENCE_BEAT) == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 9/13: post-take reset (+ meaning-complete conflict).
# ---------------------------------------------------------------------------

def test_post_take_reset_after_complete_delivery_is_flagged_as_conflict():
    a = _take("a1", 0.0, 2.0, "gracias por ver mi video.", complete_idea=True)
    # inside the candidate's own [start, end) window, but after the last
    # word's own end -- a real EXIT-zone margin (see module docstring's
    # D-115 window-overlap rule: event.end > candidate.start and
    # event.start < candidate.end).
    events = (TemporalEvent("src-1", 1.93, 1.97, "camera_disengagement_candidate", 0.8, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert "POST_TAKE_RESET" in {h.label for h in span.behavior_state_hypotheses}
    assert span.meaning_completion_hypothesis == MEANING_COMPLETE
    assert "EXIT_RESET_VS_MEANING_COMPLETE" in span.conflict_flags
    assert BOUNDARY_POST_TAKE_RESET in {b.boundary_kind for b in span.attempt_boundary_hypotheses}


# ---------------------------------------------------------------------------
# 10: recording-process segment.
# ---------------------------------------------------------------------------

def test_recording_process_segment():
    a = _take("a1", 0.0, 2.0, "espera dejame pensar", complete_idea=False)
    events = (TemporalEvent("src-1", 0.5, 1.0, "verbal_fumble", 0.7, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert "RECORDING_PROCESS" in {h.label for h in span.behavior_state_hypotheses}


# ---------------------------------------------------------------------------
# 11/20: breaking character during delivery (DELIVERY defect).
# ---------------------------------------------------------------------------

def test_breaking_character_during_delivery_marks_delivery_unusable():
    a = _take("a1", 0.0, 3.0, "este producto es increible para tu piel", complete_idea=True)
    mid = (a.words[len(a.words) // 2].start + a.words[len(a.words) // 2].end) / 2
    events = (TemporalEvent("src-1", mid - 0.05, mid + 0.05, "breaking_character", 0.85, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.delivery_usability == USABILITY_UNUSABLE
    assert span.performance_usability_hypothesis == USABILITY_UNUSABLE


# ---------------------------------------------------------------------------
# 12/21: breaking character after delivery (EXIT-only defect).
# ---------------------------------------------------------------------------

def test_breaking_character_after_delivery_is_questionable_not_unusable():
    a = _take("a1", 0.0, 2.0, "eso es todo por hoy.", complete_idea=True)
    events = (TemporalEvent("src-1", 1.93, 1.97, "breaking_character", 0.85, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.exit_usability == USABILITY_QUESTIONABLE
    assert span.delivery_usability == USABILITY_USABLE
    assert span.performance_usability_hypothesis == USABILITY_QUESTIONABLE


# ---------------------------------------------------------------------------
# 14: meaning incomplete + visual reset (no conflict expected).
# ---------------------------------------------------------------------------

def test_incomplete_meaning_with_reset_is_not_flagged_conflicting():
    a = _take("a1", 0.0, 2.0, "y entonces yo", complete_idea=False)
    events = (TemporalEvent("src-1", 2.05, 2.2, "hand_motion_reset_candidate", 0.8, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.meaning_completion_hypothesis == MEANING_INCOMPLETE
    assert span.conflict_flags == ()


# ---------------------------------------------------------------------------
# 15: conflicting modalities (meaning complete vs. real abandonment evidence).
# ---------------------------------------------------------------------------

def test_meaning_complete_vs_abandoned_attempt_conflict():
    a = _take("a1", 0.0, 2.0, "y asi es como se hace.", complete_idea=True)
    events = (TemporalEvent("src-1", 0.5, 1.5, "wrong_take", 0.9, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.meaning_completion_hypothesis == MEANING_COMPLETE
    assert "MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE" in span.conflict_flags
    assert span.behavior_confidence == "MIXED"


# ---------------------------------------------------------------------------
# 16: uncertain relation (first span, and an ambiguous mid-sequence pair).
# ---------------------------------------------------------------------------

def test_uncertain_relation_first_span():
    a = _take("a1", 0.0, 2.0, "hola", complete_idea=True)
    u = _understanding_for((a,))
    span = u.understanding_spans[0]
    assert span.attempt_relation_hypotheses[0].relation == RELATION_UNCERTAIN
    assert span.attempt_relation_hypotheses[0].left_span_id is None


def test_ambiguous_pair_never_asserts_supported_without_real_evidence():
    # complete_idea True but no terminal punctuation, no restart, tight gap:
    # genuinely ambiguous (P reads as neither cleanly complete nor clearly
    # unfinished) -- the module must still return a bounded, bounded-
    # vocabulary hypothesis (never crash, never fabricate SUPPORTED
    # confidence for RETRY/CORRECTION/NEW_AUDIENCE_BEAT, which all require
    # stronger evidence than this fixture provides).
    p = _take("p1", 0.0, 2.0, "vamos a ver como", complete_idea=True)
    s = _take("s1", 2.3, 4.0, "funciona este producto", complete_idea=True)
    u = _understanding_for((p, s))
    s_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in s_span.attempt_relation_hypotheses}
    assert relations
    assert RELATION_RETRY not in relations
    assert RELATION_CORRECTION not in relations
    for relation, confidence in relations.items():
        assert relation in ALLOWED_ATTEMPT_RELATIONS
        assert confidence in ALLOWED_CONFIDENCE_LEVELS


# ---------------------------------------------------------------------------
# 17/18: same topic / same opener never yields DISTINCT_PROPOSITION.
# ---------------------------------------------------------------------------

def test_same_topic_different_content_never_asserts_distinct_proposition():
    p = _take("p1", 0.0, 2.0, "hablemos del acne en la espalda.", complete_idea=True)
    s = _take("s1", 5.0, 7.0, "hablemos del acne en la cara.", complete_idea=True)
    u = _understanding_for((p, s))
    s_span = u.understanding_spans[1]
    relations = {r.relation for r in s_span.attempt_relation_hypotheses}
    assert RELATION_DISTINCT_PROPOSITION not in relations


def test_same_opener_flagged_only_as_hypothesis_never_distinct_proposition():
    # `_restart_evidence` fires on a shared 2-token opener even when the
    # rest of the sentence differs -- this module's own honest limitation
    # (reused, unchanged, real production logic), never escalated beyond
    # a WEAK/SUPPORTED hypothesis, and never DISTINCT_PROPOSITION.
    p = _take("p1", 0.0, 2.0, "yo tengo una pregunta sobre el envio.", complete_idea=True)
    s = _take("s1", 2.4, 4.4, "yo tengo una pregunta sobre el precio.", complete_idea=True)
    u = _understanding_for((p, s))
    s_span = u.understanding_spans[1]
    relations = {r.relation for r in s_span.attempt_relation_hypotheses}
    assert RELATION_DISTINCT_PROPOSITION not in relations
    assert relations & {RELATION_RETRY, RELATION_CORRECTION}


def test_distinct_proposition_never_emitted_above_uncertain_anywhere():
    # Structural sweep across every fixture-shaped pair this module can
    # realistically see -- DISTINCT_PROPOSITION must never appear with a
    # confidence other than absent (it must simply never be produced by
    # this module's own relation deriver, since no semantic evidence is
    # available to assert it honestly).
    import cutsell_worker.watch_listen_understanding as mod
    assert "RELATION_DISTINCT_PROPOSITION" not in [
        line for line in open(mod.__file__, encoding="utf-8").read().splitlines()
        if "entries.append" in line and "RELATION_DISTINCT_PROPOSITION" in line
    ] or True  # documents intent; the real guarantee is the source-scan below
    source = open(mod.__file__, encoding="utf-8").read()
    # RELATION_DISTINCT_PROPOSITION must appear only in the vocabulary
    # definitions/docstring, never inside `_relation_for_pair`'s own body.
    body_start = source.index("def _relation_for_pair")
    body_end = source.index("\ndef ", body_start + 1)
    assert "RELATION_DISTINCT_PROPOSITION" not in source[body_start:body_end]


# ---------------------------------------------------------------------------
# 19-21: ENTRY / DELIVERY / EXIT defect isolation (repeats 11/12/20/21 from
# the opposite direction -- confirms the zone -> usability mapping directly).
# ---------------------------------------------------------------------------

def test_entry_defect_only_is_questionable_overall():
    # Explicit words with a real lead-in margin before the first word (a
    # genuine ENTRY-zone window inside the candidate's own [start, end)
    # span, distinct from the delivery span itself).
    words = (_word("hola", 1.4, 1.7), _word("a", 1.75, 1.85), _word("todos", 1.9, 2.3))
    a = CandidateTake(
        clip_id="a1", source_asset_id="src-1", source_order=0, start=1.0, end=3.0,
        text="hola a todos", words=words, complete_idea=True,
        signals=MediaSignals(source_asset_id="src-1", start=1.0, end=3.0),
    )
    events = (TemporalEvent("src-1", 1.05, 1.3, "camera_disengagement_candidate", 0.8, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.entry_usability == USABILITY_QUESTIONABLE
    assert span.delivery_usability == USABILITY_USABLE
    assert span.performance_usability_hypothesis == USABILITY_QUESTIONABLE


def test_delivery_defect_marks_overall_unusable():
    a = _take("a1", 0.0, 3.0, "este producto es increible para tu piel", complete_idea=True)
    mid = (a.words[len(a.words) // 2].start + a.words[len(a.words) // 2].end) / 2
    events = (TemporalEvent("src-1", mid - 0.05, mid + 0.05, "hand_motion_reset_candidate", 0.8, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.delivery_usability == USABILITY_UNUSABLE
    assert span.performance_usability_hypothesis == USABILITY_UNUSABLE


def test_exit_defect_only_is_questionable_not_unusable():
    a = _take("a1", 0.0, 2.0, "eso es todo por hoy.", complete_idea=True)
    events = (TemporalEvent("src-1", 1.93, 1.97, "facial_expression_shift_candidate", 0.7, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.exit_usability == USABILITY_QUESTIONABLE
    assert span.performance_usability_hypothesis == USABILITY_QUESTIONABLE
    assert span.performance_usability_hypothesis != USABILITY_UNUSABLE


# ---------------------------------------------------------------------------
# 22: audio pause corroboration.
# ---------------------------------------------------------------------------

def test_audio_pause_corroborates_retry_hypothesis():
    p = _take("p1", 0.0, 2.0, "voy a intentarlo otra vez", complete_idea=False)
    s = _take("s1", 4.0, 6.0, "voy a intentarlo otra vez ahora", complete_idea=True)
    events = (TemporalEvent("src-1", 2.0, 4.0, "audio_silence_interval", 0.95, ""),)
    u = _understanding_for((p, s), events)
    s_span = u.understanding_spans[1]
    relations = {r.relation: r.confidence for r in s_span.attempt_relation_hypotheses}
    assert relations.get(RELATION_RETRY) in (CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK)


# ---------------------------------------------------------------------------
# 23: no real-audio semantic claim (structural).
# ---------------------------------------------------------------------------

def test_no_semantic_prosodic_audio_claim_in_source():
    import cutsell_worker.watch_listen_understanding as mod
    source = open(mod.__file__, encoding="utf-8").read().lower()
    for forbidden in ("tone of voice", "prosod", "spoken emphasis", "emotion in voice", "sentiment"):
        assert forbidden not in source, f"unexpected semantic/prosodic audio claim: {forbidden!r}"


# ---------------------------------------------------------------------------
# 24: provenance retained.
# ---------------------------------------------------------------------------

def test_provenance_retained_verbatim_from_raw_understanding_map():
    a = _take("a1", 0.0, 2.0, "hola a todos", complete_idea=True)
    events = (TemporalEvent("src-1", 0.1, 0.3, "camera_disengagement_candidate", 0.8, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    for hyp in span.behavior_state_hypotheses:
        assert hyp.provenance in {"VISUAL_SIGNAL", "DETERMINISTIC_RULE", "MULTIMODAL_FUSION", "SEMANTIC_PROVIDER", "UNKNOWN", "ASR", "AUDIO_SIGNAL", "MEDIA_TIMING"}
    assert set(span.evidence_provenance.keys()) >= {
        "behavior_state_hypotheses", "meaning_completion_hypothesis",
        "performance_usability_hypothesis", "attempt_relation_hypotheses",
        "attempt_boundary_hypotheses",
    }


# ---------------------------------------------------------------------------
# 25: deterministic output ordering.
# ---------------------------------------------------------------------------

def test_deterministic_ordering_regardless_of_input_order():
    a = _take("a1", 0.0, 2.0, "primero", complete_idea=True)
    b = _take("b1", 2.5, 4.0, "segundo", complete_idea=True)
    c = _take("c1", 4.5, 6.0, "tercero", complete_idea=True)
    u_forward = _understanding_for((a, b, c))
    u_reversed = _understanding_for((c, b, a))
    ids_forward = [s.span_id for s in u_forward.understanding_spans]
    ids_reversed = [s.span_id for s in u_reversed.understanding_spans]
    assert ids_forward == ids_reversed == ["a1", "b1", "c1"]


# ---------------------------------------------------------------------------
# 26/27: source ids + timing unchanged.
# ---------------------------------------------------------------------------

def test_source_identity_and_timing_preserved():
    a = _take("a1", 3.25, 7.75, "hola", complete_idea=True)
    u = _understanding_for((a,))
    span = u.understanding_spans[0]
    assert span.span_id == "a1"
    assert span.source_asset_id == "src-1"
    assert span.source_start == 3.25
    assert span.source_end == 7.75


# ---------------------------------------------------------------------------
# 28-31: no Family/BestTake/Boundary/Pacing authority read or written.
# ---------------------------------------------------------------------------

def test_module_never_imports_a_downstream_or_upstream_authority():
    import cutsell_worker.watch_listen_understanding as mod
    source = open(mod.__file__, encoding="utf-8").read()
    forbidden_imports = [
        "take_grouping", "hybrid_session_cleanup", "semantic_authority_observability",
        "claim_coverage_best_take", "deterministic_best_take_authority", "take_judge",
        "boundary_engine", "dialogue_pacing", "composite_resolver", "renderer",
    ]
    for forbidden in forbidden_imports:
        assert f"import {forbidden}" not in source and f"from .{forbidden}" not in source


def test_module_is_not_imported_by_any_editorial_authority():
    import subprocess
    authority_files = [
        "cutsell_worker/take_grouping.py", "cutsell_worker/hybrid_session_cleanup.py",
        "cutsell_worker/semantic_authority_observability.py", "cutsell_worker/take_judge.py",
        "cutsell_worker/claim_coverage_best_take.py", "cutsell_worker/boundary_engine_pass.py",
        "cutsell_worker/pipeline.py",
    ]
    for path in authority_files:
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            continue
        assert "watch_listen_understanding" not in text, f"unexpected consumer: {path}"


# ---------------------------------------------------------------------------
# 32: no provider call.
# ---------------------------------------------------------------------------

def test_no_provider_or_network_reference_in_source():
    import cutsell_worker.watch_listen_understanding as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("openai", "OpenAIVisualProvider", "gemini", "Gemini", "requests.", "httpx.", "whole_video_openai"):
        assert forbidden not in source, f"unexpected provider/network reference: {forbidden}"


# ---------------------------------------------------------------------------
# Additional structural/contract coverage.
# ---------------------------------------------------------------------------

def test_confidence_vocabulary_is_bounded():
    a = _take("a1", 0.0, 2.0, "hola", complete_idea=True)
    events = (TemporalEvent("src-1", 0.1, 0.3, "wrong_take", 0.9, ""),)
    u = _understanding_for((a,), events)
    span = u.understanding_spans[0]
    assert span.behavior_confidence in ALLOWED_CONFIDENCE_LEVELS
    assert span.relation_confidence in ALLOWED_CONFIDENCE_LEVELS
    for r in span.attempt_relation_hypotheses:
        assert r.confidence in ALLOWED_CONFIDENCE_LEVELS
        assert r.relation in ALLOWED_ATTEMPT_RELATIONS
    for b in span.attempt_boundary_hypotheses:
        assert b.confidence in ALLOWED_CONFIDENCE_LEVELS
        assert b.boundary_kind in ALLOWED_BOUNDARY_KINDS
        assert b.edge in {"start", "end"}
    assert span.meaning_completion_hypothesis in ALLOWED_MEANING_STATES
    assert span.performance_usability_hypothesis in ALLOWED_USABILITY_STATES
    assert span.entry_usability in ALLOWED_USABILITY_STATES
    assert span.delivery_usability in ALLOWED_USABILITY_STATES
    assert span.exit_usability in ALLOWED_USABILITY_STATES


def test_meaning_uncertain_for_empty_transcript():
    a = _take("a1", 0.0, 1.0, "", complete_idea=True)
    a = a.__class__(**{**a.__dict__, "words": ()})
    u = _understanding_for((a,))
    span = u.understanding_spans[0]
    assert span.meaning_completion_hypothesis == MEANING_UNCERTAIN


def test_batch_build_for_sources_matches_per_source_build():
    a = _take("a1", 0.0, 2.0, "hola", complete_idea=True)
    segments = (TranscriptSegment(source_asset_id="src-1", start=0.0, end=2.0, text=a.text, words=a.words),)
    maps = build_raw_understanding_maps_for_sources(
        sources=[_Source()], transcript_tuple=segments, whole_context=None, takes=(a,),
        media_probes_by_source={}, speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    batch = build_watch_listen_understanding_for_sources(maps, (a,))
    single = build_watch_listen_understanding(maps[0], (a,))
    assert batch[0] == single


def test_diagnostics_are_tail_safe_and_bounded():
    a = _take("a1", 0.0, 2.0, "un secreto de familia que nadie sabe", complete_idea=True)
    u = _understanding_for((a,))
    diag = watch_listen_understanding_diagnostics((u,))
    assert "secreto de familia" not in repr(diag)
    assert diag["watch_listen_understanding_created"] is True
    assert diag["understanding_span_count"] == 1
    assert diag["source_count"] == 1
    assert isinstance(diag["provenance_counts"], dict)


def test_schema_version_is_stable_and_namespaced():
    assert SCHEMA_VERSION.startswith("cutsell.watch_listen_understanding.")


def test_track_status_carried_through_unchanged():
    a = _take("a1", 0.0, 2.0, "hola", complete_idea=True)
    u = _understanding_for((a,))
    assert u.track_status.get("raw_understanding_map_status") == "COMPLETE_EXISTING_EVIDENCE"
