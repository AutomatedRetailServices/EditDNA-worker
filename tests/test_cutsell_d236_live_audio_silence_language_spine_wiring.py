"""D-236: LIVE AUDIO-SILENCE EVIDENCE -> LANGUAGE SPINE WIRING -- tests.

See docs/CUTSELL_DECISIONS.md D-235Z (root cause) and D-236 (this fix).
D-235Z proved that `language_spine_live_integration.build_live_language_
spine_for_source` called `segment_language_phrases(words)` with ZERO
`audio_silence_intervals`, structurally foreclosing PAUSE/RESTART_
BOUNDARY strong utterance cuts regardless of transcript punctuation
quality -- collapsing a real sibling source's 12 LanguagePhrases into 1
LanguageUtterance -> 1 LanguageAttempt -> 1 PropositionCandidate.

D-236 threads `RawUnderstandingMap.audio_events` (D-155's own field,
itself sourced from `audio_silence.py`'s real ffmpeg `silencedetect`
pass, merged in `flow_b.py` -- never recomputed) into that one call site.
No new detector, no new threshold, no ASR/ffmpeg re-invocation, no
Freeze/materiality/repair/P1/P2/BestTake/Family/Ordering/Boundary/
Pacing/Audio-Join change.

This file proves:
  1-10:  the wiring itself (consumption, isolation, boundary creation,
         coexistence with punctuation, timing fidelity).
  11-14: identity/determinism is preserved.
  15-18: the D-235Z collapse shape reproduces BEFORE this fix and
         resolves AFTER it, on a GENERIC fixture (no real transcript
         hardcoded), with no target count asserted.
  19-22: D-235X exact-identity compatibility (1->1, 1->N partition,
         distinctness) is preserved, unmodified.
  23-25: bilingual/code-switching safety (structural evidence only).
  26-36: explicit negative-authorization proofs (no new detector/
         threshold/ASR-rerun/ffmpeg-rerun/provider/RAW; no P1/P2/
         BestTake/Family/Ordering/Boundary/Freeze/materiality/repair/
         Pacing/Audio-Join mutation).
"""
from __future__ import annotations

import inspect

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.language_proposition_relation import build_proposition_candidates
from cutsell_worker.language_spine import (
    BOUNDARY_PAUSE,
    BOUNDARY_SPEECH_BOUNDARY,
    adapt_words_to_language_words,
    segment_language_phrases,
)
import cutsell_worker.language_spine_live_integration as spine_live_module
from cutsell_worker.language_spine_live_integration import (
    CAPABILITY_AVAILABLE,
    LiveLanguageSpineEvidence,
    _audio_silence_intervals_from_raw_understanding_map,
    build_live_language_spine_for_source,
    live_language_spine_diagnostics,
    live_language_spine_run_summary,
)
from cutsell_worker.language_utterance_attempt import (
    build_language_attempts,
    segment_language_utterances,
)
from cutsell_worker.raw_understanding_map import RawUnderstandingMap
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION,
    build_attempt_language_identity_matches_for_source,
)
from cutsell_worker.whole_video_analysis import TemporalEvent


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------
def _words(text: str, t0: float, *, per_word: float = 0.25, inner_gap: float = 0.05):
    words = []
    t = t0
    for tok in text.split(" "):
        start, end = t, t + per_word
        words.append(Word(text=tok, start=start, end=end, confidence=0.9))
        t = end + inner_gap
    return tuple(words), t


def _silence_event(source_asset_id: str, start: float, end: float) -> TemporalEvent:
    return TemporalEvent(
        source_asset_id=source_asset_id, start=start, end=end,
        kind="audio_silence_interval", confidence=1.0, description="real ffmpeg silencedetect",
    )


def _raw_map(source_asset_id: str, words, *, audio_events=()):
    return RawUnderstandingMap(
        source_asset_id=source_asset_id, source_duration=(max((w.end for w in words), default=0.0) + 1.0),
        source_timeline_origin="source_relative_seconds", transcript="unused -- word_timings is authoritative",
        word_timings=tuple(words), audio_events=tuple(audio_events),
    )


def _d235z_collapse_shape_words(source_asset_id: str = "sibling"):
    """Generic fixture structurally matching D-235Z's real-run shape: 12
    contiguous phrase-worthy chunks joined by timing gaps AT the
    ``DEFAULT_SPLIT_GAP_SEC`` (0.75s) WEAK-boundary threshold, with NO
    terminal punctuation anywhere and NO audio-silence evidence --
    exactly the conditions that collapsed the real sibling's 12
    LanguagePhrases into 1 LanguageUtterance (WEAK ``SPEECH_BOUNDARY``
    splits phrases but never forces an utterance cut on its own). No
    real transcript text is used (generic "chunk N" tokens only)."""
    words = []
    t = 0.0
    for i in range(12):
        chunk_words, t_after = _words(f"chunk {i} continues talking here", t, inner_gap=0.05)
        words.extend(chunk_words)
        t = t_after + 0.85  # >= DEFAULT_SPLIT_GAP_SEC (0.75s): a real WEAK phrase split, never a strong one alone
    return tuple(words)


# ===========================================================================
# 1-10: the wiring itself.
# ===========================================================================
def test_01_silence_intervals_consumed_from_raw_understanding_map():
    words, t_end = _words("hello there friend", 0.0)
    raw_map = _raw_map("s1", words, audio_events=(_silence_event("s1", 5.0, 6.0),))
    intervals = _audio_silence_intervals_from_raw_understanding_map(raw_map)
    assert intervals == ((5.0, 6.0),)


def test_02_no_silence_intervals_behaves_as_before():
    words, _ = _words("hello there friend", 0.0)
    raw_map = _raw_map("s1", words)  # audio_events defaults to ()
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert "AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED" in evidence.missing_evidence
    assert evidence.audio_silence_interval_count == 0


def test_03_source_with_no_audio_fails_open():
    words, _ = _words("hello there friend", 0.0)
    raw_map = _raw_map("s1", words, audio_events=())  # no audio track / ffmpeg failure -> empty
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert evidence.capability_status == CAPABILITY_AVAILABLE
    assert "AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED" in evidence.missing_evidence


def test_04_multi_source_isolation_source_a_never_affects_source_b():
    words_a, _ = _words("first source words here", 0.0)
    raw_map_a = _raw_map("srcA", words_a, audio_events=(_silence_event("srcA", 2.0, 3.0),))
    words_b, _ = _words("second source words here", 0.0)
    # srcB's own RawUnderstandingMap carries NO audio events at all.
    raw_map_b = _raw_map("srcB", words_b, audio_events=())

    intervals_a = _audio_silence_intervals_from_raw_understanding_map(raw_map_a)
    intervals_b = _audio_silence_intervals_from_raw_understanding_map(raw_map_b)
    assert intervals_a == ((2.0, 3.0),)
    assert intervals_b == ()

    # Defensive cross-contamination check: even if a caller mistakenly
    # attached srcA's event onto a map claiming to be srcB, the
    # source_asset_id re-filter in _audio_silence_intervals_from_raw_
    # understanding_map must reject it.
    poisoned = _raw_map("srcB", words_b, audio_events=(_silence_event("srcA", 2.0, 3.0),))
    assert _audio_silence_intervals_from_raw_understanding_map(poisoned) == ()


def test_05_silence_creates_pause_boundary_where_rules_already_allow_it():
    # Two chunks separated by a real measured silence interval that
    # overlaps the word gap -- segment_language_phrases's own EXISTING
    # rule (never changed here) already promotes this to PAUSE.
    words1, t1 = _words("first chunk of words", 0.0)
    words2, _ = _words("second chunk of words", t1 + 1.0)
    all_words = words1 + words2
    lwords = adapt_words_to_language_words("s1", all_words)
    phrases = segment_language_phrases(lwords, audio_silence_intervals=((t1, t1 + 1.0),))
    assert any(p.boundary_kind == BOUNDARY_PAUSE for p in phrases)


def test_06_punctuation_only_boundary_remains_when_silence_absent():
    words1, t1 = _words("this sentence ends.", 0.0)
    words2, _ = _words("a new one starts", t1 + 0.10)
    all_words = words1 + words2
    lwords = adapt_words_to_language_words("s1", all_words)
    phrases_before = segment_language_phrases(lwords)
    phrases_after = segment_language_phrases(lwords, audio_silence_intervals=())
    assert [p.boundary_kind for p in phrases_before] == [p.boundary_kind for p in phrases_after]


def test_07_silence_and_punctuation_coexist():
    words1, t1 = _words("this sentence ends.", 0.0)
    words2, t2 = _words("a real pause follows", t1 + 1.0)
    words3, _ = _words("and more words after", t2 + 0.10)
    all_words = words1 + words2 + words3
    lwords = adapt_words_to_language_words("s1", all_words)
    phrases = segment_language_phrases(lwords, audio_silence_intervals=((t1, t1 + 1.0),))
    kinds = [p.boundary_kind for p in phrases]
    assert BOUNDARY_PAUSE in kinds
    # The pre-existing terminal punctuation on the first sentence is
    # PRESERVED in its own phrase's text even though the real silence
    # interval (a stronger, more specific split reason) is what decided
    # this particular boundary's own kind -- coexistence, not loss.
    assert phrases[0].text_raw.endswith(".")
    assert len(phrases) >= 2


def test_08_multiple_silence_intervals_all_consumed():
    words1, t1 = _words("chunk one here", 0.0)
    words2, t2 = _words("chunk two here", t1 + 1.0)
    words3, _ = _words("chunk three here", t2 + 1.0)
    all_words = words1 + words2 + words3
    lwords = adapt_words_to_language_words("s1", all_words)
    intervals = ((t1, t1 + 1.0), (t2, t2 + 1.0))
    phrases = segment_language_phrases(lwords, audio_silence_intervals=intervals)
    pause_count = sum(1 for p in phrases if p.boundary_kind == BOUNDARY_PAUSE)
    assert pause_count == 2


def test_09_irrelevant_silence_does_not_fabricate_meaning():
    words, _ = _words("this cream removes wrinkles fast", 0.0)
    raw_map_no_silence = _raw_map("s1", words)
    raw_map_with_far_silence = _raw_map(
        "s1", words, audio_events=(_silence_event("s1", 500.0, 501.0),),  # far outside any real word gap
    )
    ev1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_no_silence)
    ev2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_with_far_silence)
    assert [a.text_normalized for a in ev1.attempts] == [a.text_normalized for a in ev2.attempts]
    assert [p.claim_signature.claim_type for p in ev1.proposition_candidates] == \
        [p.claim_signature.claim_type for p in ev2.proposition_candidates]


def test_10_source_relative_timing_preserved_exactly():
    words, t1 = _words("first chunk of words", 10.0)
    words2, _ = _words("second chunk of words", t1 + 1.0)
    all_words = words + words2
    lwords = adapt_words_to_language_words("s1", all_words)
    phrases = segment_language_phrases(lwords, audio_silence_intervals=((t1, t1 + 1.0),))
    # No offset/rebasing anywhere -- phrase source_start/source_end stay
    # in the SAME source-relative timing domain the interval was given in.
    assert phrases[0].source_start == 10.0
    assert all(p.source_start >= 10.0 for p in phrases)


# ===========================================================================
# 11-14: determinism / identity.
# ===========================================================================
def test_11_phrase_ids_stable_when_membership_unchanged():
    words, _ = _words("hello there friend", 0.0)
    raw_map_no_silence = _raw_map("s1", words)
    raw_map_far_silence = _raw_map("s1", words, audio_events=(_silence_event("s1", 500.0, 501.0),))
    ev1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_no_silence)
    ev2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_far_silence)
    assert [p.phrase_id for p in ev1.phrases] == [p.phrase_id for p in ev2.phrases]


def test_12_utterance_determinism():
    words = _d235z_collapse_shape_words()
    raw_map = _raw_map("s1", words, audio_events=(_silence_event("s1", 3.0, 4.0), _silence_event("s1", 7.0, 8.0)))
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    e2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert tuple(u.utterance_id for u in e1.utterances) == tuple(u.utterance_id for u in e2.utterances)


def test_13_attempt_determinism():
    words = _d235z_collapse_shape_words()
    raw_map = _raw_map("s1", words, audio_events=(_silence_event("s1", 3.0, 4.0),))
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    e2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert tuple(a.attempt_id for a in e1.attempts) == tuple(a.attempt_id for a in e2.attempts)


def test_14_proposition_candidate_determinism():
    words = _d235z_collapse_shape_words()
    raw_map = _raw_map("s1", words, audio_events=(_silence_event("s1", 3.0, 4.0),))
    e1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    e2 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert tuple(p.proposition_candidate_id for p in e1.proposition_candidates) == \
        tuple(p.proposition_candidate_id for p in e2.proposition_candidates)


# ===========================================================================
# 15-18: D-235Z collapse shape before/after, healthy control, no forced count.
# ===========================================================================
def test_15_d235z_collapse_shape_reproduces_before_fix():
    """WITHOUT audio-silence evidence, the generic D-235Z-shaped fixture
    (12 small chunks joined by sub-threshold timing gaps, no terminal
    punctuation) collapses to exactly ONE utterance -- reproducing the
    real run's own observed shape (12 phrases -> 1 utterance) on
    synthetic data, proving this is the SAME structural mechanism, not
    a real-transcript-specific fluke."""
    words = _d235z_collapse_shape_words()
    raw_map = _raw_map("s1", words, audio_events=())
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.phrases) >= 5  # multiple real phrases exist
    assert len(evidence.utterances) == 1  # BEFORE: collapsed to one
    assert len(evidence.attempts) == 1
    assert len(evidence.proposition_candidates) == 1


def test_16_d235z_collapse_shape_resolves_after_fix():
    """WITH real audio-silence intervals placed at several of the
    fixture's own inter-chunk gaps, the SAME fixture now yields MORE
    THAN ONE utterance -- no target count asserted (this task's own
    explicit instruction), only that supplying real evidence changes
    the outcome versus test_15's own baseline."""
    words = _d235z_collapse_shape_words()
    # Place real silence over several (not all) of the fixture's own
    # 0.30s inter-chunk gaps -- exactly the kind of real, measured pause
    # evidence audio_silence.py already produces upstream.
    lwords = adapt_words_to_language_words("s1", words)
    silence_intervals = tuple(
        (lwords[i].end, lwords[i + 1].start)
        for i in range(len(lwords) - 1)
        if (lwords[i + 1].start - lwords[i].end) > 0.20
    )[:4]
    assert silence_intervals, "fixture must contain at least one real inter-chunk gap to silence-confirm"
    raw_map = _raw_map("s1", words, audio_events=tuple(
        _silence_event("s1", start, end) for start, end in silence_intervals
    ))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert evidence.audio_silence_interval_count == len(silence_intervals)
    assert len(evidence.utterances) > 1  # AFTER: no longer a single whole-source utterance
    assert len(evidence.attempts) > 1
    assert len(evidence.proposition_candidates) > 1


def test_17_video00_style_healthy_control_not_regressed():
    """A fixture where punctuation ALONE already creates healthy, multi-
    utterance segmentation (mirrors Video00's own real shape, per
    D-235Z's comparison) must NOT collapse, merge, or lose determinism
    when real (but non-conflicting) silence evidence is additionally
    supplied."""
    parts = [
        "this cream removes wrinkles fast.",
        "it works for every skin type.",
        "doctors recommend it daily.",
        "results appear within two weeks.",
    ]
    words = []
    t = 0.0
    for part in parts:
        chunk, t_after = _words(part, t)
        words.extend(chunk)
        t = t_after + 0.10  # small gap only -- punctuation itself drives the split
    words = tuple(words)

    raw_map_no_silence = _raw_map("s1", words, audio_events=())
    ev_before = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_no_silence)
    assert len(ev_before.utterances) == len(parts)  # punctuation alone already separates all four

    # A real silence interval at a location that does not straddle any
    # existing boundary must not merge/collapse anything.
    raw_map_with_silence = _raw_map(
        "s1", words, audio_events=(_silence_event("s1", words[-1].end + 10.0, words[-1].end + 11.0),),
    )
    ev_after = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map_with_silence)
    assert len(ev_after.utterances) == len(ev_before.utterances)
    assert [u.text_normalized for u in ev_after.utterances] == [u.text_normalized for u in ev_before.utterances]


def test_18_no_target_count_asserted_only_correct_consumption():
    """Documents this task's own 'do not assert target counts' rule:
    the fix is judged by whether real evidence is CONSUMED (pause
    boundary count moves from 0 to >0 when real intervals are supplied
    over a real gap), never by a specific hardcoded phrase/utterance/
    attempt count."""
    words1, t1 = _words("a chunk of words", 0.0)
    words2, _ = _words("another chunk follows", t1 + 1.0)
    all_words = words1 + words2
    lwords = adapt_words_to_language_words("s1", all_words)
    without = segment_language_phrases(lwords)
    with_silence = segment_language_phrases(lwords, audio_silence_intervals=((t1, t1 + 1.0),))
    pause_before = sum(1 for p in without if p.boundary_kind == BOUNDARY_PAUSE)
    pause_after = sum(1 for p in with_silence if p.boundary_kind == BOUNDARY_PAUSE)
    assert pause_before == 0
    assert pause_after > pause_before


# ===========================================================================
# 19-22: D-235X exact-identity compatibility (unmodified, proven compatible).
# ===========================================================================
def _candidate_take(clip_id: str, words, source_asset_id: str = "s1", attempt_id: str | None = None):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words),
        words=tuple(words), attempt_id=attempt_id,
    )


def test_19_exact_word_identity_one_to_one_after_fix():
    words1, t1 = _words("first real chunk of many spoken words", 0.0)
    words2, _ = _words("second real chunk of many spoken words", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.attempts) >= 2  # the silence-confirmed split produced >=2 real attempts

    utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
    take = _candidate_take("clipA", words1)
    matches = build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=(take,), canonical_words=evidence.words,
        language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
    )
    assert matches[0].relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES


def test_20_exact_word_identity_one_to_many_partition():
    words1, t1 = _words("first real chunk of many spoken words", 0.0)
    words2, _ = _words("second real chunk of many spoken words", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    utterances_by_id = {u.utterance_id: u for u in evidence.utterances}

    # A single reconstructed attempt spanning BOTH real, now-smaller
    # LanguageAttempts should exact-partition, not silently fail.
    combined_take = _candidate_take("clipCombined", all_words)
    matches = build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=(combined_take,), canonical_words=evidence.words,
        language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
    )
    assert matches[0].relationship_status == RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION


def test_21_heuristic_never_overrides_an_available_exact_match():
    # shared_attempt_word_identity.py itself is untouched by D-236 (see
    # test_35 below) -- its own AUTHORITATIVE_RELATIONSHIP_STATUSES
    # precedence (exact-match-first) is unchanged, proven by D-235P/X's
    # own existing regression suite. This test proves the SAME contract
    # still holds end-to-end once real silence-driven attempts exist.
    words1, t1 = _words("first real chunk of many spoken words", 0.0)
    words2, _ = _words("second real chunk of many spoken words", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
    take = _candidate_take("clipA", words1)
    match = build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=(take,), canonical_words=evidence.words,
        language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
    )[0]
    assert match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    assert match.exact_shared_word_count == match.reconstructed_word_count


def test_22_same_text_different_positions_remain_distinct():
    words1, t1 = _words("please try again", 0.0)
    words2, _ = _words("please try again", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.attempts) >= 2
    ids = [a.attempt_id for a in evidence.attempts]
    assert len(ids) == len(set(ids))  # distinct ids despite identical text


# ===========================================================================
# 23-25: bilingual / code-switching safety.
# ===========================================================================
def test_23_spanish_silence_wiring():
    words1, t1 = _words("esta crema elimina las arrugas rapido", 0.0)
    words2, _ = _words("funciona para todo tipo de piel", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.attempts) >= 2


def test_24_english_silence_wiring():
    words1, t1 = _words("this cream removes wrinkles very fast", 0.0)
    words2, _ = _words("it works for every skin type", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.attempts) >= 2


def test_25_spanglish_silence_wiring():
    words1, t1 = _words("this crema removes arrugas very fast", 0.0)
    words2, _ = _words("funciona para every skin type today", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    assert len(evidence.attempts) >= 2


# ===========================================================================
# 26-31: no new detector / threshold / ASR-rerun / ffmpeg-rerun / provider / RAW.
# ===========================================================================
def _import_lines(module) -> str:
    """Only this module's own top-level `import `/`from ` statement lines
    -- deliberately excludes docstrings/comments (which legitimately
    discuss banned concepts BY NAME, e.g. explaining a firewall this
    module does NOT cross, per the exact same precedent D-199's own
    module docstring already established). The real, structural proof
    that a module never touches another authority is that it never
    IMPORTS from it -- mirrors D-235P's own established "no live module
    imports this gate" test style."""
    lines = inspect.getsource(module).splitlines()
    return "\n".join(line for line in lines if line.strip().startswith(("import ", "from ")))


def test_26_no_new_threshold_constant_added():
    source = inspect.getsource(spine_live_module)
    # No new numeric-tolerance constant introduced by this task -- the
    # only floats this module's own new code touches are float(event.start)/
    # float(event.end) casts, never a new comparison threshold.
    assert "DEFAULT_SPLIT_GAP_SEC" not in source
    assert "_BOUNDARY_MATCH_TOLERANCE_SEC" not in source


def test_27_no_new_detector_function_defined():
    functions = {name for name, _ in inspect.getmembers(spine_live_module, inspect.isfunction)}
    # The only new callable this task adds is the extraction helper --
    # never a new silence/pause DETECTOR (that stays exclusively in
    # audio_silence.py, untouched).
    assert "_audio_silence_intervals_from_raw_understanding_map" in functions
    assert not any("detect" in name.lower() for name in functions)


def test_28_no_asr_reinvocation():
    imports = _import_lines(spine_live_module)
    assert "import asr" not in imports
    assert "from .asr" not in imports
    # No actual call to a transcription entry point anywhere in the
    # module (docstring PROSE explaining the "no ASR rerun" contract by
    # name is fine and expected -- only a real call would be a violation).
    assert ".transcribe(" not in inspect.getsource(spine_live_module)


def test_29_no_ffmpeg_silence_rerun():
    imports = _import_lines(spine_live_module)
    assert "import subprocess" not in imports
    assert "from .audio_silence import detect_audio_silence_intervals" not in imports
    assert "from .audio_silence import audio_silence_events" not in imports
    # Only the event-KIND constant is imported (D-236's own additive
    # import) -- never the detector functions themselves.
    assert "AUDIO_SILENCE_EVENT_KIND" in imports
    source = inspect.getsource(spine_live_module)
    assert "detect_audio_silence_intervals(" not in source
    assert "audio_silence_events(" not in source


def test_30_no_provider_call():
    imports = _import_lines(spine_live_module)
    for banned in ("openai", "gemini", "anthropic", "requests", "httpx"):
        assert banned not in imports.lower()


def test_31_no_workflow_or_raw_dispatch_file_touched():
    # This task is offline-only -- proven at the deliverable level (no
    # workflow_dispatch / Modal / RunPod call anywhere in this module).
    # Checked against IMPORT lines only, never bare substring-in-prose --
    # "modal" as a bare substring also matches "canonical" in the
    # module's own pre-existing docstring, so a whole-source scan would
    # false-positive.
    imports = _import_lines(spine_live_module)
    for banned in ("modal", "runpod"):
        assert banned not in imports.lower()
    assert "workflow_dispatch" not in inspect.getsource(spine_live_module)


# ===========================================================================
# 32-36: no P1/P2/BestTake/Family/Ordering/Boundary/Freeze/materiality/
# repair/Pacing/Audio-Join mutation -- this module's own authority
# firewall (D-199's own module docstring) stays intact.
# ===========================================================================
def test_32_no_p1_p2_authority_mutation():
    # This module's own docstring legitimately DISCUSSES editorial_moment_
    # sequence_integration.build_editorial_moments_for_source BY NAME (it
    # explains the pre-existing bridge to P1's architecture) -- the real
    # firewall proof is that this module never IMPORTS from it.
    imports = _import_lines(spine_live_module)
    assert "from .editorial_moment_sequence_integration" not in imports
    assert "from .whole_video_openai" not in imports


def test_33_no_besttake_family_mutation():
    imports = _import_lines(spine_live_module)
    for banned in ("deterministic_best_take_authority", "take_grouping", "semantic_idea_equivalence"):
        assert banned not in imports


def test_34_no_ordering_boundary_mutation():
    imports = _import_lines(spine_live_module)
    for banned in ("ordering_engine", "boundary_engine_pass"):
        assert banned not in imports


def test_35_no_freeze_materiality_repair_mutation():
    imports = _import_lines(spine_live_module)
    for banned in (
        "lost_semantic_atom_freeze_authority", "complete_lost_semantic_atom_materiality",
        "lost_atom_repair_suppression", "repair_loop", "final_story_coherence_validation",
        "shared_attempt_word_identity",
    ):
        assert banned not in imports


def test_36_no_pacing_audio_join_mutation():
    imports = _import_lines(spine_live_module)
    for banned in ("pacing_v2", "dialogue_pacing_transition", "audio_join_treatment", "prosodic_audio_v2"):
        assert banned not in imports


# ===========================================================================
# Diagnostics surface (compact, counts-only, no transcript dump).
# ===========================================================================
def test_37_diagnostics_expose_required_compact_fields():
    words1, t1 = _words("first real chunk", 0.0)
    words2, _ = _words("second real chunk", t1 + 1.0)
    all_words = words1 + words2
    raw_map = _raw_map("s1", all_words, audio_events=(_silence_event("s1", t1, t1 + 1.0),))
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    diag = live_language_spine_diagnostics(evidence)
    for key in (
        "language_spine_audio_silence_evidence_status", "audio_silence_interval_count",
        "restart_marker_evidence_status", "phrase_count", "utterance_count", "attempt_count",
        "proposition_candidate_count", "pause_boundary_count", "restart_boundary_count",
    ):
        assert key in diag
    assert diag["language_spine_audio_silence_evidence_status"] == "SUPPLIED"
    assert diag["audio_silence_interval_count"] == 1
    assert diag["restart_marker_evidence_status"] == "NOT_AVAILABLE"
    assert diag["pause_boundary_count"] >= 1
    # No transcript dump -- no raw text field leaks into the diagnostic dict.
    assert "text_raw" not in diag
    assert "transcript" not in diag


def test_38_diagnostics_report_not_supplied_when_empty():
    words, _ = _words("hello there friend", 0.0)
    raw_map = _raw_map("s1", words)
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map)
    diag = live_language_spine_diagnostics(evidence)
    assert diag["language_spine_audio_silence_evidence_status"] == "NOT_SUPPLIED"
    assert diag["audio_silence_interval_count"] == 0


def test_39_run_summary_aggregates_interval_count():
    words1, t1 = _words("first real chunk", 0.0)
    raw_map1 = _raw_map("s1", words1, audio_events=(_silence_event("s1", 5.0, 6.0),))
    words2, t2 = _words("second real chunk", 0.0)
    raw_map2 = _raw_map("s2", words2, audio_events=(_silence_event("s2", 1.0, 2.0), _silence_event("s2", 3.0, 4.0)))
    ev1 = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=raw_map1)
    ev2 = build_live_language_spine_for_source(source_asset_id="s2", raw_understanding_map=raw_map2)
    summary = live_language_spine_run_summary((ev1, ev2))
    assert summary["audio_silence_interval_count"] == 3


# ===========================================================================
# Backward compatibility -- existing direct construction sites (pre-D-236
# test fixtures across the codebase) stay valid with the new additive,
# defaulted field.
# ===========================================================================
def test_40_backward_compatible_direct_construction_without_new_field():
    evidence = LiveLanguageSpineEvidence(
        source_asset_id="s1", words=(), phrases=(), utterances=(), attempts=(),
        proposition_candidates=(), relation_evidence=(), capability_status="NOT_EVALUABLE",
        missing_evidence=(), conflicts=(), provenance=(),
    )
    assert evidence.audio_silence_interval_count == 0


def test_41_word_timings_absent_branch_unaffected():
    evidence = build_live_language_spine_for_source(source_asset_id="s1", raw_understanding_map=None)
    assert evidence.audio_silence_interval_count == 0
    assert evidence.capability_status != CAPABILITY_AVAILABLE
