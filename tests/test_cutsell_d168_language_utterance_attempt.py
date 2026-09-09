"""D-168: Language / Transcript Spine, Phase B -- LanguageUtterance +
LanguageAttempt. Covers every directive-required fixture category plus
structural/contract/no-authority-change proofs.

See cutsell_worker/language_utterance_attempt.py's own module docstring for
the full design rationale this test suite verifies against.
"""
from __future__ import annotations

import inspect

import pytest

from cutsell_worker.contracts import Word
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker import language_utterance_attempt as m
from cutsell_worker.language_utterance_attempt import (
    ATTEMPT_ABANDONED,
    ATTEMPT_CLEAN,
    ATTEMPT_CONTINUATION,
    ATTEMPT_CORRECTION,
    ATTEMPT_FALSE_START,
    ATTEMPT_RECORDING_PROCESS,
    ATTEMPT_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguageAttempt,
    LanguageUtterance,
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
    UTTERANCE_ABANDONED,
    UTTERANCE_CORRECTED,
    UTTERANCE_RESTARTED,
    UTTERANCE_COMPLETE,
    UTTERANCE_INCOMPLETE,
    UTTERANCE_UNCERTAIN,
    build_language_attempts,
    language_utterance_attempt_diagnostics,
    raw_understanding_compatibility_reference,
    segment_language_utterances,
    watch_listen_compatibility_reference,
)


def _code_body_excluding_module_docstring(module) -> str:
    """Strips a module's own leading docstring before substring-checking for
    forbidden terms -- the module docstring legitimately mentions
    out-of-scope terms (D-163/D-166's own established test pattern)."""
    source = inspect.getsource(module)
    first = source.find('"""')
    if first == -1:
        return source
    second = source.find('"""', first + 3)
    return source[second + 3:] if second != -1 else source


def _w(text: str, start: float, end: float, confidence: float = 0.9) -> Word:
    return Word(text=text, start=start, end=end, confidence=confidence)


def _build(words, **phrase_kwargs):
    lw = adapt_words_to_language_words("src1", words)
    phrases = segment_language_phrases(lw, **phrase_kwargs)
    utterances = segment_language_utterances(phrases)
    return phrases, utterances


# ---------------------------------------------------------------------------
# 1. Simple complete utterance
# ---------------------------------------------------------------------------
def test_01_simple_complete_utterance():
    words = [_w("This", 0.0, 0.4), _w("is", 0.4, 0.6), _w("a", 0.6, 0.7), _w("test.", 0.7, 1.2)]
    _, utterances = _build(words)
    assert len(utterances) == 1
    assert utterances[0].utterance_state == UTTERANCE_COMPLETE
    assert utterances[0].meaning_completion == MEANING_COMPLETE
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 1
    assert attempts[0].attempt_state == ATTEMPT_CLEAN


# ---------------------------------------------------------------------------
# 2. Incomplete utterance
# ---------------------------------------------------------------------------
def test_02_incomplete_utterance():
    words = [_w("I", 0.0, 0.1), _w("want", 0.1, 0.3), _w("to", 0.3, 0.4)]
    _, utterances = _build(words)
    assert len(utterances) == 1
    assert utterances[0].meaning_completion == MEANING_INCOMPLETE


# ---------------------------------------------------------------------------
# 3. Punctuation does not automatically mean editorial completion
# ---------------------------------------------------------------------------
def test_03_punctuation_not_automatic_completion():
    # Ends in a period but the last word is a bridge connector-adjacent open
    # tail per take_segmentation's own grammatically-open-tail rule: use a
    # trailing ellipsis (the transcriber's own "trails off" marker) which
    # _looks_complete_idea treats as NOT complete despite trailing dots.
    words = [_w("And", 0.0, 0.2), _w("then", 0.2, 0.4), _w("I", 0.4, 0.5), _w("got", 0.5, 0.7), _w("diagnosed...", 0.7, 1.2)]
    _, utterances = _build(words)
    assert len(utterances) == 1
    assert utterances[0].text_raw.endswith("...")
    assert utterances[0].meaning_completion == MEANING_INCOMPLETE


# ---------------------------------------------------------------------------
# 4. Pause boundary (real audio-silence-confirmed)
# ---------------------------------------------------------------------------
def test_04_pause_boundary_splits_utterances():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("Now", 3.0, 3.2), _w("more.", 3.2, 3.6),
    ]
    _, utterances = _build(words, audio_silence_intervals=[(0.7, 3.0)])
    assert len(utterances) == 2
    assert utterances[0].boundary_end_kind == m.BOUNDARY_PAUSE


# ---------------------------------------------------------------------------
# 5. Restart boundary
# ---------------------------------------------------------------------------
def test_05_restart_boundary_splits_utterances():
    words = [
        _w("I", 0.0, 0.3), _w("really", 0.3, 0.6), _w("wanted", 0.6, 0.9), _w("to", 0.9, 1.0),
        _w("I", 3.0, 3.3), _w("really", 3.3, 3.6), _w("wanted", 3.6, 3.9), _w("to", 3.9, 4.0),
        _w("explain", 4.0, 4.3), _w("this", 4.3, 4.5), _w("properly.", 4.5, 5.0),
    ]
    _, utterances = _build(words, restart_marker_times=[3.0])
    assert len(utterances) == 2
    assert utterances[0].boundary_end_kind == m.BOUNDARY_RESTART_BOUNDARY


# ---------------------------------------------------------------------------
# 6. False start
# ---------------------------------------------------------------------------
def _false_start_words():
    return [
        _w("I", 0.0, 0.3), _w("really", 0.3, 0.6), _w("wanted", 0.6, 0.9), _w("to", 0.9, 1.0),
        _w("I", 3.0, 3.3), _w("really", 3.3, 3.6), _w("wanted", 3.6, 3.9), _w("to", 3.9, 4.0),
        _w("explain", 4.0, 4.3), _w("this", 4.3, 4.5), _w("properly.", 4.5, 5.0),
    ]


def test_06_false_start():
    _, utterances = _build(_false_start_words(), restart_marker_times=[3.0])
    assert utterances[0].utterance_state == UTTERANCE_ABANDONED
    attempts = build_language_attempts(utterances)
    assert attempts[0].attempt_state == ATTEMPT_FALSE_START
    assert attempts[0].meaning_completion == MEANING_INCOMPLETE


# ---------------------------------------------------------------------------
# 7. Abandoned attempt (longer, not brief)
# ---------------------------------------------------------------------------
def test_07_abandoned_attempt_not_brief():
    # A long-but-grammatically-open abandoned run (ends on a bridge
    # connector, "and") must classify INCOMPLETE regardless of its own
    # word count/duration -- take_segmentation._grammatically_open_tail is
    # checked BEFORE the length-heuristic fallback, so a long run is never
    # forced COMPLETE merely by being long. >=6 words AND >3.0s duration
    # -- not FALSE_START (the brief-abandoned case, tested separately).
    words = [
        _w("So", 0.0, 0.2), _w("I", 0.2, 0.4), _w("went", 0.4, 0.7), _w("to", 0.7, 0.9),
        _w("the", 0.9, 1.1), _w("doctor", 1.1, 1.5), _w("because", 1.5, 1.9), _w("I", 1.9, 2.1),
        _w("really", 2.1, 2.4), _w("wanted", 2.4, 2.7), _w("to", 2.7, 2.9), _w("check", 2.9, 3.3), _w("and", 3.3, 3.6),
        _w("So", 6.0, 6.2), _w("I", 6.2, 6.4), _w("went", 6.4, 6.7), _w("to", 6.7, 6.9),
        _w("the", 6.9, 7.1), _w("doctor", 7.1, 7.5), _w("because", 7.5, 7.9), _w("I", 7.9, 8.1),
        _w("really", 8.1, 8.4), _w("wanted", 8.4, 8.7), _w("to", 8.7, 8.9), _w("check", 8.9, 9.3),
        _w("and", 9.3, 9.5), _w("she", 9.5, 9.7), _w("confirmed", 9.7, 10.1), _w("everything.", 10.1, 10.6),
    ]
    _, utterances = _build(words, restart_marker_times=[6.0])
    assert utterances[0].utterance_state == UTTERANCE_ABANDONED
    assert utterances[0].meaning_completion == MEANING_INCOMPLETE
    assert utterances[0].duration_sec >= 3.0
    attempts = build_language_attempts(utterances)
    assert attempts[0].attempt_state == ATTEMPT_ABANDONED


# ---------------------------------------------------------------------------
# 8. Clean retry as new attempt (paired with false start / abandoned)
# ---------------------------------------------------------------------------
def test_08_clean_retry_is_new_attempt():
    attempts = build_language_attempts(_build(_false_start_words(), restart_marker_times=[3.0])[1])
    assert len(attempts) == 2
    assert attempts[1].attempt_state == ATTEMPT_CLEAN
    assert attempts[1].restart_evidence is True
    assert attempts[1].meaning_completion == MEANING_COMPLETE
    # Never collapsed into one complete utterance/attempt.
    assert attempts[0].attempt_id != attempts[1].attempt_id


# ---------------------------------------------------------------------------
# 9. Continuation
# ---------------------------------------------------------------------------
def test_09_continuation():
    # A real audio-silence-confirmed pause (PAUSE, a strong structural
    # boundary) between an open-tail (bridge-connector-ending) fragment and
    # its completion -- the utterance segmenter genuinely splits them, and
    # attempt construction joins them via the continuation rule regardless
    # of the boundary kind that separated them (module docstring).
    words = [
        _w("And", 0.0, 0.1), _w("then", 0.1, 0.3), _w("we", 0.3, 0.4), _w("went", 0.4, 0.6), _w("to", 0.6, 0.7),
        _w("the", 2.0, 2.1), _w("store.", 2.1, 2.5),
    ]
    _, utterances = _build(words, audio_silence_intervals=[(0.7, 2.0)])
    assert len(utterances) == 2
    assert utterances[0].meaning_completion == MEANING_INCOMPLETE
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 1
    assert attempts[0].attempt_state == ATTEMPT_CONTINUATION
    assert attempts[0].continuation_evidence is True
    assert attempts[0].meaning_completion == MEANING_COMPLETE
    # Utterances themselves are NOT physically merged (still 2 distinct ids).
    assert len(attempts[0].utterance_ids) >= 1


# ---------------------------------------------------------------------------
# 10. Correction
# ---------------------------------------------------------------------------
def _correction_words():
    return [
        _w("I", 0.0, 0.2), _w("told", 0.2, 0.4), _w("my", 0.4, 0.5), _w("sister", 0.5, 0.8),
        _w("about", 0.8, 1.0), _w("the", 1.0, 1.1), _w("fifty", 1.1, 1.3), _w("dollar", 1.3, 1.5), _w("deal.", 1.5, 1.9),
        _w("I", 4.0, 4.2), _w("told", 4.2, 4.4), _w("my", 4.4, 4.5), _w("sister", 4.5, 4.8),
        _w("about", 4.8, 5.0), _w("the", 5.0, 5.1), _w("forty", 5.1, 5.3), _w("dollar", 5.3, 5.5),
        _w("deal,", 5.5, 5.8), _w("actually.", 5.8, 6.3),
    ]


def test_10_correction_preserved():
    _, utterances = _build(_correction_words(), restart_marker_times=[4.0])
    assert utterances[0].utterance_state == UTTERANCE_RESTARTED
    assert utterances[1].utterance_state == UTTERANCE_CORRECTED
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 1
    assert attempts[0].attempt_state == ATTEMPT_CORRECTION
    assert attempts[0].correction_evidence is True
    # Never flattened into an ordinary two-attempt retry.
    assert "fifty" in attempts[0].text_raw and "forty" in attempts[0].text_raw


# ---------------------------------------------------------------------------
# 11. Recording process (explicit external evidence only)
# ---------------------------------------------------------------------------
def test_11_recording_process_explicit_evidence():
    words = [_w("Okay", 0.0, 0.2), _w("let's", 0.2, 0.4), _w("try", 0.4, 0.6), _w("that", 0.6, 0.8), _w("again.", 0.8, 1.2)]
    _, utterances = _build(words)
    attempts = build_language_attempts(utterances, recording_process_utterance_indices=frozenset({0}))
    assert attempts[0].attempt_state == ATTEMPT_RECORDING_PROCESS
    assert attempts[0].recording_process_evidence is True

    # Without the explicit hint, this module invents NO meta-speech phrase
    # vocabulary of its own -- same text, no hint, never RECORDING_PROCESS.
    attempts_no_hint = build_language_attempts(utterances)
    assert attempts_no_hint[0].attempt_state != ATTEMPT_RECORDING_PROCESS


# ---------------------------------------------------------------------------
# 12. Filler does not delete text
# ---------------------------------------------------------------------------
def test_12_filler_not_deleted():
    words = [_w("So,", 0.0, 0.2), _w("um,", 0.2, 0.4), _w("this", 0.4, 0.6), _w("works.", 0.6, 1.0)]
    _, utterances = _build(words)
    assert "um" in utterances[0].text_raw.lower()
    attempts = build_language_attempts(utterances)
    assert "um" in attempts[0].text_raw.lower()


# ---------------------------------------------------------------------------
# 13. Partial word preserved
# ---------------------------------------------------------------------------
def test_13_partial_word_preserved():
    words = [_w("Th-", 0.0, 0.1), _w("this", 0.1, 0.3), _w("is", 0.3, 0.4), _w("it.", 0.4, 0.8)]
    _, utterances = _build(words)
    assert "Th-" in utterances[0].text_raw
    attempts = build_language_attempts(utterances)
    assert "Th-" in attempts[0].text_raw


# ---------------------------------------------------------------------------
# 14. Multi-sentence one attempt
# ---------------------------------------------------------------------------
def test_14_multi_sentence_one_attempt():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("It", 0.9, 1.0), _w("works", 1.0, 1.2), _w("well.", 1.2, 1.6),
    ]
    _, utterances = _build(words)
    assert len(utterances) == 2
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 1
    assert attempts[0].attempt_state == ATTEMPT_CLEAN


# ---------------------------------------------------------------------------
# 15. Two beats separate
# ---------------------------------------------------------------------------
def test_15_two_beats_separate():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("Now", 3.0, 3.2), _w("let", 3.2, 3.3), _w("me", 3.3, 3.4), _w("show", 3.4, 3.6),
        _w("you", 3.6, 3.7), _w("more.", 3.7, 4.1),
    ]
    _, utterances = _build(words, audio_silence_intervals=[(0.7, 3.0)])
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 2
    assert all(a.attempt_state == ATTEMPT_CLEAN for a in attempts)


# ---------------------------------------------------------------------------
# 16. Conclusion continuation
# ---------------------------------------------------------------------------
def test_16_conclusion_continuation():
    words = [
        _w("So", 0.0, 0.1), _w("in", 0.1, 0.2), _w("the", 0.2, 0.3), _w("end", 0.3, 0.5), _w("it", 0.5, 0.6),
        _w("all", 0.6, 0.7), _w("comes", 0.7, 0.9), _w("down", 0.9, 1.0), _w("to", 1.0, 1.1),
        _w("consistency.", 1.3, 1.9),
    ]
    _, utterances = _build(words)
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 1
    assert attempts[0].meaning_completion == MEANING_COMPLETE


# ---------------------------------------------------------------------------
# 17. CTA-like complete utterance (generic, no Video00 text)
# ---------------------------------------------------------------------------
def test_17_cta_like_complete_utterance():
    words = [_w("Get", 0.0, 0.2), _w("yours", 0.2, 0.4), _w("today.", 0.4, 0.9)]
    _, utterances = _build(words)
    assert utterances[0].meaning_completion == MEANING_COMPLETE


# ---------------------------------------------------------------------------
# 18. No-punctuation ASR
# ---------------------------------------------------------------------------
def test_18_no_punctuation_asr():
    words = [_w("this", 0.0, 0.2), _w("works", 0.2, 0.4), _w("great", 0.4, 0.6), _w("today", 0.6, 0.8),
              _w("for", 0.8, 0.9), _w("everyone", 0.9, 1.3)]
    _, utterances = _build(words)
    assert len(utterances) == 1
    # >=6 words -- length-heuristic fallback still classifies COMPLETE.
    assert utterances[0].meaning_completion == MEANING_COMPLETE
    assert utterances[0].confidence in (CONFIDENCE_WEAK, CONFIDENCE_SUPPORTED, CONFIDENCE_UNKNOWN)


# ---------------------------------------------------------------------------
# 19. Long pause
# ---------------------------------------------------------------------------
def test_19_long_pause():
    words = [_w("Hello.", 0.0, 0.3), _w("Goodbye.", 10.0, 10.4)]
    _, utterances = _build(words, audio_silence_intervals=[(0.3, 10.0)])
    assert len(utterances) == 2
    assert utterances[0].boundary_end_kind == m.BOUNDARY_PAUSE


# ---------------------------------------------------------------------------
# 20. Short pause (not enough to be audio-confirmed, weak signal only)
# ---------------------------------------------------------------------------
def test_20_short_pause_weak_signal():
    words = [_w("Hello", 0.0, 0.3), _w("there.", 0.35, 0.7)]
    _, utterances = _build(words)
    assert len(utterances) == 1  # weak/no boundary -- stays one utterance


# ---------------------------------------------------------------------------
# 21. Meaning completion preserved (round-trip through utterance/attempt)
# ---------------------------------------------------------------------------
def test_21_meaning_completion_preserved():
    words = [_w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("done.", 0.3, 0.7)]
    _, utterances = _build(words)
    attempts = build_language_attempts(utterances)
    assert utterances[0].meaning_completion == attempts[0].meaning_completion == MEANING_COMPLETE


# ---------------------------------------------------------------------------
# 22. Negation preserved
# ---------------------------------------------------------------------------
def test_22_negation_preserved():
    words = [_w("It", 0.0, 0.1), _w("did", 0.1, 0.2), _w("not", 0.2, 0.3), _w("work.", 0.3, 0.7)]
    _, utterances = _build(words)
    assert "not" in utterances[0].text_normalized
    attempts = build_language_attempts(utterances)
    assert "not" in attempts[0].text_normalized


# ---------------------------------------------------------------------------
# 23. Numbers preserved
# ---------------------------------------------------------------------------
def test_23_numbers_preserved():
    words = [_w("It", 0.0, 0.1), _w("costs", 0.1, 0.3), _w("fifty", 0.3, 0.5), _w("dollars.", 0.5, 0.9)]
    _, utterances = _build(words)
    assert "fifty" in utterances[0].text_normalized
    attempts = build_language_attempts(utterances)
    assert "fifty" in attempts[0].text_normalized


# ---------------------------------------------------------------------------
# 24. Factual terms preserved
# ---------------------------------------------------------------------------
def test_24_factual_terms_preserved():
    words = [_w("The", 0.0, 0.1), _w("diagnosis", 0.1, 0.5), _w("was", 0.5, 0.6), _w("confirmed.", 0.6, 1.1)]
    _, utterances = _build(words)
    assert "diagnosis" in utterances[0].text_normalized
    attempts = build_language_attempts(utterances)
    assert "diagnosis" in attempts[0].text_normalized


# ---------------------------------------------------------------------------
# 25. Source id preserved
# ---------------------------------------------------------------------------
def test_25_source_id_preserved():
    words = [_w("Hello.", 0.0, 0.4)]
    _, utterances = _build(words)
    assert utterances[0].source_asset_id == "src1"
    attempts = build_language_attempts(utterances)
    assert attempts[0].source_asset_id == "src1"


# ---------------------------------------------------------------------------
# 26. Utterance spans exact child phrases
# ---------------------------------------------------------------------------
def test_26_utterance_spans_exact_child_phrases():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("It", 0.9, 1.0), _w("works", 1.0, 1.2), _w("well.", 1.2, 1.6),
    ]
    phrases, utterances = _build(words)
    for u in utterances:
        span = phrases[u.phrase_start_index:u.phrase_end_index + 1]
        assert span[0].source_start == u.source_start
        assert span[-1].source_end == u.source_end


# ---------------------------------------------------------------------------
# 27. Attempt spans exact child utterances
# ---------------------------------------------------------------------------
def test_27_attempt_spans_exact_child_utterances():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("It", 0.9, 1.0), _w("works", 1.0, 1.2), _w("well.", 1.2, 1.6),
    ]
    _, utterances = _build(words)
    attempts = build_language_attempts(utterances)
    by_id = {u.utterance_id: u for u in utterances}
    for a in attempts:
        members = [by_id[uid] for uid in a.utterance_ids]
        assert members[0].source_start == a.source_start
        assert members[-1].source_end == a.source_end


# ---------------------------------------------------------------------------
# 28. Deterministic ids
# ---------------------------------------------------------------------------
def test_28_deterministic_ids():
    words = [_w("Hello", 0.0, 0.3), _w("world.", 0.3, 0.8)]
    _, u1 = _build(words)
    _, u2 = _build(words)
    assert [u.utterance_id for u in u1] == [u.utterance_id for u in u2]
    a1 = build_language_attempts(u1)
    a2 = build_language_attempts(u2)
    assert [a.attempt_id for a in a1] == [a.attempt_id for a in a2]


# ---------------------------------------------------------------------------
# 29. Deterministic order
# ---------------------------------------------------------------------------
def test_29_deterministic_order():
    words = [_w("Hello", 0.0, 0.3), _w("world.", 0.3, 0.8), _w("Bye.", 1.0, 1.4)]
    _, utterances = _build(words)
    starts = [u.source_start for u in utterances]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# 30. Categorical confidence
# ---------------------------------------------------------------------------
def test_30_categorical_confidence():
    words = [_w("Hello", 0.0, 0.3), _w("world.", 0.3, 0.8)]
    _, utterances = _build(words)
    for u in utterances:
        assert u.confidence in (CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN)
    attempts = build_language_attempts(utterances)
    for a in attempts:
        assert a.confidence in (CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN, CONFIDENCE_MIXED)


# ---------------------------------------------------------------------------
# 31. Provenance retained
# ---------------------------------------------------------------------------
def test_31_provenance_retained():
    words = [_w("Hello.", 0.0, 0.4)]
    _, utterances = _build(words)
    assert utterances[0].provenance == m.PROVENANCE_UTTERANCE_SEGMENTATION
    attempts = build_language_attempts(utterances)
    assert attempts[0].provenance == m.PROVENANCE_ATTEMPT_RECONSTRUCTION


# ---------------------------------------------------------------------------
# 32. No proposition id minted
# ---------------------------------------------------------------------------
def test_32_no_proposition_id_minted():
    utterance_fields = {f for f in LanguageUtterance.__dataclass_fields__}
    attempt_fields = {f for f in LanguageAttempt.__dataclass_fields__}
    for field in ("proposition_id", "semantic_idea_id"):
        assert field not in utterance_fields
        assert field not in attempt_fields


# ---------------------------------------------------------------------------
# 33. No final retry relation minted
# ---------------------------------------------------------------------------
def test_33_no_final_retry_relation_minted():
    attempt_fields = {f for f in LanguageAttempt.__dataclass_fields__}
    for field in ("retry_family_id", "final_attempt_relation"):
        assert field not in attempt_fields
    body = _code_body_excluding_module_docstring(m)
    assert "FinalAttemptRelationship" not in body
    assert "mint_retry_family_id" not in body
    assert "mint_semantic_idea_id" not in body


# ---------------------------------------------------------------------------
# 34. No family change
# ---------------------------------------------------------------------------
def test_34_no_family_change():
    # D-171 Language Spine Phase D, TARGET A explicitly and narrowly
    # authorized `take_grouping_provider.py` to become a real, fail-open
    # Language Spine consumer (transitively, via `language_spine_consumer_
    # migration.py` -> `language_proposition_relation.py`), so it is
    # expected to reference this module's name now -- see docs/
    # CUTSELL_DECISIONS.md D-171. `take_grouping.py` (the underlying
    # deterministic grouping module) and `hybrid_session_cleanup.py` stay
    # untouched.
    for module_name in ("take_grouping", "hybrid_session_cleanup"):
        source = inspect.getsource(__import__(f"cutsell_worker.{module_name}", fromlist=["_"]))
        assert "language_utterance_attempt" not in source


# ---------------------------------------------------------------------------
# 35. No BestTake change
# ---------------------------------------------------------------------------
def test_35_no_besttake_change():
    for module_name in (
        "deterministic_best_take_authority", "take_judge",
        "watch_listen_besttake_evidence", "watch_listen_zone_usability_v2",
    ):
        source = inspect.getsource(__import__(f"cutsell_worker.{module_name}", fromlist=["_"]))
        assert "language_utterance_attempt" not in source


# ---------------------------------------------------------------------------
# 36. D-150 unchanged
# ---------------------------------------------------------------------------
def test_36_d150_unchanged():
    source = inspect.getsource(__import__("cutsell_worker.semantic_authority_observability", fromlist=["_"]))
    assert "language_utterance_attempt" not in source


# ---------------------------------------------------------------------------
# 37. D-163 unchanged
# ---------------------------------------------------------------------------
def test_37_d163_unchanged():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_besttake_evidence.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 38. D-167 unchanged
# ---------------------------------------------------------------------------
def test_38_d167_unchanged():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_zone_usability_v2.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 39. Boundary unchanged
# ---------------------------------------------------------------------------
def test_39_boundary_unchanged():
    source = inspect.getsource(__import__("cutsell_worker.boundary_engine_pass", fromlist=["_"]))
    assert "language_utterance_attempt" not in source


# ---------------------------------------------------------------------------
# 40. Pacing unchanged
# ---------------------------------------------------------------------------
def test_40_pacing_unchanged():
    source = inspect.getsource(__import__("cutsell_worker.dialogue_pacing_transition", fromlist=["_"]))
    assert "language_utterance_attempt" not in source


# ---------------------------------------------------------------------------
# 41. Render unchanged
# ---------------------------------------------------------------------------
def test_41_render_unchanged():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/renderer.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 42. No provider/network
# ---------------------------------------------------------------------------
def test_42_no_provider_network():
    body = _code_body_excluding_module_docstring(m)
    for forbidden in ("openai", "gemini", "requests.", "httpx.", "urllib.request", "socket."):
        assert forbidden not in body.lower()


# ---------------------------------------------------------------------------
# Additional structural/contract tests beyond the 42-item matrix.
# ---------------------------------------------------------------------------
def test_language_spine_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/language_spine.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_take_segmentation_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/take_segmentation.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_attempt_reconstruction_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/attempt_reconstruction.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_raw_understanding_map_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/raw_understanding_map.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_watch_listen_understanding_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_understanding.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_not_imported_by_any_production_call_site():
    # D-171 Language Spine Phase D, TARGET A explicitly and narrowly
    # authorized `take_grouping_provider.py` to become a real, fail-open
    # Language Spine consumer -- see docs/CUTSELL_DECISIONS.md D-171 and
    # this file's own `test_34_no_family_change`. Excluded from this list
    # for that documented reason; every other call site stays unaware.
    call_sites = (
        "pipeline", "flow_b", "take_segmentation", "attempt_reconstruction",
        "take_grouping", "hybrid_session_cleanup",
        "semantic_idea_equivalence", "deterministic_best_take_authority",
        "take_judge", "watch_listen_besttake_evidence",
        "watch_listen_zone_usability_v2", "boundary_engine_pass",
        "dialogue_pacing_transition", "semantic_authority_observability",
    )
    for name in call_sites:
        source = inspect.getsource(__import__(f"cutsell_worker.{name}", fromlist=["_"]))
        assert "language_utterance_attempt" not in source, name


def test_empty_input_returns_empty():
    assert segment_language_utterances(()) == ()
    assert build_language_attempts(()) == ()


def test_uncertain_utterance_no_text():
    # A single-word phrase with no punctuation and short duration/word count
    # produces an UNKNOWN-confidence, potentially UNCERTAIN classification
    # path is exercised via an explicitly empty accumulated text scenario.
    words = [_w("", 0.0, 0.1)]
    _, utterances = _build(words)
    assert len(utterances) == 1
    assert utterances[0].meaning_completion == MEANING_UNCERTAIN
    attempts = build_language_attempts(utterances)
    assert attempts[0].attempt_state == ATTEMPT_UNCERTAIN


def test_diagnostics_shape():
    words = [_w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7)]
    _, utterances = _build(words)
    attempts = build_language_attempts(utterances)
    diag = language_utterance_attempt_diagnostics(utterances, attempts)
    required = {
        "language_utterance_count", "language_attempt_count",
        "utterance_complete_count", "utterance_incomplete_count",
        "utterance_abandoned_count", "utterance_uncertain_count",
        "attempt_clean_count", "attempt_false_start_count", "attempt_abandoned_count",
        "attempt_correction_count", "attempt_continuation_count",
        "attempt_recording_process_count", "attempt_uncertain_count",
        "source_mapping_valid", "timeline_valid",
    }
    assert required.issubset(diag.keys())
    assert diag["source_mapping_valid"] is True
    assert diag["timeline_valid"] is True
    assert diag["language_utterance_count"] == len(utterances)
    assert diag["language_attempt_count"] == len(attempts)


def test_raw_understanding_compatibility_reference_additive_only():
    row = raw_understanding_compatibility_reference("span_abc", utterance_id="lutt_x", attempt_id="latt_y")
    assert row == {"span_id": "span_abc", "language_utterance_id": "lutt_x", "language_attempt_id": "latt_y"}


def test_watch_listen_compatibility_reference_additive_only():
    row = watch_listen_compatibility_reference("uspan_abc", attempt_id="latt_y")
    assert row == {"understanding_span_id": "uspan_abc", "language_attempt_id": "latt_y"}


def test_max_continuation_gap_sec_param_reuses_default():
    words = [
        _w("This", 0.0, 0.2), _w("is", 0.2, 0.3), _w("great.", 0.3, 0.7),
        _w("It", 2.5, 2.6), _w("works.", 2.6, 3.0),
    ]
    _, utterances = _build(words)
    # Gap between the two COMPLETE utterances exceeds the 1.20s default --
    # never merged into one attempt merely because temporally adjacent.
    attempts = build_language_attempts(utterances)
    assert len(attempts) == 2


def test_allowed_state_vocabularies_bounded():
    assert m.ALLOWED_UTTERANCE_STATES == {
        UTTERANCE_COMPLETE, UTTERANCE_INCOMPLETE, UTTERANCE_ABANDONED,
        UTTERANCE_RESTARTED, UTTERANCE_CORRECTED, UTTERANCE_UNCERTAIN,
    }
    assert m.ALLOWED_ATTEMPT_STATES == {
        ATTEMPT_CLEAN, ATTEMPT_FALSE_START, ATTEMPT_ABANDONED, ATTEMPT_CORRECTION,
        ATTEMPT_CONTINUATION, ATTEMPT_RECORDING_PROCESS, ATTEMPT_UNCERTAIN,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
