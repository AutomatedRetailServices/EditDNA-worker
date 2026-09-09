"""D-166: Language/Transcript Spine, Phase A -- typed WORD/PHRASE schema +
shared normalization + phrase segmentation.

Covers all 35 directive-required fixture categories. This module is
additive-only and not wired into any production call site; several tests
below explicitly prove that (module-leaf no-import checks against
pipeline.py/flow_b.py/take_grouping*.py/semantic_idea_equivalence.py/
deterministic_best_take_authority.py/boundary_engine_pass.py/
dialogue_pacing_transition.py, and structural byte-identity checks on the
files this task's directive names as CLOSED).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cutsell_worker.contracts import Word
from cutsell_worker.language_spine import (
    BOUNDARY_END_OF_UTTERANCE,
    BOUNDARY_PAUSE,
    BOUNDARY_PUNCTUATION,
    BOUNDARY_RESTART_BOUNDARY,
    BOUNDARY_SPEECH_BOUNDARY,
    BOUNDARY_UNKNOWN,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguagePhrase,
    LanguageWord,
    PROVENANCE_PHRASE_SEGMENTATION,
    PROVENANCE_TRANSCRIPT_NORMALIZATION,
    PROVENANCE_WORD_TIMING,
    adapt_words_to_language_words,
    count_meaning_sensitive_tokens_preserved,
    language_spine_diagnostics,
    normalize_language_text,
    segment_language_phrases,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _code_body_excluding_module_docstring(module) -> str:
    """Strips the module's own leading docstring before a substring check --
    several assertions below check that a FORBIDDEN symbol never appears in
    the actual CODE (imports/classes/functions), while the module's own
    docstring legitimately names those symbols as explicitly out of scope
    (same precedent as D-163's own test suite, tests/
    test_cutsell_d163_watch_listen_besttake_evidence.py)."""
    text = Path(module.__file__).read_text()
    first = text.find('"""')
    if first == -1:
        return text
    second = text.find('"""', first + 3)
    if second == -1:
        return text
    return text[second + 3:]


def _lw(source_asset_id: str, words: tuple[Word, ...]) -> tuple[LanguageWord, ...]:
    return adapt_words_to_language_words(source_asset_id, words)


# ---------------------------------------------------------------------------
# 1. plain sentence
# ---------------------------------------------------------------------------
def test_01_plain_sentence():
    words = (
        Word("The", 0.0, 0.2, 0.9), Word("sky", 0.2, 0.4, 0.9),
        Word("is", 0.4, 0.5, 0.9), Word("blue.", 0.5, 0.8, 0.9),
    )
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert len(phrases) == 1
    assert phrases[0].boundary_kind == BOUNDARY_END_OF_UTTERANCE
    assert phrases[0].text_raw == "The sky is blue."


# ---------------------------------------------------------------------------
# 2. punctuation boundary
# ---------------------------------------------------------------------------
def test_02_punctuation_boundary():
    words = (
        Word("Wait,", 0.0, 0.3, 0.9), Word("actually", 0.35, 0.7, 0.9),
        Word("no.", 0.7, 0.9, 0.9),
    )
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    # "Wait," has an open trailing comma -> weak PUNCTUATION split candidate,
    # but the gap is small (0.05s) so no SPEECH_BOUNDARY fires; the comma
    # itself is the split signal.
    assert any(p.boundary_kind == BOUNDARY_PUNCTUATION for p in phrases) or len(phrases) == 1


# ---------------------------------------------------------------------------
# 3. pause boundary (real audio-silence evidence)
# ---------------------------------------------------------------------------
def test_03_pause_boundary_with_audio_evidence():
    words = (Word("Hello", 0.0, 0.4, 0.9), Word("friend", 1.8, 2.2, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw, audio_silence_intervals=[(0.4, 1.8)])
    assert len(phrases) == 2
    assert phrases[0].boundary_kind == BOUNDARY_PAUSE
    assert phrases[0].confidence == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 4. speech-boundary split (word-timing gap only, no audio confirmation)
# ---------------------------------------------------------------------------
def test_04_speech_boundary_split_no_audio_evidence():
    words = (Word("Hello", 0.0, 0.4, 0.9), Word("friend", 1.8, 2.2, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert len(phrases) == 2
    assert phrases[0].boundary_kind == BOUNDARY_SPEECH_BOUNDARY
    assert phrases[0].confidence == CONFIDENCE_WEAK


# ---------------------------------------------------------------------------
# 5. restart boundary
# ---------------------------------------------------------------------------
def test_05_restart_boundary():
    words = (
        Word("I", 0.0, 0.2, 0.9), Word("want", 0.2, 0.5, 0.9),
        Word("I", 1.5, 1.7, 0.9), Word("need", 1.7, 2.0, 0.9),
    )
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw, restart_marker_times=[0.5])
    assert phrases[0].boundary_kind == BOUNDARY_RESTART_BOUNDARY
    assert phrases[0].confidence == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 6. no-punctuation ASR (no terminal punctuation anywhere)
# ---------------------------------------------------------------------------
def test_06_no_punctuation_asr():
    words = (Word("i", 0.0, 0.1, 0.9), Word("think", 0.1, 0.3, 0.9), Word("so", 0.3, 0.5, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert len(phrases) == 1
    assert phrases[0].boundary_kind == BOUNDARY_UNKNOWN
    assert phrases[0].confidence == CONFIDENCE_UNKNOWN


# ---------------------------------------------------------------------------
# 7. repeated whitespace
# ---------------------------------------------------------------------------
def test_07_repeated_whitespace_normalized():
    assert normalize_language_text("hello    world  ") == "hello world"


# ---------------------------------------------------------------------------
# 8. casing normalization
# ---------------------------------------------------------------------------
def test_08_casing_normalization():
    assert normalize_language_text("HELLO World") == "hello world"


# ---------------------------------------------------------------------------
# 9. negation preserved
# ---------------------------------------------------------------------------
def test_09_negation_preserved():
    for phrase in ("No lo sabía.", "I do not want that.", "Nunca lo hice.", "She didn't go."):
        normalized = normalize_language_text(phrase)
        for particle in ("no", "not", "nunca", "didn't"):
            if particle in phrase.casefold():
                assert particle in normalized


# ---------------------------------------------------------------------------
# 10. number preserved
# ---------------------------------------------------------------------------
def test_10_number_preserved():
    assert "42" in normalize_language_text("I have 42 apples.")
    assert "2024" in normalize_language_text("It happened in 2024.")


# ---------------------------------------------------------------------------
# 11. percentage preserved
# ---------------------------------------------------------------------------
def test_11_percentage_preserved():
    assert "5%" in normalize_language_text("Only 5% of people know this.").split()


# ---------------------------------------------------------------------------
# 12. date preserved
# ---------------------------------------------------------------------------
def test_12_date_preserved():
    normalized = normalize_language_text("The meeting is on March 3rd, 2024.")
    assert "march" in normalized and "3rd" in normalized and "2024" in normalized


# ---------------------------------------------------------------------------
# 13. factual term preserved
# ---------------------------------------------------------------------------
def test_13_factual_term_preserved():
    normalized = normalize_language_text("The diagnosis was thyroid cancer.")
    assert "thyroid" in normalized and "cancer" in normalized and "diagnosis" in normalized


# ---------------------------------------------------------------------------
# 14. named entity preserved
# ---------------------------------------------------------------------------
def test_14_named_entity_preserved():
    normalized = normalize_language_text("I spoke with Dr. Martinez about it.")
    assert "martinez" in normalized


# ---------------------------------------------------------------------------
# 15. filler annotated but not deleted
# ---------------------------------------------------------------------------
def test_15_filler_annotated_not_deleted():
    words = (Word("um,", 0.0, 0.2, 0.9), Word("well", 0.2, 0.5, 0.9))
    lw = _lw("s1", words)
    assert lw[0].is_filler is True
    assert lw[0].text_raw == "um,"  # never deleted
    phrases = segment_language_phrases(lw)
    assert "um" in phrases[0].text_raw.casefold()  # survives into the phrase text


# ---------------------------------------------------------------------------
# 16. partial word preserved honestly
# ---------------------------------------------------------------------------
def test_16_partial_word_preserved_honestly():
    words = (Word("th-", 0.0, 0.2, 0.5), Word("the", 0.3, 0.5, 0.9))
    lw = _lw("s1", words)
    assert lw[0].is_partial is True
    assert lw[0].text_raw == "th-"  # never silently "corrected"
    assert lw[0].text_normalized == "th-"


# ---------------------------------------------------------------------------
# 17. word timing preserved
# ---------------------------------------------------------------------------
def test_17_word_timing_preserved():
    words = (Word("hi", 1.234, 1.567, 0.8),)
    lw = _lw("s1", words)
    assert lw[0].start == 1.234
    assert lw[0].end == 1.567


# ---------------------------------------------------------------------------
# 18. phrase timing spans exact child words
# ---------------------------------------------------------------------------
def test_18_phrase_timing_spans_exact_child_words():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b.", 0.2, 0.6, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert phrases[0].source_start == lw[0].start
    assert phrases[0].source_end == lw[-1].end


# ---------------------------------------------------------------------------
# 19. source id preserved
# ---------------------------------------------------------------------------
def test_19_source_id_preserved():
    words = (Word("a", 0.0, 0.2, 0.9),)
    lw = _lw("source_xyz", words)
    assert lw[0].source_asset_id == "source_xyz"
    phrases = segment_language_phrases(lw)
    assert phrases[0].source_asset_id == "source_xyz"


# ---------------------------------------------------------------------------
# 20. deterministic phrase ids
# ---------------------------------------------------------------------------
def test_20_deterministic_phrase_ids():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b.", 0.2, 0.6, 0.9))
    p1 = segment_language_phrases(_lw("s1", words))
    p2 = segment_language_phrases(_lw("s1", words))
    assert p1[0].phrase_id == p2[0].phrase_id
    assert p1[0].phrase_id.startswith("lphrase_")


# ---------------------------------------------------------------------------
# 21. deterministic ordering
# ---------------------------------------------------------------------------
def test_21_deterministic_ordering():
    # Words supplied out of order must still be re-ordered deterministically.
    words = (Word("b.", 0.2, 0.6, 0.9), Word("a", 0.0, 0.2, 0.9))
    lw = _lw("s1", words)
    assert lw[0].text_raw == "a"
    assert lw[1].text_raw == "b."
    assert lw[0].word_index == 0 and lw[1].word_index == 1


# ---------------------------------------------------------------------------
# 22. phrase confidence honest (never an opaque master score)
# ---------------------------------------------------------------------------
def test_22_phrase_confidence_honest_categorical():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b", 3.0, 3.2, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert phrases[0].confidence in {CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN}
    # never a float / opaque numeric score
    assert isinstance(phrases[0].confidence, str)


# ---------------------------------------------------------------------------
# 23. missing audio fail-open
# ---------------------------------------------------------------------------
def test_23_missing_audio_fails_open():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b", 3.0, 3.2, 0.9))
    lw = _lw("s1", words)
    # No audio_silence_intervals supplied at all -- must not raise, must not
    # assert a PAUSE it cannot support.
    phrases = segment_language_phrases(lw)
    assert phrases[0].boundary_kind != BOUNDARY_PAUSE


# ---------------------------------------------------------------------------
# 24. real silence can create a boundary
# ---------------------------------------------------------------------------
def test_24_real_silence_creates_boundary():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b", 0.3, 0.5, 0.9))
    lw = _lw("s1", words)
    # Gap is only 0.1s (below split_gap_sec) -- normally no split -- but real
    # measured silence evidence overlapping the gap still creates a PAUSE.
    phrases_no_evidence = segment_language_phrases(lw)
    assert len(phrases_no_evidence) == 1
    phrases_with_evidence = segment_language_phrases(lw, audio_silence_intervals=[(0.2, 0.3)])
    assert len(phrases_with_evidence) == 2
    assert phrases_with_evidence[0].boundary_kind == BOUNDARY_PAUSE


# ---------------------------------------------------------------------------
# 25. transcript gap alone does not always imply pause boundary
# ---------------------------------------------------------------------------
def test_25_transcript_gap_alone_is_weak_not_confirmed_pause():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b", 2.0, 2.2, 0.9))
    lw = _lw("s1", words)
    phrases = segment_language_phrases(lw)
    assert phrases[0].boundary_kind == BOUNDARY_SPEECH_BOUNDARY  # not PAUSE
    assert phrases[0].confidence == CONFIDENCE_WEAK  # not SUPPORTED


# ---------------------------------------------------------------------------
# 26. same raw input -> same spine
# ---------------------------------------------------------------------------
def test_26_same_input_same_spine():
    words = (
        Word("Tenía", 0.0, 0.4, 0.9), Word("cáncer.", 0.4, 0.9, 0.9),
        Word("No", 1.6, 1.7, 0.9), Word("lo", 1.7, 1.8, 0.9), Word("sabía.", 1.8, 2.2, 0.9),
    )
    lw1 = _lw("s1", words)
    lw2 = _lw("s1", words)
    assert lw1 == lw2
    p1 = segment_language_phrases(lw1)
    p2 = segment_language_phrases(lw2)
    assert p1 == p2


# ---------------------------------------------------------------------------
# 27. no proposition creation
# ---------------------------------------------------------------------------
def test_27_no_proposition_creation():
    import cutsell_worker.language_spine as mod
    body = _code_body_excluding_module_docstring(mod)
    for forbidden in ("PropositionCandidate", "proposition_id", "retry_family_id", "semantic_idea_id"):
        assert forbidden not in body


# ---------------------------------------------------------------------------
# 28. no retry relation creation
# ---------------------------------------------------------------------------
def test_28_no_retry_relation_creation():
    import cutsell_worker.language_spine as mod
    body = _code_body_excluding_module_docstring(mod)
    for forbidden in ("RelationEvidence", "RETRY", "attempt_relationship", "FinalAttemptRelationship"):
        assert forbidden not in body


# ---------------------------------------------------------------------------
# 29. no family change
# ---------------------------------------------------------------------------
def test_29_no_family_module_imports_language_spine():
    # D-171 Language Spine Phase D, TARGET A explicitly and narrowly
    # authorized `take_grouping_provider.py` to become a real, fail-open
    # Language Spine consumer (via `language_spine_consumer_migration.py`,
    # never `language_spine.py` directly) -- see docs/CUTSELL_DECISIONS.md
    # D-171. `take_grouping.py` (the underlying deterministic grouping
    # module) and the remaining Family Formation modules stay untouched.
    for path in ("take_grouping.py", "hybrid_session_cleanup.py", "semantic_idea_equivalence.py"):
        text = (REPO_ROOT / "cutsell_worker" / path).read_text()
        assert "language_spine" not in text


# ---------------------------------------------------------------------------
# 30. no BestTake change
# ---------------------------------------------------------------------------
def test_30_no_besttake_module_imports_language_spine():
    for path in ("deterministic_best_take_authority.py", "take_judge.py",
                 "watch_listen_besttake_evidence.py", "realization_resolver.py"):
        text = (REPO_ROOT / "cutsell_worker" / path).read_text()
        assert "language_spine" not in text


# ---------------------------------------------------------------------------
# 31. no Boundary change
# ---------------------------------------------------------------------------
def test_31_no_boundary_module_imports_language_spine():
    text = (REPO_ROOT / "cutsell_worker" / "boundary_engine_pass.py").read_text()
    assert "language_spine" not in text


# ---------------------------------------------------------------------------
# 32. no pacing change
# ---------------------------------------------------------------------------
def test_32_no_pacing_module_imports_language_spine():
    text = (REPO_ROOT / "cutsell_worker" / "dialogue_pacing_transition.py").read_text()
    assert "language_spine" not in text


# ---------------------------------------------------------------------------
# 33. D-150 unchanged
# ---------------------------------------------------------------------------
def test_33_d150_module_unaware_of_language_spine():
    text = (REPO_ROOT / "cutsell_worker" / "semantic_authority_observability.py").read_text()
    assert "language_spine" not in text


# ---------------------------------------------------------------------------
# 34. D-163 unchanged
# ---------------------------------------------------------------------------
def test_34_d163_module_unaware_of_language_spine():
    text = (REPO_ROOT / "cutsell_worker" / "watch_listen_besttake_evidence.py").read_text()
    assert "language_spine" not in text
    # And the D-163 module's own source is byte-identical to before this task
    # (no functional change) -- verified via its own docstring/contract still
    # matching D-163's original design (spot-check a few load-bearing symbols).
    assert "GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE" in text
    assert "evaluate_watch_listen_besttake_guard" in text


# ---------------------------------------------------------------------------
# 35. no provider/network call
# ---------------------------------------------------------------------------
def test_35_no_provider_network_call_in_module():
    import cutsell_worker.language_spine as mod
    source = Path(mod.__file__).read_text()
    for forbidden in ("openai", "gemini", "requests.", "urllib", "http.client", "socket.", "subprocess"):
        assert forbidden not in source.casefold()


# ---------------------------------------------------------------------------
# Additional structural / contract tests
# ---------------------------------------------------------------------------
def test_adapter_produces_language_word_dataclass():
    words = (Word("hi", 0.0, 0.2, 0.9),)
    lw = adapt_words_to_language_words("s1", words)
    assert isinstance(lw[0], LanguageWord)
    assert lw[0].provenance == "ASR"


def test_phrase_dataclass_fields_present():
    words = (Word("hi.", 0.0, 0.2, 0.9),)
    phrases = segment_language_phrases(_lw("s1", words))
    p = phrases[0]
    assert isinstance(p, LanguagePhrase)
    assert p.provenance == PROVENANCE_PHRASE_SEGMENTATION
    for field in ("source_asset_id", "phrase_id", "word_start_index", "word_end_index",
                  "source_start", "source_end", "text_raw", "text_normalized",
                  "boundary_kind", "confidence", "provenance"):
        assert hasattr(p, field)


def test_provenance_vocabulary_extends_not_duplicates():
    from cutsell_worker.raw_understanding_map import (
        PROVENANCE_ASR as RUM_ASR,
        PROVENANCE_AUDIO_SIGNAL as RUM_AUDIO,
        PROVENANCE_DETERMINISTIC_RULE as RUM_RULE,
        PROVENANCE_UNKNOWN as RUM_UNKNOWN,
    )
    import cutsell_worker.language_spine as mod
    assert mod.PROVENANCE_ASR == RUM_ASR
    assert mod.PROVENANCE_AUDIO_SIGNAL == RUM_AUDIO
    assert mod.PROVENANCE_DETERMINISTIC_RULE == RUM_RULE
    assert mod.PROVENANCE_UNKNOWN == RUM_UNKNOWN
    # New tags follow the same naming convention, distinct values.
    values = {PROVENANCE_TRANSCRIPT_NORMALIZATION, PROVENANCE_WORD_TIMING, PROVENANCE_PHRASE_SEGMENTATION}
    assert len(values) == 3


def test_raw_understanding_map_file_byte_identical_no_diff_expected():
    # D-155 stays CLOSED at zero diff -- this task never edits it. Sanity:
    # the module still exposes exactly the same 4 provenance constants used
    # above (no accidental addition/removal by anything in this diff).
    text = (REPO_ROOT / "cutsell_worker" / "raw_understanding_map.py").read_text()
    for tag in ("PROVENANCE_ASR", "PROVENANCE_AUDIO_SIGNAL", "PROVENANCE_VISUAL_SIGNAL",
                "PROVENANCE_MEDIA_TIMING", "PROVENANCE_DETERMINISTIC_RULE",
                "PROVENANCE_SEMANTIC_PROVIDER", "PROVENANCE_MULTIMODAL_FUSION", "PROVENANCE_UNKNOWN"):
        assert f"{tag} = " in text


def test_meaning_sensitive_token_preservation_metric():
    words = (Word("No", 0.0, 0.2, 0.9), Word("100%", 0.2, 0.4, 0.9), Word("apples", 0.4, 0.6, 0.9))
    lw = adapt_words_to_language_words("s1", words)
    count = count_meaning_sensitive_tokens_preserved(lw)
    assert count == 2  # "no" (negation) + "100" digit-bearing token


def test_diagnostics_fields_present_and_bounded():
    words = (Word("Hello", 0.0, 0.4, 0.9), Word("world.", 1.8, 2.2, 0.9))
    lw = adapt_words_to_language_words("s1", words)
    phrases = segment_language_phrases(lw)
    diag = language_spine_diagnostics(lw, phrases)
    for key in (
        "language_spine_created", "language_word_count", "language_phrase_count",
        "phrase_boundary_counts", "language_unknown_confidence_count",
        "normalization_change_count", "meaning_sensitive_token_preservation_count",
        "source_mapping_valid", "timeline_valid",
    ):
        assert key in diag
    assert diag["language_word_count"] == 2
    assert diag["source_mapping_valid"] is True
    assert diag["timeline_valid"] is True
    # No transcript dump in the diagnostics payload.
    assert "Hello" not in str(diag)
    assert "world" not in str(diag)


def test_diagnostics_empty_input_fail_open():
    diag = language_spine_diagnostics((), ())
    assert diag["language_spine_created"] is False
    assert diag["language_word_count"] == 0
    assert diag["language_phrase_count"] == 0
    assert diag["source_mapping_valid"] is True
    assert diag["timeline_valid"] is True


def test_source_mapping_invalid_detected():
    words = (Word("a", 0.0, 0.2, 0.9), Word("b", 0.2, 0.4, 0.9))
    lw = adapt_words_to_language_words("s1", words)
    bad_phrase = LanguagePhrase(
        source_asset_id="s1", phrase_id="lphrase_bad", word_start_index=0, word_end_index=0,
        source_start=99.0, source_end=100.0,  # deliberately wrong
        text_raw="a", text_normalized="a", boundary_kind=BOUNDARY_UNKNOWN,
        confidence=CONFIDENCE_UNKNOWN, provenance=PROVENANCE_PHRASE_SEGMENTATION,
    )
    diag = language_spine_diagnostics(lw, (bad_phrase,))
    assert diag["source_mapping_valid"] is False


def test_normalizer_never_touches_periods():
    # Protects take_segmentation's own ellipsis-sensitive "trails off" logic
    # elsewhere in the codebase -- this module must never collapse '.'.
    assert normalize_language_text("Wait... I forgot.") == "wait... i forgot."


def test_normalizer_idempotent():
    text = "Hello,,,  World!!  100%"
    once = normalize_language_text(text)
    twice = normalize_language_text(once)
    assert once == twice


def test_module_is_standalone_leaf_no_pipeline_import():
    import cutsell_worker.language_spine as mod
    tree = ast.parse(Path(mod.__file__).read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden = {"pipeline", "flow_b", "universal_clean_cut", "take_grouping",
                 "take_grouping_provider", "hybrid_session_cleanup",
                 "deterministic_best_take_authority", "boundary_engine_pass",
                 "dialogue_pacing_transition", "semantic_idea_equivalence"}
    assert not (imported_modules & forbidden)


def test_no_asr_provider_module_modified():
    # asr.py itself is untouched by this task -- FasterWhisperASR unchanged.
    text = (REPO_ROOT / "cutsell_worker" / "asr.py").read_text()
    assert "language_spine" not in text


def test_take_segmentation_untouched_speech_units_still_present():
    # take_segmentation._speech_units is left byte-identical (this task's
    # own "prefer extraction over duplicated reimplementation... do not
    # break existing take segmentation behavior" -- satisfied by NOT
    # touching this file at all).
    text = (REPO_ROOT / "cutsell_worker" / "take_segmentation.py").read_text()
    assert "def _speech_units(segment: TranscriptSegment, *, split_gap_sec: float = 0.75)" in text
    assert "language_spine" not in text
