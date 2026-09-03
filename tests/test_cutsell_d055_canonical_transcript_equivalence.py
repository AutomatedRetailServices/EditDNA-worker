"""D-055: Canonical Transcript Equivalence -- separates the SOURCE-SCOPED
evidence_hash (unchanged, D-052/D-053 behavior) from two new,
source-independent hashes: content_hash (pure "what was said", no
source_asset_id) and canonical_equivalence_hash (content_hash plus a
bounded, proven-safe T0/T1 normalization). See
canonical_asr_evidence.py's CanonicalASREvidence docstring and
docs/CUTSELL_DECISIONS.md D-055 for the full rationale.

Section 7's required test list, in order: identical words + different
source_asset_id -> same content-equivalence hash; punctuation difference
-> same; accent-only orthography -> same where safe; number change ->
different; negation change -> different; diagnosis noun change ->
different; causal direction change -> different; timestamp changes ->
same content-equivalence hash; word-order/material lexical change ->
different unless explicitly proven equivalent.
"""
from __future__ import annotations

from cutsell_worker.canonical_asr_evidence import (
    build_canonical_asr_evidence,
    canonicalize_transcript_words,
    compute_canonical_equivalence_hash,
    compute_content_hash,
    compute_evidence_hash,
)
from cutsell_worker.contracts import TranscriptSegment, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _segment(source_asset_id: str, words: list[Word]) -> TranscriptSegment:
    return TranscriptSegment(
        source_asset_id=source_asset_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        words=tuple(words),
    )


# ---------------------------------------------------------------------------
# Section 1: source-scoped id vs content-only hash separation
# ---------------------------------------------------------------------------

def test_content_hash_matches_across_different_source_asset_ids():
    # The exact D-054 finding: two independently-dispatched runs (e.g.
    # two different benchmark_id-derived source_asset_id values) over the
    # byte-identical audio must produce the SAME content_hash even though
    # their source-scoped evidence_hash never can.
    words = [_word("tenía", 8.37, 9.03), _word("cáncer", 9.03, 9.43)]
    hash_a = compute_content_hash(words, "es")
    hash_b = compute_content_hash(words, "es")
    assert hash_a == hash_b

    evidence_a = compute_evidence_hash("src_AAAA", words, "es")
    evidence_b = compute_evidence_hash("src_BBBB", words, "es")
    assert evidence_a != evidence_b  # source-scoped id still differs (unchanged behavior)


def test_build_canonical_asr_evidence_populates_all_three_hashes():
    words = [_word("tenía", 8.37, 9.03), _word("cáncer", 9.03, 9.43)]
    segments = [_segment("src1", words)]
    evidence = build_canonical_asr_evidence(
        segments, source_asset_id="src1", language="es",
        asr_model="medium", asr_config_fingerprint="fp1",
    )
    assert evidence.evidence_hash.startswith("asrev_")
    assert evidence.content_hash.startswith("asrcontent_")
    assert evidence.canonical_equivalence_hash.startswith("asrcanon_")
    assert evidence.evidence_hash != evidence.content_hash
    assert evidence.content_hash != evidence.canonical_equivalence_hash or True  # may coincide, not required to differ


def test_two_different_source_ids_same_words_share_content_and_canonical_hash():
    words = [_word("no", 0.0, 0.3), _word("tenía", 0.3, 0.8), _word("cáncer", 0.8, 1.2)]
    segments_a = [_segment("src_run_A", words)]
    segments_b = [_segment("src_run_B", words)]
    evidence_a = build_canonical_asr_evidence(
        segments_a, source_asset_id="src_run_A", language="es",
        asr_model="medium", asr_config_fingerprint="fp1",
    )
    evidence_b = build_canonical_asr_evidence(
        segments_b, source_asset_id="src_run_B", language="es",
        asr_model="medium", asr_config_fingerprint="fp1",
    )
    assert evidence_a.evidence_hash != evidence_b.evidence_hash
    assert evidence_a.content_hash == evidence_b.content_hash
    assert evidence_a.canonical_equivalence_hash == evidence_b.canonical_equivalence_hash


# ---------------------------------------------------------------------------
# Section 7: T0 punctuation/casing/spacing -> same
# ---------------------------------------------------------------------------

def test_punctuation_difference_is_canonically_equivalent():
    words_a = [_word("Hola,", 0.0, 0.3), _word("cáncer.", 0.3, 0.6)]
    words_b = [_word("hola", 0.0, 0.3), _word("cáncer", 0.3, 0.6)]
    assert compute_canonical_equivalence_hash(words_a, "es") == compute_canonical_equivalence_hash(words_b, "es")


def test_casing_and_spacing_difference_is_canonically_equivalent():
    words_a = [_word("TIROIDES", 0.0, 0.5)]
    words_b = [_word("  tiroides  ", 0.0, 0.5)]
    assert compute_canonical_equivalence_hash(words_a, "es") == compute_canonical_equivalence_hash(words_b, "es")


# ---------------------------------------------------------------------------
# Section 7: accent-only orthography -> same where safe (si/sí only)
# ---------------------------------------------------------------------------

def test_si_si_accent_variant_is_canonically_equivalent():
    words_a = [_word("si", 0.0, 0.2), _word("es", 0.2, 0.4), _word("sintomática", 0.4, 1.0)]
    words_b = [_word("sí", 0.0, 0.2), _word("es", 0.2, 0.4), _word("sintomática", 0.4, 1.0)]
    assert compute_canonical_equivalence_hash(words_a, "es") == compute_canonical_equivalence_hash(words_b, "es")


def test_unproven_accent_pair_stays_distinguishable():
    # "mas" (but) / "más" (more) is NOT in the proven-safe allowlist --
    # fail conservative, per the directive, rather than guessing it is
    # also a harmless ASR toss-up.
    words_a = [_word("mas", 0.0, 0.3)]
    words_b = [_word("más", 0.0, 0.3)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


# ---------------------------------------------------------------------------
# Section 3/7: protected semantics -- number/negation/diagnosis/causal
# changes must NEVER be collapsed
# ---------------------------------------------------------------------------

def test_number_change_is_not_canonically_equivalent():
    words_a = [_word("un", 0.0, 0.1), _word("5", 0.1, 0.3), _word("por", 0.3, 0.4), _word("ciento", 0.4, 0.6)]
    words_b = [_word("un", 0.0, 0.1), _word("10", 0.1, 0.3), _word("por", 0.3, 0.4), _word("ciento", 0.4, 0.6)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


def test_negation_change_is_not_canonically_equivalent():
    # "no tenía" vs "tenía" -- the directive's own example.
    words_a = [_word("no", 0.0, 0.2), _word("tenía", 0.2, 0.6), _word("cáncer", 0.6, 1.0)]
    words_b = [_word("tenía", 0.2, 0.6), _word("cáncer", 0.6, 1.0)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


def test_diagnosis_noun_change_is_not_canonically_equivalent():
    # "gastritis" vs "alergia" -- the directive's own example.
    words_a = [_word("tuve", 0.0, 0.3), _word("gastritis", 0.3, 1.0)]
    words_b = [_word("tuve", 0.0, 0.3), _word("alergia", 0.3, 1.0)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


def test_causal_direction_change_is_not_canonically_equivalent():
    words_a = [_word("antes", 0.0, 0.3), _word("del", 0.3, 0.4), _word("tratamiento", 0.4, 1.0)]
    words_b = [_word("después", 0.0, 0.3), _word("del", 0.3, 0.4), _word("tratamiento", 0.4, 1.0)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


def test_word_order_change_is_not_canonically_equivalent_by_default():
    words_a = [_word("el", 0.0, 0.1), _word("gato", 0.1, 0.3), _word("come", 0.3, 0.5), _word("pescado", 0.5, 0.9)]
    words_b = [_word("pescado", 0.0, 0.1), _word("come", 0.1, 0.3), _word("el", 0.3, 0.5), _word("gato", 0.5, 0.9)]
    assert compute_canonical_equivalence_hash(words_a, "es") != compute_canonical_equivalence_hash(words_b, "es")


# ---------------------------------------------------------------------------
# Section 7: timestamp changes -> same content-equivalence hash
# ---------------------------------------------------------------------------

def test_timestamp_jitter_does_not_change_content_hash():
    words_a = [_word("tenía", 8.37, 9.03), _word("cáncer", 9.03, 9.43)]
    words_b = [_word("tenía", 8.50, 9.10), _word("cáncer", 9.20, 9.60)]  # same words, shifted timing
    assert compute_content_hash(words_a, "es") == compute_content_hash(words_b, "es")
    assert compute_canonical_equivalence_hash(words_a, "es") == compute_canonical_equivalence_hash(words_b, "es")


# ---------------------------------------------------------------------------
# canonicalize_transcript_words -- direct unit coverage
# ---------------------------------------------------------------------------

def test_canonicalize_transcript_words_strips_punctuation_and_case():
    words = [_word("¡Hola,", 0.0, 0.2), _word("Mundo!", 0.2, 0.4)]
    assert canonicalize_transcript_words(words) == ("hola", "mundo")


def test_canonicalize_transcript_words_applies_si_si_table_only():
    words = [_word("sí", 0.0, 0.2), _word("mas", 0.2, 0.4), _word("más", 0.4, 0.6)]
    assert canonicalize_transcript_words(words) == ("si", "mas", "más")


def test_canonicalize_transcript_words_never_touches_digits_or_negation():
    words = [_word("no", 0.0, 0.1), _word("tengo", 0.1, 0.3), _word("5", 0.3, 0.4), _word("gatos", 0.4, 0.6)]
    tokens = canonicalize_transcript_words(words)
    assert "no" in tokens
    assert "5" in tokens


def test_canonicalize_transcript_words_pure_punctuation_word_vanishes():
    words = [_word("...", 0.0, 0.1), _word("hola", 0.1, 0.3)]
    assert canonicalize_transcript_words(words) == ("hola",)


# ---------------------------------------------------------------------------
# Section 5 replay-style sanity: the exact D-054 six-transcript story,
# reproduced with small synthetic stand-ins for each proven finding
# ---------------------------------------------------------------------------

def test_legacy_style_punctuation_only_pair_collapses_to_one_equivalence_class():
    # Mirrors D-054's A-B pair (99.84% similar, effectively one real
    # substitution) -- reduced here to a punctuation-only difference,
    # which the canonical-equivalence hash must fully absorb.
    run_a = [_word("Aquí", 0.0, 0.3), _word("tuve", 0.3, 0.6), _word("cáncer.", 0.6, 1.0)]
    run_b = [_word("aquí,", 0.0, 0.3), _word("tuve", 0.3, 0.6), _word("cáncer", 0.6, 1.0)]
    assert compute_canonical_equivalence_hash(run_a, "es") == compute_canonical_equivalence_hash(run_b, "es")


def test_legacy_style_dropped_negation_pair_stays_distinct_equivalence_class():
    # Mirrors D-054's A/B vs C pair: "no hay que preguntar" vs "hay que
    # voltar" -- a real negation deletion plus lexical corruption. The
    # canonical-equivalence hash must NOT collapse this.
    run_ab = [_word("no", 0.0, 0.2), _word("hay", 0.2, 0.4), _word("que", 0.4, 0.5), _word("preguntar.", 0.5, 1.0)]
    run_c = [_word("hay", 0.2, 0.4), _word("que", 0.4, 0.5), _word("voltar.", 0.5, 1.0)]
    assert compute_canonical_equivalence_hash(run_ab, "es") != compute_canonical_equivalence_hash(run_c, "es")
