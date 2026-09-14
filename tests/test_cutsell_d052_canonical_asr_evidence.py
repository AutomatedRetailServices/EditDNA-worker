"""D-052 Part A: canonical ASR evidence + deterministic segment normalization.

See docs/CUTSELL_DECISIONS.md D-051 (audit) / D-052 (this fix). Covers:
- evidence_hash is content-only (timestamp/segmentation independent)
- normalize_transcript_segments produces equivalent output regardless of
  how the input was segmented (2 vs 3 segments, jitter, punctuation)
- take_segmentation.segment_takes's CUTSELL_ASR_CANONICAL_NORMALIZATION
  flag is OFF by default (parity with pre-D-052 behavior) and, when on,
  is genuinely segment-boundary independent end to end.
"""
from __future__ import annotations

from cutsell_worker.canonical_asr_evidence import (
    ASRConfigFingerprint,
    build_asr_config_fingerprint,
    build_canonical_asr_evidence,
    compute_evidence_hash,
    normalize_transcript_segments,
)
from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _source(source_asset_id: str = "src1", duration_sec: float = 60.0) -> SourceAsset:
    return SourceAsset(
        source_asset_id=source_asset_id,
        project_id="p1",
        user_id="u1",
        original_name="video.mp4",
        source_order=0,
        duration_sec=duration_sec,
        uri="s3://bucket/video.mp4",
    )


# ---------------------------------------------------------------------------
# evidence_hash / CanonicalASREvidence
# ---------------------------------------------------------------------------

def test_evidence_hash_is_identical_across_different_segment_groupings():
    words = [
        _word("Tuve", 0.0, 0.3),
        _word("problemas", 0.35, 0.8),
        _word("de", 0.85, 0.95),
        _word("estomago.", 1.0, 1.6),
        _word("Fui", 2.4, 2.6),
        _word("al", 2.65, 2.75),
        _word("doctor.", 2.8, 3.3),
    ]
    # Grouping A: everything in one segment.
    segments_a = (TranscriptSegment(
        source_asset_id="src1", start=0.0, end=3.3,
        text="Tuve problemas de estomago. Fui al doctor.", words=tuple(words),
    ),)
    # Grouping B: split into two segments at an arbitrary Whisper boundary.
    segments_b = (
        TranscriptSegment(source_asset_id="src1", start=0.0, end=1.6, text="Tuve problemas de estomago.", words=tuple(words[:4])),
        TranscriptSegment(source_asset_id="src1", start=2.4, end=3.3, text="Fui al doctor.", words=tuple(words[4:])),
    )
    evidence_a = build_canonical_asr_evidence(
        segments_a, source_asset_id="src1", language="es", asr_model="medium", asr_config_fingerprint="fp1",
    )
    evidence_b = build_canonical_asr_evidence(
        segments_b, source_asset_id="src1", language="es", asr_model="medium", asr_config_fingerprint="fp1",
    )
    assert evidence_a.evidence_hash == evidence_b.evidence_hash


def test_evidence_hash_is_timestamp_independent():
    words_a = [_word("hola", 0.0, 0.5), _word("mundo", 0.6, 1.0)]
    words_b = [_word("hola", 0.05, 0.55), _word("mundo", 0.72, 1.14)]  # jittered timing
    segments_a = (TranscriptSegment(source_asset_id="src1", start=0.0, end=1.0, text="hola mundo", words=tuple(words_a)),)
    segments_b = (TranscriptSegment(source_asset_id="src1", start=0.0, end=1.14, text="hola mundo", words=tuple(words_b)),)
    hash_a = compute_evidence_hash("src1", words_a, "es")
    hash_b = compute_evidence_hash("src1", words_b, "es")
    assert hash_a == hash_b
    ev_a = build_canonical_asr_evidence(segments_a, source_asset_id="src1", language="es", asr_model="m", asr_config_fingerprint="f")
    ev_b = build_canonical_asr_evidence(segments_b, source_asset_id="src1", language="es", asr_model="m", asr_config_fingerprint="f")
    assert ev_a.evidence_hash == ev_b.evidence_hash


def test_evidence_hash_changes_when_words_actually_differ():
    words_a = [_word("hola", 0.0, 0.5), _word("mundo", 0.6, 1.0)]
    words_b = [_word("hola", 0.0, 0.5), _word("amigo", 0.6, 1.0)]
    hash_a = compute_evidence_hash("src1", words_a, "es")
    hash_b = compute_evidence_hash("src1", words_b, "es")
    assert hash_a != hash_b


def test_build_canonical_asr_evidence_only_includes_matching_source():
    words1 = (_word("uno", 0.0, 0.3),)
    words2 = (_word("dos", 0.0, 0.3),)
    segments = (
        TranscriptSegment(source_asset_id="src1", start=0.0, end=0.3, text="uno", words=words1),
        TranscriptSegment(source_asset_id="src2", start=0.0, end=0.3, text="dos", words=words2),
    )
    evidence = build_canonical_asr_evidence(
        segments, source_asset_id="src1", language="es", asr_model="m", asr_config_fingerprint="f",
    )
    assert evidence.normalized_words == words1


# ---------------------------------------------------------------------------
# ASRConfigFingerprint
# ---------------------------------------------------------------------------

def test_config_fingerprint_is_stable_and_content_sensitive():
    fp1 = build_asr_config_fingerprint(
        model_name="medium", device="auto", compute_type="auto", beam_size=5, best_of=5,
        temperature_ladder=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), condition_on_previous_text=True,
        word_timestamps=True, vad_filter=True, language_hint="es",
    )
    fp2 = build_asr_config_fingerprint(
        model_name="medium", device="auto", compute_type="auto", beam_size=5, best_of=5,
        temperature_ladder=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), condition_on_previous_text=True,
        word_timestamps=True, vad_filter=True, language_hint="es",
    )
    assert fp1.fingerprint() == fp2.fingerprint()

    fp3 = build_asr_config_fingerprint(
        model_name="medium", device="auto", compute_type="float16", beam_size=5, best_of=5,
        temperature_ladder=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), condition_on_previous_text=True,
        word_timestamps=True, vad_filter=True, language_hint="es",
    )
    assert fp1.fingerprint() != fp3.fingerprint()
    assert isinstance(fp1, ASRConfigFingerprint)


# ---------------------------------------------------------------------------
# normalize_transcript_segments -- the D-052 Section 12 equivalence classes
# ---------------------------------------------------------------------------

def _base_words():
    # ~4 seconds of continuous speech, one true speech gap at ~4.5s, then a
    # second sentence.
    return [
        _word("Tuve", 0.00, 0.30),
        _word("problemas", 0.35, 0.80),
        _word("de", 0.85, 0.95),
        _word("estomago.", 1.00, 1.60),
        _word("Fui", 4.50, 4.70),
        _word("al", 4.75, 4.85),
        _word("doctor.", 4.90, 5.40),
    ]


def test_same_words_two_vs_three_segments_produce_identical_normalized_output():
    words = _base_words()
    two_segments = (
        TranscriptSegment(source_asset_id="s", start=0.0, end=1.60, text="Tuve problemas de estomago.", words=tuple(words[:4])),
        TranscriptSegment(source_asset_id="s", start=4.50, end=5.40, text="Fui al doctor.", words=tuple(words[4:])),
    )
    three_segments = (
        TranscriptSegment(source_asset_id="s", start=0.0, end=0.80, text="Tuve problemas", words=tuple(words[:2])),
        TranscriptSegment(source_asset_id="s", start=0.85, end=1.60, text="de estomago.", words=tuple(words[2:4])),
        TranscriptSegment(source_asset_id="s", start=4.50, end=5.40, text="Fui al doctor.", words=tuple(words[4:])),
    )
    normalized_two = normalize_transcript_segments(two_segments)
    normalized_three = normalize_transcript_segments(three_segments)
    assert [seg.text for seg in normalized_two] == [seg.text for seg in normalized_three]
    assert len(normalized_two) == 2


def test_timestamp_jitter_50_100_250ms_does_not_change_normalized_segmentation():
    base = _base_words()
    for jitter in (0.05, 0.10, 0.25):
        jittered = [Word(text=w.text, start=w.start + jitter, end=w.end + jitter, confidence=w.confidence) for w in base]
        segments = (TranscriptSegment(source_asset_id="s", start=jittered[0].start, end=jittered[-1].end, text="x", words=tuple(jittered)),)
        normalized = normalize_transcript_segments(segments)
        assert [seg.text for seg in normalized] == ["Tuve problemas de estomago.", "Fui al doctor."]


def test_vad_boundary_variation_within_the_gap_threshold_still_merges():
    # A Whisper run that drew its VAD boundary slightly differently (word
    # gap grouped into segment differently) but the underlying word-level
    # gap is still below split_gap_sec must normalize identically.
    words = [
        _word("Tuve", 0.00, 0.30),
        _word("problemas", 0.35, 0.80),
    ]
    variant_a = (TranscriptSegment(source_asset_id="s", start=0.0, end=0.80, text="Tuve problemas", words=tuple(words)),)
    variant_b = (
        TranscriptSegment(source_asset_id="s", start=0.0, end=0.30, text="Tuve", words=(words[0],)),
        TranscriptSegment(source_asset_id="s", start=0.35, end=0.80, text="problemas", words=(words[1],)),
    )
    assert [s.text for s in normalize_transcript_segments(variant_a)] == [s.text for s in normalize_transcript_segments(variant_b)]


def test_large_gap_still_splits_regardless_of_original_grouping():
    words = _base_words()
    one_segment = (TranscriptSegment(source_asset_id="s", start=0.0, end=5.40, text="all of it", words=tuple(words)),)
    normalized = normalize_transcript_segments(one_segment)
    # The 2.9s gap between "estomago." and "Fui" is >= the 0.75s split
    # threshold, so it must split into two segments even though the input
    # arrived as one.
    assert len(normalized) == 2
    assert normalized[0].text == "Tuve problemas de estomago."
    assert normalized[1].text == "Fui al doctor."


def test_punctuation_variation_is_a_deterministic_split_signal():
    # Same words, but the sentence-ending period is/isn't attached to the
    # final word's text -- both must still segment consistently on the
    # word-gap rule (punctuation is a secondary signal, gap is primary).
    words_with_period = [_word("Hola.", 0.0, 0.3), _word("Adios", 1.2, 1.5)]
    words_without_period = [_word("Hola", 0.0, 0.3), _word("Adios", 1.2, 1.5)]
    seg_with = (TranscriptSegment(source_asset_id="s", start=0.0, end=1.5, text="Hola. Adios", words=tuple(words_with_period)),)
    seg_without = (TranscriptSegment(source_asset_id="s", start=0.0, end=1.5, text="Hola Adios", words=tuple(words_without_period)),)
    normalized_with = normalize_transcript_segments(seg_with)
    normalized_without = normalize_transcript_segments(seg_without)
    # Both split on the >=0.75s gap regardless of punctuation.
    assert len(normalized_with) == 2
    assert len(normalized_without) == 2


def test_multiple_sources_are_normalized_independently_and_order_preserved():
    words_a = (_word("uno", 0.0, 0.3),)
    words_b = (_word("dos", 0.0, 0.3),)
    segments = (
        TranscriptSegment(source_asset_id="srcA", start=0.0, end=0.3, text="uno", words=words_a),
        TranscriptSegment(source_asset_id="srcB", start=0.0, end=0.3, text="dos", words=words_b),
    )
    normalized = normalize_transcript_segments(segments)
    assert [seg.source_asset_id for seg in normalized] == ["srcA", "srcB"]


def test_empty_input_returns_empty_output():
    assert normalize_transcript_segments(()) == ()


# ---------------------------------------------------------------------------
# take_segmentation.segment_takes flag integration
# ---------------------------------------------------------------------------

def test_segment_takes_flag_off_by_default_preserves_current_behavior():
    words = _base_words()
    segments = (
        TranscriptSegment(source_asset_id="src1", start=0.0, end=1.60, text="Tuve problemas de estomago.", words=tuple(words[:4])),
        TranscriptSegment(source_asset_id="src1", start=4.50, end=5.40, text="Fui al doctor.", words=tuple(words[4:])),
    )
    source = _source()
    default_env_takes = segment_takes(segments, (source,))
    explicit_off_takes = segment_takes(segments, (source,), env={"CUTSELL_ASR_CANONICAL_NORMALIZATION": "0"})
    assert [t.text for t in default_env_takes] == [t.text for t in explicit_off_takes]


def test_segment_takes_flag_on_is_segment_boundary_independent():
    words = _base_words()
    two_segments = (
        TranscriptSegment(source_asset_id="src1", start=0.0, end=1.60, text="Tuve problemas de estomago.", words=tuple(words[:4])),
        TranscriptSegment(source_asset_id="src1", start=4.50, end=5.40, text="Fui al doctor.", words=tuple(words[4:])),
    )
    three_segments = (
        TranscriptSegment(source_asset_id="src1", start=0.0, end=0.80, text="Tuve problemas", words=tuple(words[:2])),
        TranscriptSegment(source_asset_id="src1", start=0.85, end=1.60, text="de estomago.", words=tuple(words[2:4])),
        TranscriptSegment(source_asset_id="src1", start=4.50, end=5.40, text="Fui al doctor.", words=tuple(words[4:])),
    )
    source = _source()
    env = {"CUTSELL_ASR_CANONICAL_NORMALIZATION": "1"}
    takes_two = segment_takes(two_segments, (source,), env=env)
    takes_three = segment_takes(three_segments, (source,), env=env)
    assert [t.text for t in takes_two] == [t.text for t in takes_three]
