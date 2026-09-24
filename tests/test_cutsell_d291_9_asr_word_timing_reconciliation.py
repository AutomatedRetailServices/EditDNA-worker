"""D-291.9: ASR word spans reconciled against the measured source silence.

Real data: RAW #126 (project video00-modal-35931819172... no: video00-modal-
35931561397-1), its own timed ASR words and ffmpeg silencedetect events.
Verified on the source audio (speech energy) and on the frames: "resorcina"
is spoken at ~189.6-190.0 (the ASR put it at 191.14-191.74, inside the
190.29-192.28 measured silence, and the clip rendered for it is idle,
silent footage); "No" is spoken at ~275.9-276.1 right before "quiero"
(the ASR put it at 269.37-270.17, inside the relaxed 268.12-270.97
silence); "pastillas." runs to 269.37 over a measured 268.33-268.96 pause.
"""
from __future__ import annotations

import pytest

from cutsell_worker.audio_silence import (
    AUDIO_SILENCE_EVENT_KIND,
    WORD_RECONCILIATION_RULE_AFTER,
    WORD_RECONCILIATION_RULE_BEFORE,
    WORD_RECONCILIATION_RULE_END,
    WORD_RECONCILIATION_RULE_NO_ROOM,
    reconcile_transcript_words_with_measured_silence,
)
from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes
from cutsell_worker.whole_video_analysis import TemporalEvent

SRC = "src"


def _seg(start, end, words):
    ws = tuple(Word(t, s, e) for t, s, e in words)
    return TranscriptSegment(source_asset_id=SRC, start=start, end=end, text=" ".join(t for t, _, _ in words), words=ws)


def _silence(start, end, confidence=1.0):
    return TemporalEvent(SRC, start, end, AUDIO_SILENCE_EVENT_KIND, confidence, "silencedetect")


# RAW #126 words (timed_asr_replay_evidence.raw_segments) and silences (whole_video_context)
ACNE = [
    _seg(185.24, 189.84, [("Por", 185.24, 185.56), ("temporada", 185.56, 186.2), ("me", 186.2, 186.48), ("salió", 186.48, 186.76),
                          ("un", 186.76, 186.94), ("acné", 186.94, 187.24), ("en", 187.24, 187.46), ("la", 187.46, 187.7),
                          ("espalda", 187.7, 188.1), ("con", 188.1, 188.3), ("la", 188.3, 188.44), ("que", 188.44, 188.54),
                          ("yo", 188.54, 188.92), ("resolvía", 188.92, 189.54), ("con", 189.54, 189.84)]),
    _seg(191.14, 195.12, [("resorcina.", 191.14, 191.74), ("También", 192.44, 192.7), ("me", 192.7, 192.98), ("salían", 192.98, 193.98),
                          ("espinillas.", 193.98, 194.78), ("Era", 195.12, 195.12)]),
    _seg(195.12, 201.26, [("como", 195.12, 195.4), ("un", 195.4, 195.68), ("rush,", 195.68, 196.36), ("una", 196.86, 197.04),
                          ("alergia.", 197.04, 198.12), ("También", 198.88, 199.06), ("me", 199.06, 199.28), ("salían", 199.28, 199.64),
                          ("espinillas", 199.64, 200.28), ("en", 200.28, 200.58), ("esta", 200.58, 200.76), ("parte", 200.76, 201.04),
                          ("de", 201.04, 201.26)]),
]
GASTRITIS = [
    _seg(265.31, 269.37, [("tenía", 265.31, 265.63), ("gastritis", 265.63, 266.15), ("y", 266.15, 266.27), ("me", 266.27, 266.45),
                          ("mandaron", 266.45, 267.01), ("tres", 267.01, 267.37), ("meses", 267.37, 267.69), ("con", 267.69, 267.95),
                          ("pastillas.", 267.95, 269.37)]),
    _seg(269.37, 280.77, [("No", 269.37, 270.17), ("quiero", 276.09, 276.35), ("sonar", 276.35, 276.77), ("a", 276.77, 276.89),
                          ("conspiración", 276.89, 277.87), ("pero", 277.87, 278.47), ("todos", 278.47, 278.83), ("estos", 278.83, 279.11),
                          ("síntomas", 279.11, 279.93), ("comenzaron", 279.93, 280.55), ("a", 280.55, 280.77)]),
    _seg(280.77, 283.67, [("pasarme", 280.77, 281.43), ("después", 281.43, 281.89), ("de", 281.89, 282.23), ("la", 282.23, 282.43),
                          ("vacuna", 282.43, 282.81), ("en", 282.81, 283.01), ("la", 283.01, 283.23), ("pandemia.", 283.23, 283.67)]),
]
SILENCES = [
    _silence(190.12, 192.285, 0.9), _silence(190.292, 192.281, 1.0), _silence(196.011, 196.781, 1.0),
    _silence(197.236, 198.832, 0.9), _silence(197.437, 198.823, 1.0),
    _silence(268.122, 270.968, 0.9), _silence(268.326, 268.964, 1.0), _silence(272.773, 275.944, 0.9),
    _silence(274.736, 275.936, 1.0), _silence(282.932, 287.009, 0.9), _silence(283.218, 286.408, 1.0),
]


def _words(segments):
    return [(w.text, w.start, w.end) for seg in segments for w in seg.words]


def _rows_by_word(rows):
    return {r["word"]: r for r in rows}


def test_resorcina_is_reanchored_before_the_measured_silence_and_joins_the_acne_sentence():
    segments, rows = reconcile_transcript_words_with_measured_silence(ACNE, {SRC: SILENCES})
    row = _rows_by_word(rows)["resorcina."]
    assert row["rule"] == WORD_RECONCILIATION_RULE_BEFORE
    # the primary-floor covering silence (190.292) bounds the word end, never the relaxed floor
    assert row["to_start"] == pytest.approx(189.84) and row["to_end"] == pytest.approx(190.292, abs=1e-3)
    assert row["moved_to_adjacent_segment"] is True
    acne = segments[0]
    assert [w.text for w in acne.words][-3:] == ["resolvía", "con", "resorcina."]
    assert acne.end == pytest.approx(190.292, abs=1e-3)
    assert acne.text.endswith("resolvía con resorcina.")
    # the next segment lost only that word; nothing else moved
    assert [w.text for w in segments[1].words] == ["También", "me", "salían", "espinillas.", "Era"]
    assert _words(segments[1:])[0][1] == 192.44


def test_no_is_reanchored_after_the_last_silence_before_quiero_and_stays_with_its_clause():
    segments, rows = reconcile_transcript_words_with_measured_silence(GASTRITIS, {SRC: SILENCES})
    by = _rows_by_word(rows)
    assert by["No"]["rule"] == WORD_RECONCILIATION_RULE_AFTER
    assert by["No"]["to_start"] == pytest.approx(275.944, abs=1e-3) and by["No"]["to_end"] == pytest.approx(276.09)
    assert by["No"]["moved_to_adjacent_segment"] is False
    clause = segments[1]
    assert [w.text for w in clause.words][:2] == ["No", "quiero"]
    assert clause.start == pytest.approx(275.944, abs=1e-3)
    # the padded sentence-final "pastillas." ends where the measured silence starts
    assert by["pastillas."]["rule"] == WORD_RECONCILIATION_RULE_END
    assert by["pastillas."]["to_end"] == pytest.approx(268.326, abs=1e-3)  # primary floor, not the relaxed 268.122
    assert segments[0].end == pytest.approx(268.326, abs=1e-3)


def test_segmentation_now_yields_a_complete_acne_take_and_a_negated_clause():
    source = SourceAsset(source_asset_id=SRC, project_id="p", user_id="u", original_name="raw.mp4", source_order=0,
                         duration_sec=367.0, uri="s3://b/raw.mp4")
    segments, _ = reconcile_transcript_words_with_measured_silence(ACNE + GASTRITIS, {SRC: SILENCES})
    takes = segment_takes(segments, (source,))
    texts = {t.text for t in takes}
    acne = next(t for t in takes if t.text.startswith("Por temporada"))
    assert acne.text.endswith("con resorcina.") and acne.complete_idea
    assert acne.end == pytest.approx(190.292, abs=1e-3)
    assert not any(t.text.strip() in {"resorcina.", "No"} for t in takes)
    negated = next(t for t in takes if "conspiración" in t.text)
    assert negated.text.startswith("No quiero sonar a conspiración")
    assert negated.start == pytest.approx(275.944, abs=1e-3)


# --- controls ----------------------------------------------------------------

def test_no_measured_silence_returns_the_same_segment_objects():
    segments, rows = reconcile_transcript_words_with_measured_silence(ACNE, {SRC: ()})
    assert rows == () and all(a is b for a, b in zip(segments, ACNE))
    segments2, rows2 = reconcile_transcript_words_with_measured_silence(ACNE, {})
    assert rows2 == () and all(a is b for a, b in zip(segments2, ACNE))


def test_words_outside_any_silence_and_zero_length_words_are_untouched():
    segments, rows = reconcile_transcript_words_with_measured_silence(ACNE, {SRC: SILENCES})
    changed = {r["word"] for r in rows}
    assert changed <= {"resorcina.", "alergia.", "rush,"}
    era = next(w for seg in segments for w in seg.words if w.text == "Era")
    assert (era.start, era.end) == (195.12, 195.12)
    assert _words(segments)[:5] == _words(ACNE)[:5]


def test_low_confidence_silence_is_ignored():
    weak = [_silence(190.12, 192.285, 0.5), _silence(190.292, 192.281, 0.6)]
    segments, rows = reconcile_transcript_words_with_measured_silence(ACNE, {SRC: weak})
    assert rows == () and [w.end for w in segments[0].words][-1] == 189.84


def test_a_word_inside_silence_with_no_adjacent_room_is_reported_and_left_alone():
    # a lone word whose neighbours sit flush against the silence on both sides
    seg = _seg(10.0, 13.0, [("uno", 10.0, 10.5), ("dos", 10.8, 11.4), ("tres", 12.0, 13.0)])
    sil = [_silence(10.5, 12.0, 1.0)]
    segments, rows = reconcile_transcript_words_with_measured_silence([seg], {SRC: sil})
    assert rows[0]["word"] == "dos" and rows[0]["rule"] == WORD_RECONCILIATION_RULE_NO_ROOM
    assert [(w.start, w.end) for w in segments[0].words] == [(10.0, 10.5), (10.8, 11.4), (12.0, 13.0)]


def test_a_spoken_pause_between_two_sentences_does_not_move_either_neighbour():
    seg_a = _seg(0.0, 2.0, [("una", 0.0, 0.5), ("frase.", 0.6, 2.0)])
    seg_b = _seg(4.0, 6.0, [("otra", 4.0, 4.5), ("frase.", 4.6, 6.0)])
    sil = [_silence(2.2, 3.9, 1.0)]
    segments, rows = reconcile_transcript_words_with_measured_silence([seg_a, seg_b], {SRC: sil})
    assert rows == () and segments[0] is seg_a and segments[1] is seg_b


def test_numbers_and_negations_keep_their_order_and_text():
    segments, _ = reconcile_transcript_words_with_measured_silence(GASTRITIS, {SRC: SILENCES})
    words = [w.text for w in segments[1].words]
    assert words == ["No", "quiero", "sonar", "a", "conspiración", "pero", "todos", "estos", "síntomas", "comenzaron", "a"]
    starts = [w.start for seg in segments for w in seg.words]
    assert starts == sorted(starts)


# --- RAW #126 whole-source dry-run cases that shaped the evidence rules -----

def test_a_quiet_trailing_word_is_clamped_by_the_primary_floor_only():
    # "funcionando" 44.16-44.84 still peaks above the -35 dB floor until 44.84
    # (the relaxed -30 dB floor starts at 44.703 inside it); "perfectamente."
    # 45.86-46.42 lies inside the primary silence 45.214-48.884 and is spoken
    # in the room before it.
    segs = [_seg(40.22, 44.84, [("estaba", 43.86, 44.16), ("funcionando", 44.16, 44.84)]),
            _seg(45.86, 53.53, [("perfectamente.", 45.86, 46.42), ("El", 48.97, 49.03), ("año", 49.03, 49.23)])]
    sil = [_silence(44.703, 48.884, 0.9), _silence(45.214, 48.884, 1.0)]
    out, rows = reconcile_transcript_words_with_measured_silence(segs, {SRC: sil})
    by = _rows_by_word(rows)
    assert "funcionando" not in by
    assert by["perfectamente."]["rule"] == WORD_RECONCILIATION_RULE_BEFORE
    assert by["perfectamente."]["to_start"] == pytest.approx(44.84) and by["perfectamente."]["to_end"] == pytest.approx(45.214, abs=1e-3)
    assert [w.text for w in out[0].words] == ["estaba", "funcionando", "perfectamente."]
    assert [w.text for w in out[1].words] == ["El", "año"]


def test_a_sentence_final_word_never_moves_right_into_the_next_sentence():
    # "bien." (73.86-74.42) inside the 73.88-76.92 silence with 0.02 s of room
    # before it and the next sentence 8 s later: reported, left in place.
    segs = [_seg(62.12, 74.42, [("alimentar", 73.38, 73.86), ("bien.", 73.86, 74.42)]),
            _seg(82.82, 90.6, [("Al", 82.82, 83.0), ("terminar", 83.0, 83.4)])]
    sil = [_silence(73.678, 82.732, 0.9), _silence(73.88, 76.922, 1.0), _silence(80.818, 82.73, 1.0)]
    out, rows = reconcile_transcript_words_with_measured_silence(segs, {SRC: sil})
    by = _rows_by_word(rows)
    assert by["bien."]["rule"] == WORD_RECONCILIATION_RULE_NO_ROOM
    assert out[0] is segs[0] and out[1] is segs[1]


def test_a_right_reanchor_never_leaves_the_words_own_asr_segment():
    # a non-terminal word inside silence whose next word belongs to the NEXT
    # ASR segment: the ASR did not group them, so it is not moved there.
    segs = [_seg(10.0, 12.0, [("dice", 10.0, 10.4), ("que", 10.5, 12.0)]),
            _seg(15.0, 16.0, [("mañana", 15.0, 15.5), ("vuelve.", 15.5, 16.0)])]
    sil = [_silence(10.45, 14.95, 1.0)]
    out, rows = reconcile_transcript_words_with_measured_silence(segs, {SRC: sil})
    by = _rows_by_word(rows)
    assert by["que"]["rule"] == WORD_RECONCILIATION_RULE_NO_ROOM
    assert out[0] is segs[0]
