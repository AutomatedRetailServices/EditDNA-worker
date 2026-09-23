"""D-289.11 -- re-opened closing restatement (RAW #124 MP4 review, user-
reported CTA repetition).

RAW #124 rendered the preserved conclusion W ("... Así que cuídate.") and,
after the family-history aside A, the standalone CTA C ("Por eso cuídate,
aliméntate bien, hidrátate y haz ejercicio."). QA `cta_preserved` PASSED on
presence alone. Human Gold keeps the full conclusion and the CTA's advice
but not the re-opened "Por eso cuídate,".

Production owner: `final_boundary_authority._trim_reopened_closings`, a
pre-Freeze, word-start, fail-open trim of the LATER clip's re-opened
closing phrase (the complete earlier delivery is never edited). QA owner:
`validate_video00_regression_qa.find_repeated_closings` / the
`repeated_closing_absent` check kind (presence is a precondition, never
the verdict).

ALL word timings here are SYNTHETIC (evenly spaced over the recorded clip
spans): the RAW #124 package states `word_precise_timing_available:
false`. The clip TEXTS and clip spans are the recorded ones; the timings
prove the mechanism (word-start cut, pause/punctuation break, floors), not
the real render offsets. No production code reads any of these phrases.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks.validate_video00_regression_qa import (
    _repeated_closing_match,
    find_repeated_closings,
    validate,
)
from cutsell_worker.boundary_engine_pass import apply_post_freeze_boundary_pass
from cutsell_worker.contracts import (
    SCHEMA_VERSION,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SemanticRole,
    TranscriptSegment,
    Word,
)
from cutsell_worker.final_boundary_authority import (
    _reopened_closing_match,
    _reopened_closing_refusal,
    _trim_reopened_closings,
    enforce_complete_idea_boundaries,
)
from cutsell_worker.selection_boundary_contract import (
    enforce_selection_contract,
    freeze_selection_contract,
    semantic_token_stream,
)

# Recorded RAW #124 / RAW #122 selected texts (evidence packages), spans in
# source seconds. SYNTHETIC word timings -- see module docstring.
W_TEXT = (
    "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. "
    "Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. "
    "Más bien solo un 5 -10 % son de carácter hereditario. Mayormente son nuestras "
    "elecciones de vida. Así que cuídate."
)
A_TEXT = (
    "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un "
    "carcinoma papilar en la tiroides ni sufre de la tiroides."
)
C_TEXT = "Por eso cuídate, aliméntate bien, hidrátate y haz ejercicio."
R_TEXT = "Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"
T_TEXT = "cánceres son hereditarios."
C122_TEXT = "Por eso cuídate. Aliméntate bien. Hidrátate. Haz ejercicio."


def _synthetic_words(text: str, start: float, end: float, *, pause_after: dict | None = None) -> tuple[Word, ...]:
    tokens = text.split()
    slot = (end - start) / len(tokens)
    out: list[Word] = []
    cursor = start
    for index, token in enumerate(tokens):
        word_end = round(cursor + slot * 0.9, 3)
        out.append(Word(text=token, start=round(cursor, 3), end=word_end))
        cursor += slot
        if pause_after and index in pause_after:
            cursor += pause_after[index]
    return tuple(out)


def _clip(clip_id: str, text: str, start: float, end: float, order: int, *, source: str = "src", words=None) -> DraftClip:
    return DraftClip(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=order,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=words if words is not None else _synthetic_words(text, start, end),
        semantic_role=SemanticRole.OTHER,
    )


def _source_map(clips) -> dict[str, tuple[Word, ...]]:
    out: dict[str, list[Word]] = {}
    for clip in clips:
        out.setdefault(clip.source_asset_id, []).extend(clip.words)
    return {key: tuple(sorted(value, key=lambda w: (w.start, w.end))) for key, value in out.items()}


def _run(clips):
    return _trim_reopened_closings(list(clips), _source_map(clips))


def _trims(rows):
    return [row for row in rows if row["action"] == "trim_reopened_closing_restatement"]


def _refusals(rows):
    return [row for row in rows if row["action"] == "keep_reopened_closing"]


def _raw124_sequence():
    return (
        _clip("W", W_TEXT, 295.52, 313.50, 0),
        _clip("A", A_TEXT, 319.38, 327.44, 1),
        _clip("C", C_TEXT, 356.21, 361.55, 2),
    )


# --- A. reproduction: RAW #124 W -> A -> C and RAW #122 W -> R -> T -> C ---

def test_raw124_reopened_closing_is_trimmed_from_the_cta_at_a_word_start():
    clips = _raw124_sequence()
    out, rows = _run(clips)
    trims = _trims(rows)
    assert len(trims) == 1
    row = trims[0]
    assert (row["left_clip_id"], row["right_clip_id"]) == ("W", "C")
    assert row["repeated_tokens"] == ["cuídate"]
    assert row["removed_leading_tokens"] == ["por", "eso", "cuídate"]
    assert row["intervening_clip_count"] == 1
    assert row["first_remaining_word"] == "aliméntate"
    # The complete conclusion is untouched; the aside is untouched.
    assert out[0] == clips[0]
    assert out[1] == clips[1]
    # The CTA keeps its identity and ALL of its advice; only the re-opened
    # closing is gone, and the new start IS the first remaining word's start.
    cta = out[2]
    assert cta.clip_id == "C"
    assert cta.text == "aliméntate bien, hidrátate y haz ejercicio."
    assert cta.start == pytest.approx(clips[2].words[3].start)
    assert cta.end == clips[2].end
    assert cta.words == clips[2].words[3:]
    assert row["result_start"] == pytest.approx(cta.start, abs=1e-3)


def test_raw122_shape_two_intervening_clips_within_recency_is_trimmed():
    clips = (
        _clip("W", W_TEXT, 295.52, 313.50, 0),
        _clip("R", R_TEXT, 327.78, 334.24, 1),
        _clip("T", T_TEXT, 340.18, 342.58, 2),
        _clip("C", C122_TEXT, 356.21, 361.61, 3),
    )
    out, rows = _run(clips)
    trims = _trims(rows)
    assert len(trims) == 1
    assert trims[0]["intervening_clip_count"] == 2
    assert trims[0]["intervening_sec"] == pytest.approx(8.86, abs=0.01)
    assert out[3].text == "Aliméntate bien. Hidrátate. Haz ejercicio."
    assert out[0].text == W_TEXT and out[1].text == R_TEXT and out[2].text == T_TEXT


def test_adjacent_reopened_closing_is_trimmed():
    clips = (_clip("W", W_TEXT, 295.52, 313.50, 0), _clip("C", C_TEXT, 356.21, 361.55, 1))
    out, rows = _run(clips)
    assert len(_trims(rows)) == 1 and _trims(rows)[0]["intervening_clip_count"] == 0
    assert out[1].text == "aliméntate bien, hidrátate y haz ejercicio."


def test_english_full_closing_restatement_is_trimmed():
    clips = (
        _clip("L", "So take care of yourself.", 1.0, 3.0, 0),
        _clip("R", "So take care of yourself, eat well, hydrate and exercise.", 4.0, 8.0, 1),
    )
    out, rows = _run(clips)
    assert _trims(rows)[0]["repeated_tokens"] == ["so", "take", "care", "of", "yourself"]
    assert out[1].text == "eat well, hydrate and exercise."


def test_longest_repeated_phrase_is_preferred_over_a_shorter_one():
    match = _reopened_closing_match(
        _synthetic_words("Así que cuídate.", 1.0, 2.0),
        _synthetic_words("Así que cuídate, aliméntate bien.", 3.0, 5.0),
    )
    assert match == (0, 3)


# --- B. bounds and refusals (fail open, every refusal recorded) ---

def test_no_repeated_phrase_means_no_change_and_no_row():
    clips = (
        _clip("L", "So take care of yourself.", 1.0, 3.0, 0),
        _clip("R", "Eat well, hydrate and exercise.", 4.0, 8.0, 1),
    )
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_number_in_the_repeated_phrase_refuses_the_trim():
    clips = (
        _clip("L", "Son solo 5 por ciento.", 1.0, 3.0, 0),
        _clip("R", "Solo 5 por ciento, y el resto son elecciones de vida.", 4.0, 8.0, 1),
    )
    out, rows = _run(clips)
    assert _refusals(rows)[0]["reason"] == "removed_prefix_carries_number"
    assert tuple(out) == clips


def test_negation_in_the_repeated_phrase_refuses_the_trim():
    clips = (
        _clip("L", "Así que no te preocupes.", 1.0, 3.0, 0),
        _clip("R", "No te preocupes, aliméntate bien y haz ejercicio.", 4.0, 8.0, 1),
    )
    out, rows = _run(clips)
    assert _refusals(rows)[0]["reason"] == "removed_prefix_carries_negation"
    assert tuple(out) == clips


def test_distinct_addition_marker_in_the_removed_prefix_refuses_the_trim():
    words = _synthetic_words("Otro síntoma, aliméntate bien y haz ejercicio.", 4.0, 8.0)
    assert _reopened_closing_refusal(words, 0, 2) == "removed_prefix_carries_distinct_addition_marker"


def test_topic_re_mention_without_a_phrase_break_is_not_a_reopened_closing():
    clips = (
        _clip("L", "Tienes que hacer ejercicio.", 1.0, 3.0, 0),
        _clip("R", "Ejercicio es lo más importante para tu salud.", 4.0, 8.0, 1),
    )
    out, rows = _run(clips)
    assert _refusals(rows)[0]["reason"] == "repeated_closing_not_a_separate_phrase"
    assert tuple(out) == clips


def test_determiner_plus_noun_re_mention_is_never_matched():
    # RAW #122 rows 9/10: "... que se mandó a biopsia." -> "La biopsia confirmó ..."
    clips = (
        _clip("L", "Apareció un nódulo sospechoso que se mandó a biopsia.", 1.0, 4.0, 0),
        _clip("R", "La biopsia confirmó que era un cáncer papilar de tiroides.", 5.0, 9.0, 1),
    )
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_measured_pause_counts_as_a_phrase_break_when_asr_punctuation_is_missing():
    left = _clip("L", "Así que cuídate.", 1.0, 2.5, 0)
    right_words = _synthetic_words("Por eso cuídate aliméntate bien hidrátate y haz ejercicio", 4.0, 8.0, pause_after={2: 0.4})
    right = _clip("R", "Por eso cuídate aliméntate bien hidrátate y haz ejercicio", 4.0, right_words[-1].end, 1, words=right_words)
    out, rows = _run((left, right))
    assert len(_trims(rows)) == 1
    assert out[1].text == "aliméntate bien hidrátate y haz ejercicio"


def test_remaining_delivery_below_content_floor_refuses():
    clips = (_clip("L", "Así que cuídate.", 1.0, 3.0, 0), _clip("R", "Por eso cuídate, mucho.", 4.0, 6.0, 1))
    out, rows = _run(clips)
    assert _refusals(rows)[0]["reason"] == "remaining_delivery_below_content_floor"
    assert tuple(out) == clips


def test_remaining_delivery_opening_on_a_dangling_word_refuses():
    clips = (_clip("L", "Así que cuídate.", 1.0, 3.0, 0), _clip("R", "Por eso cuídate, y aliméntate bien.", 4.0, 8.0, 1))
    out, rows = _run(clips)
    assert _refusals(rows)[0]["reason"] == "remaining_delivery_would_open_on_dangling_word"
    assert tuple(out) == clips


def test_earlier_clip_without_terminal_punctuation_is_not_a_closing():
    clips = (_clip("L", "Así que cuídate", 1.0, 3.0, 0), _clip("R", C_TEXT, 4.0, 9.0, 1))
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_recency_bound_intervening_output_over_ten_seconds_is_left_alone():
    clips = (
        _clip("W", W_TEXT, 295.52, 313.50, 0),
        _clip("A", A_TEXT, 319.38, 331.38, 1),  # 12.0 s of intervening output
        _clip("C", C_TEXT, 356.21, 361.55, 2),
    )
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_three_intervening_clips_is_beyond_the_lookback():
    clips = (
        _clip("W", "Así que cuídate.", 1.0, 3.0, 0),
        _clip("X", "uno.", 4.0, 5.0, 1),
        _clip("Y", "dos.", 6.0, 7.0, 2),
        _clip("Z", "tres.", 8.0, 9.0, 3),
        _clip("C", C_TEXT, 10.0, 15.0, 4),
    )
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_different_sources_are_never_paired():
    clips = (_clip("L", "Así que cuídate.", 1.0, 3.0, 0, source="a"), _clip("R", C_TEXT, 4.0, 9.0, 1, source="b"))
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_source_order_violation_is_never_paired():
    clips = (_clip("L", "Así que cuídate.", 10.0, 12.0, 0), _clip("R", C_TEXT, 4.0, 9.0, 1))
    out, rows = _run(clips)
    assert rows == [] and tuple(out) == clips


def test_one_trim_per_later_clip_and_the_earlier_clip_is_never_edited():
    clips = (
        _clip("L1", "Así que cuídate.", 1.0, 3.0, 0),
        _clip("L2", "Así que cuídate.", 4.0, 6.0, 1),
        _clip("R", C_TEXT, 7.0, 12.0, 2),
    )
    out, rows = _run(clips)
    assert len(_trims(rows)) == 1 and _trims(rows)[0]["left_clip_id"] == "L2"
    assert out[0] == clips[0] and out[1] == clips[1]


# --- C. end to end through the pre-Freeze owner, then Freeze + Boundary contract ---

class _FakeASR:
    def __init__(self, words_by_source):
        self._words = words_by_source

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        words = self._words[source_asset_id]
        if not words:
            return ()
        return (TranscriptSegment(source_asset_id=source_asset_id, start=words[0].start, end=words[-1].end,
                                  text=" ".join(w.text for w in words), words=tuple(words)),)


def _result(selected, diagnostics=None):
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected), alternates=(), discarded=(), diagnostics=dict(diagnostics or {}),
    )
    return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p", state=JobState.DRAFT_READY, draft=draft, stage_status={})


def test_enforce_complete_idea_boundaries_trims_before_freeze_and_boundary_keeps_the_stream():
    clips = _raw124_sequence()
    # Full source transcript = the selected words plus unselected filler between
    # them (a real source has speech the selection dropped).
    filler = _synthetic_words("material descartado entre tomas.", 330.0, 340.0)
    source_words = tuple(sorted((*_source_map(clips)["src"], *filler), key=lambda w: (w.start, w.end)))
    result = enforce_complete_idea_boundaries(_result(clips), {"src": "/nonexistent.mp4"}, asr_provider=_FakeASR({"src": source_words}))

    selected = result.draft.selected
    assert [c.clip_id for c in selected] == ["W", "A", "C"]
    assert selected[2].text == "aliméntate bien, hidrátate y haz ejercicio."
    assert selected[0].text == W_TEXT and selected[1].text == A_TEXT
    diag = result.draft.diagnostics
    assert diag["final_boundary_reopened_closing_trim_count"] == 1
    assert diag["final_boundary_reopened_closing_refusal_count"] == 0
    assert "re-opened closing restatement trim (D-289.11" in diag["final_boundary_authority_rule"]
    trim_rows = [r for r in diag["final_boundary_authority"] if r.get("action") == "trim_reopened_closing_restatement"]
    assert trim_rows and trim_rows[0]["right_clip_id"] == "C"

    # Freeze AFTER the trim: the frozen stream is the trimmed stream, and the
    # post-Freeze BoundaryEngine pass + contract verification hold on it.
    frozen = freeze_selection_contract(result.draft)
    # (the contract canonicalizes accents; the stream is the trimmed one)
    assert semantic_token_stream(frozen.selected)[-6:] == ("alimentate", "bien", "hidratate", "y", "haz", "ejercicio")
    assert "cuidate" not in semantic_token_stream(frozen.selected)[-7:]
    after_boundary = apply_post_freeze_boundary_pass(replace(result, draft=frozen))
    verified = enforce_selection_contract(after_boundary.draft)
    assert verified.diagnostics["selection_boundary_contract"]["status"] == "verified"


def test_freeze_before_the_trim_would_reject_it_so_the_rule_must_stay_pre_freeze():
    clips = _raw124_sequence()
    frozen = freeze_selection_contract(_result(clips).draft)
    trimmed, rows = _run(clips)
    assert _trims(rows)
    with pytest.raises(RuntimeError, match="Boundary changed frozen Selection semantic content"):
        enforce_selection_contract(replace(frozen, selected=tuple(trimmed)))


def test_no_selected_words_fails_open_end_to_end():
    bare = tuple(replace(c, words=()) for c in _raw124_sequence())
    result = enforce_complete_idea_boundaries(_result(bare), {"src": "/nonexistent.mp4"}, asr_provider=_FakeASR({"src": ()}))
    assert [c.text for c in result.draft.selected] == [W_TEXT, A_TEXT, C_TEXT]
    assert result.draft.diagnostics["final_boundary_reopened_closing_trim_count"] == 0


# --- D. QA: presence is not uniqueness ---

def _qa_rows(texts):
    return [(f"clip_{i}", t) for i, t in enumerate(texts)]


def test_qa_detector_flags_raw124_shape_and_raw122_shape():
    found = find_repeated_closings(_qa_rows([W_TEXT, A_TEXT, C_TEXT]))
    assert len(found) == 1
    assert found[0]["repeated_tokens"] == ["cuídate"] and found[0]["intervening_rows"] == 1
    assert found[0]["leading_connective_count"] == 2
    found122 = find_repeated_closings(_qa_rows([W_TEXT, R_TEXT, T_TEXT, C122_TEXT]))
    assert len(found122) == 1 and found122[0]["intervening_rows"] == 2


def test_qa_detector_ignores_determiner_re_mention_and_non_terminal_closings():
    assert find_repeated_closings(_qa_rows([
        "Apareció un nódulo sospechoso que se mandó a biopsia.",
        "La biopsia confirmó que era un cáncer papilar de tiroides.",
    ])) == []
    assert _repeated_closing_match("Así que cuídate", C_TEXT) is None
    assert find_repeated_closings(_qa_rows([W_TEXT, A_TEXT, "aliméntate bien, hidrátate y haz ejercicio."])) == []


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _manifest():
    return {
        "schema_version": "qa.v1",
        "baseline_run_id": "synthetic",
        "checks": [
            {"id": "cta_preserved", "kind": "required_exact", "text": C_TEXT},
            {"id": "cta_unique_closing", "kind": "repeated_closing_absent", "text": C_TEXT},
        ],
    }


def test_qa_presence_passes_while_repeated_closing_fails_on_the_raw124_shape(tmp_path):
    result = {"selected": [{"clip_id": f"c{i}", "text": t} for i, t in enumerate([W_TEXT, A_TEXT, C_TEXT])]}
    ok, report = validate(_write(tmp_path, "r.json", result), _write(tmp_path, "m.json", _manifest()))
    assert not ok
    assert "cta_preserved" in report["passed_checks"]
    failed = {row["id"]: row for row in report["failed_checks"]}
    assert failed["cta_unique_closing"]["reason"] == "repeated_closing_reopens_required_segment"
    assert failed["cta_unique_closing"]["detail"]["repeated_tokens"] == ["cuídate"]
    assert failed["cta_unique_closing"]["detail"]["earlier_clip_id"] == "c0"
    scan = [w for w in report["warnings"] if w["kind"] == "repeated_closing_detected"]
    assert len(scan) == 1 and scan[0]["later_clip_id"] == "c2"


def test_qa_repeated_closing_passes_once_the_reopened_closing_is_trimmed(tmp_path):
    trimmed = "aliméntate bien, hidrátate y haz ejercicio."
    result = {"selected": [{"clip_id": f"c{i}", "text": t} for i, t in enumerate([W_TEXT, A_TEXT, trimmed])]}
    ok, report = validate(_write(tmp_path, "r.json", result), _write(tmp_path, "m.json", _manifest()))
    assert ok, report
    assert {"cta_preserved", "cta_unique_closing"} <= set(report["passed_checks"])
    assert [w for w in report["warnings"] if w["kind"] == "repeated_closing_detected"] == []


def test_qa_repeated_closing_check_fails_when_the_segment_is_missing_entirely(tmp_path):
    result = {"selected": [{"clip_id": "c0", "text": W_TEXT}, {"clip_id": "c1", "text": A_TEXT}]}
    ok, report = validate(_write(tmp_path, "r.json", result), _write(tmp_path, "m.json", _manifest()))
    assert not ok
    reasons = {row["id"]: row["reason"] for row in report["failed_checks"]}
    assert reasons["cta_unique_closing"] == "missing_required_segment"


def test_baseline_manifest_is_untouched_by_this_decision():
    manifest = json.load(open("benchmarks/video00_regression_qa.json", encoding="utf-8"))
    kinds = {c["kind"] for c in manifest["checks"]}
    assert "repeated_closing_absent" not in kinds
    assert any(c["id"] == "cta_preserved" and c["kind"] == "required_exact" for c in manifest["checks"])
