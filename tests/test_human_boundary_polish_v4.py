from dataclasses import replace
from types import SimpleNamespace

import cutsell_worker.human_boundary_polish_v4 as v4
from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
    Word,
)


def _clip(clip_id, start, end, text, words, selected=True):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=tuple(words),
        selected=selected,
    )


def _result(selected, alternates=(), diagnostics=None):
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=(),
        diagnostics=diagnostics or {},
    )
    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={},
    )


def test_promotes_nearby_more_complete_later_retry(monkeypatch):
    monkeypatch.setattr(v4, "polish_human_boundaries_v3", lambda result, _paths: result)
    bad_words = (
        Word("cada", 1.0, 1.3), Word("año", 1.35, 1.6), Word("hacía", 1.65, 2.0),
        Word("dos", 2.05, 2.3), Word("estudios", 2.35, 2.8),
    )
    good_words = (
        Word("cada", 4.0, 4.3), Word("año", 4.35, 4.6), Word("hacía", 4.65, 5.0),
        Word("dos", 5.05, 5.3), Word("estudios", 5.35, 5.8),
        Word("y", 5.85, 5.95), Word("todo", 6.0, 6.3), Word("salía", 6.35, 6.7),
        Word("perfectamente", 6.75, 7.4),
    )
    bad = _clip("bad", 1.0, 2.8, "cada año hacía dos estudios", bad_words)
    good = _clip("good", 4.0, 7.4, "cada año hacía dos estudios y todo salía perfectamente", good_words, selected=False)
    result = _result([bad], [good])
    out = v4.polish_human_boundaries_v4(result, {})
    assert [clip.clip_id for clip in out.draft.selected] == ["good"]
    assert "perfectamente" in out.draft.selected[0].text
    assert any(clip.clip_id == "bad" for clip in out.draft.alternates)


def test_removes_word_gap_only_with_visual_reset(monkeypatch):
    monkeypatch.setattr(v4, "polish_human_boundaries_v3", lambda result, _paths: result)
    words = (
        Word("antes", 1.0, 1.4),
        Word("continúa", 3.3, 3.8),
        Word("bien", 3.9, 4.2),
    )
    clip = _clip("c", 1.0, 4.2, "antes continúa bien", words)
    event = SimpleNamespace(
        start=1.5,
        end=3.1,
        kind="facial_expression_shift_candidate",
        confidence=0.96,
    )
    timeline = SimpleNamespace(source_asset_id="src", events=(event,))
    monkeypatch.setattr(v4, "_timeline_proxies", lambda _result: {"src": timeline})
    out = v4.polish_human_boundaries_v4(_result([clip]), {})
    assert len(out.draft.selected) == 2
    assert out.draft.selected[0].end == 1.4
    assert out.draft.selected[1].start == 3.3
    rows = out.draft.diagnostics["human_boundary_polish"]
    assert any(row.get("action") == "remove_visual_reset_word_gap" for row in rows)


def test_does_not_remove_long_word_gap_without_visual_reset(monkeypatch):
    monkeypatch.setattr(v4, "polish_human_boundaries_v3", lambda result, _paths: result)
    words = (Word("antes", 1.0, 1.4), Word("continúa", 3.3, 3.8))
    clip = _clip("c", 1.0, 3.8, "antes continúa", words)
    timeline = SimpleNamespace(source_asset_id="src", events=())
    monkeypatch.setattr(v4, "_timeline_proxies", lambda _result: {"src": timeline})
    out = v4.polish_human_boundaries_v4(_result([clip]), {})
    assert len(out.draft.selected) == 1
    assert out.draft.selected[0].start == 1.0
    assert out.draft.selected[0].end == 3.8
