"""D-097 Priority C/E -- one post-Freeze BoundaryEngine pass, physical
ownership contract, source-vs-render dead-air reconciliation.

D-096 root cause #4: the D-095.2 interior dead-air trimmer ran inside
`build_flow_b_draft`, BEFORE the authority decided the final KEEP set, so a
realization restored/reshaped later was never trimmed (run 34008386434: the
2.32 s pause that invalidated the render sat inside a clip the trimmer had
never seen -- no split, no rejection trace). The same functions now run
once, after Selection Freeze, on the frozen final keep set; entries/exits
gain an audio-evidenced owner; the renderer records its trailing trims; and
every post-render silence finding is reconciled back to the source
measurement. Generic fixtures only.
"""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker import audio_silence
from cutsell_worker.boundary_engine_pass import (
    BOUNDARY_REASON_AUDIO_ENTRY,
    BOUNDARY_REASON_AUDIO_EXIT,
    PHYSICAL_OWNERSHIP_CONTRACT,
    apply_post_freeze_boundary_pass,
    reconcile_silence_findings,
    tighten_selected_audio_edges,
)
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.post_render_watch_listen_qc import LINGERING_ACCIDENTAL_SILENCE, PostRenderFinding
from cutsell_worker.post_selection_interior_gap_trim import (
    AUDIO_SILENCE_EVENT_KIND, BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE,
)
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.selection_boundary_contract import enforce_selection_contract, freeze_selection_contract


def _words(text, start, end):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step + 0.05, start + (i + 1) * step - 0.05) for i, t in enumerate(tokens))


def _clip(clip_id, start, end, text, *, words=None):
    words = _words(text, start, end) if words is None else words
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text,
        caption_text=text, words=words, semantic_role=SemanticRole.STORY, selected=True,
    )


def _diag(silences, extra=None):
    return {
        "whole_video_context": {"sources": [{
            "source_asset_id": "src",
            "events": [
                {"kind": AUDIO_SILENCE_EVENT_KIND, "start": s, "end": e, "confidence": c}
                for s, e, c in silences
            ],
        }]},
        **(extra or {}),
    }


def _draft(clips, diagnostics):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics,
    )


def _result(draft):
    return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})


# --- audio-evidenced entry / exit -----------------------------------------------

def test_leading_silence_tightens_the_entry_but_never_past_the_first_word():
    clip = _clip("c", 10.0, 16.0, "uno dos tres cuatro", words=_words("uno dos tres cuatro", 11.2, 16.0))
    out, audit = tighten_selected_audio_edges((clip,), _diag([(9.4, 11.0, 1.0)]))
    assert out[0].start == pytest.approx(10.9)  # silence end minus pad
    assert out[0].end == 16.0 and out[0].boundary_reason == BOUNDARY_REASON_AUDIO_ENTRY
    assert audit[0]["actions"][0]["action"] == BOUNDARY_REASON_AUDIO_ENTRY
    # the first word is the floor
    late = _clip("c", 10.0, 16.0, "uno dos", words=_words("uno dos", 10.5, 16.0))
    out, _ = tighten_selected_audio_edges((late,), _diag([(9.4, 11.0, 1.0)]))
    assert out[0].start == pytest.approx(late.words[0].start)


def test_trailing_silence_tightens_the_exit_but_never_before_the_last_word():
    clip = _clip("c", 10.0, 16.0, "uno dos tres", words=_words("uno dos tres", 10.0, 14.0))
    out, audit = tighten_selected_audio_edges((clip,), _diag([(14.3, 16.1, 0.9)]))
    assert out[0].end == pytest.approx(14.4)
    assert out[0].boundary_reason == BOUNDARY_REASON_AUDIO_EXIT
    assert audit[0]["actions"][0]["silence_confidence"] == 0.9  # relaxed-floor evidence is still evidence
    assert [w.text for w in out[0].words] == ["uno", "dos", "tres"]


def test_edge_tightening_ignores_interior_and_immaterial_silences():
    clip = _clip("c", 10.0, 16.0, "uno dos tres cuatro cinco")
    out, audit = tighten_selected_audio_edges((clip,), _diag([(12.0, 13.5, 1.0), (15.9, 16.4, 1.0)]))
    assert out == (clip,) and audit == ()


# --- the pass on the frozen keep set ---------------------------------------------

def test_post_freeze_pass_splits_interior_dead_air_on_the_final_keep_set_and_keeps_the_contract():
    words = (Word("por", 0.2, 1.9), Word("temporada,", 2.0, 3.8), Word("me", 3.9, 5.7), Word("salía", 5.8, 7.6), Word("acné", 7.7, 9.4))
    clip = _clip("kept", 0.0, 9.5, " ".join(w.text for w in words), words=words)
    draft = freeze_selection_contract(_draft([clip], _diag([(2.0, 4.36, 1.0)])))
    result = apply_post_freeze_boundary_pass(_result(draft))
    pieces = result.draft.selected
    assert len(pieces) == 2
    assert {p.boundary_reason for p in pieces} == {BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE}
    assert all(p.parent_semantic_clip_id == "kept" for p in pieces)
    diag = result.draft.diagnostics
    assert diag["boundary_engine_pass"]["stage"] == "post_freeze"
    assert diag["boundary_engine_pass"]["interior_split_count"] == 1
    assert diag["post_selection_interior_gap_trim"][0]["evidence_mode"] == "long_audio_silence"
    assert [row["concern"] for row in diag["boundary_engine_pass"]["ownership_contract"]] == [row["concern"] for row in PHYSICAL_OWNERSHIP_CONTRACT]
    # Boundary never changed the frozen token stream
    enforce_selection_contract(result.draft)


def test_post_freeze_pass_is_a_no_op_without_evidence():
    clip = _clip("kept", 0.0, 5.0, "uno dos tres cuatro")
    result = apply_post_freeze_boundary_pass(_result(_draft([clip], _diag([]))))
    assert result.draft.selected == (clip,)
    assert result.draft.diagnostics["boundary_engine_pass"]["interior_split_count"] == 0


def test_draft_wrappers_skip_when_the_post_freeze_pass_owns_boundary(monkeypatch):
    from cutsell_worker import post_selection_edge_only_boundary as edge, post_selection_interior_gap_trim as interior

    calls = []
    monkeypatch.setattr(interior, "split_selected_interior_performance_gaps", lambda *a, **k: (calls.append("interior") or ((), ())))
    monkeypatch.setattr(edge, "trim_locked_selection_edges", lambda *a, **k: (calls.append("edge") or ((), ())))
    from cutsell_worker import pipeline
    draft = _draft([_clip("c", 0.0, 3.0, "uno dos")], _diag([]))
    original = pipeline.build_flow_b_draft
    # the installed chain: walk down to a fake original via the wrappers' own closures
    sentinel = _result(draft)
    seen = []

    def fake_original(*args, **kwargs):
        seen.append(kwargs.get("boundary_owner"))
        return sentinel

    monkeypatch.setattr(pipeline, "build_flow_b_draft", fake_original)
    interior.install_post_selection_interior_gap_trim()
    edge.install_post_selection_edge_only_boundary()
    wrapped = pipeline.build_flow_b_draft
    monkeypatch.setattr(pipeline, "build_flow_b_draft", original)
    assert wrapped(object(), (), boundary_owner="post_freeze") is sentinel
    assert calls == []
    wrapped(object(), (), boundary_owner="pre_freeze")
    assert calls == ["interior", "edge"]  # inner wrapper (interior) runs first on the way out
    assert seen == ["post_freeze", "pre_freeze"]  # the kwarg reaches the original untouched


def test_universal_path_runs_the_pass_after_freeze_and_verifies_the_contract(monkeypatch):
    words = (Word("por", 0.2, 1.9), Word("temporada,", 2.0, 3.8), Word("me", 3.9, 5.7), Word("salía", 5.8, 7.6), Word("acné", 7.7, 9.4))
    kept = _clip("kept", 0.0, 9.5, " ".join(w.text for w in words), words=words)
    diag = _diag([(2.0, 4.36, 1.0)], {"take_judge_groups": [{"group_id": "g1", "ranked": [{"clip_id": "kept", "score": 0.9, "reason": "x"}]}]})
    captured = {}

    def fake_process(request, local_paths, **kwargs):
        captured.update(kwargs)
        return _result(_draft([kept], diag))

    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)
    result = universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)
    assert captured["boundary_owner"] == "post_freeze"
    assert result.stage_status["boundary_engine_pass"] == "post_freeze_edge_interior_audio_edges_complete"
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
    assert len(result.draft.selected) == 2
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "verified"
    assert result.draft.diagnostics["boundary_engine_pass"]["interior_split_count"] == 1


# --- reconciliation (C-12) ---------------------------------------------------------

def _segment(clip_id, start, end, parent=None):
    return RenderSegment(clip_id=clip_id, source_asset_id="src", source_path="/x.mp4", start=start, end=end, parent_semantic_clip_id=parent)


def test_reconciliation_maps_the_finding_to_source_and_reports_missing_source_measurement():
    draft = _draft([], _diag([], {"boundary_engine_pass": {"stage": "post_freeze"}}))
    segments = (_segment("a", 10.0, 15.0), _segment("b", 100.0, 110.0))
    windows = [(0.0, 5.0), (5.0, 15.0)]
    finding = PostRenderFinding(kind=LINGERING_ACCIDENTAL_SILENCE, start=8.6, end=11.0, detail={}, routes_to="BoundaryEngine")
    rows = reconcile_silence_findings(draft, segments, (finding,), windows)
    assert rows[0]["clip_id"] == "b"
    assert rows[0]["source_start"] == pytest.approx(103.6) and rows[0]["source_end"] == pytest.approx(106.0)
    assert rows[0]["verdict"] == "source_not_measured"


def test_reconciliation_reports_the_trimmer_rejection_when_the_source_did_measure_it():
    diag = _diag([(103.5, 106.1, 1.0)], {
        "boundary_engine_pass": {"stage": "post_freeze"},
        "post_selection_interior_gap_trace": [{"parent_clip_id": "b", "decision": "reject", "reason": "audio_silence_edge_margin", "evidence_mode": "long_audio_silence"}],
    })
    segments = (_segment("b__frag", 100.0, 110.0, parent="b"),)
    finding = PostRenderFinding(kind=LINGERING_ACCIDENTAL_SILENCE, start=3.6, end=6.0, detail={}, routes_to="BoundaryEngine")
    rows = reconcile_silence_findings(_draft([], diag), segments, (finding,), [(0.0, 10.0)])
    assert rows[0]["source_silence_events_overlapping"][0]["start"] == 103.5
    assert rows[0]["verdict"] == "source_measured_trimmer_rejected:audio_silence_edge_margin"


def test_reconciliation_ignores_non_silence_findings():
    finding = PostRenderFinding(kind="FROZEN_OR_REPEATED_FRAME", start=1.0, end=2.0, detail={}, routes_to="BoundaryEngine")
    assert reconcile_silence_findings(_draft([], _diag([])), (_segment("a", 0.0, 5.0),), (finding,), [(0.0, 5.0)]) == ()


# --- source measurement robustness (C-12 mechanism) ----------------------------------

def _synth(tmp_path, *, noise_db):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path / f"tst{abs(int(noise_db))}.wav"
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5:sample_rate=48000",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=1:duration=2.3:sample_rate=48000:seed=7",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5:sample_rate=48000",
        "-filter_complex", f"[1:a]volume={noise_db}dB[n];[0:a][n][2:a]concat=n=3:v=0:a=1[out]", "-map", "[out]", str(path),
    ])
    return str(path)


def test_merge_silence_runs_joins_fragments_of_one_pause():
    merged = audio_silence.merge_silence_runs([(1.5, 2.0), (2.2, 2.9), (3.1, 3.8), (6.0, 7.0)])
    assert merged == ((1.5, 3.8), (6.0, 7.0))


def test_near_floor_room_tone_is_measured_as_one_pause_and_the_relaxed_floor_backs_it_up(tmp_path):
    path = _synth(tmp_path, noise_db=-33)
    primary = audio_silence.detect_audio_silence_intervals(path)
    events = audio_silence.audio_silence_events({"src": path})["src"]
    # whatever the primary floor fragments, the published evidence covers the
    # full 1.5-3.8 s pause at one of the two floors
    covering = [e for e in events if e.start <= 1.7 and e.end >= 3.6]
    assert covering, (primary, [(e.start, e.end, e.confidence) for e in events])
    assert {e.confidence for e in events} <= {1.0, audio_silence.RELAXED_CONFIDENCE}


def test_a_clear_pause_is_still_one_primary_interval(tmp_path):
    path = _synth(tmp_path, noise_db=-60)
    intervals = audio_silence.detect_audio_silence_intervals(path)
    assert len(intervals) == 1
    assert intervals[0][0] == pytest.approx(1.5, abs=0.1) and intervals[0][1] == pytest.approx(3.8, abs=0.1)
    events = audio_silence.audio_silence_events({"src": path})["src"]
    assert [e.confidence for e in events] == [1.0]  # relaxed pass adds nothing already covered


def test_renderer_records_its_trailing_trims(monkeypatch, tmp_path):
    from cutsell_worker import render
    seg = RenderSegment(clip_id="a", source_asset_id="src", source_path="/x.mp4", start=0.0, end=5.0)
    monkeypatch.setattr(render, "tighten_trailing_silence", lambda s: s.__class__(**{**s.__dict__, "end": 4.6}))
    monkeypatch.setattr(render, "_run", lambda command: None)
    monkeypatch.setattr(render, "_concat_render_command", lambda *a, **k: ["true"])
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    report: list = []
    render.render_preview((seg,), str(out), trim_report=report)
    assert report == [{
        "clip_id": "a", "render_fragment_id": None, "original_end": 5.0, "tightened_end": 4.6,
        "trim_sec": 0.4, "owner": "render.tighten_trailing_silence",
    }]


def test_no_video00_constants_in_the_new_boundary_code():
    from pathlib import Path
    for rel in ("cutsell_worker/boundary_engine_pass.py", "cutsell_worker/audio_silence.py", "cutsell_worker/polarity_safety.py"):
        src = Path(rel).read_text(encoding="utf-8")
        for needle in ("VIDEO-2026", "D40F1D43", "5E01F214", "resorcina", "espinillas", "conspiraci"):
            assert needle not in src, (rel, needle)
