"""D-095.2: objective audio dead air inside a kept take is removed before Freeze.

Run 33995806350 (Video00): a 12.5 s kept candidate carried a 2.36 s interior
silence (-35 dB) that no ASR word gap revealed (word timestamps stretched over
the silence); the render's LINGERING_ACCIDENTAL_SILENCE finding was mid-segment
and unrepairable, so the whole candidate was invalidated. The same measurement
now feeds the existing interior-gap trimmer as evidence, and the final Boundary
authority preserves the removed gap for fragment-provenance siblings.
No Video00 timestamps or texts are used here.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from cutsell_worker.audio_silence import (
    AUDIO_SILENCE_EVENT_KIND,
    audio_silence_events,
    detect_audio_silence_intervals,
    merge_audio_silence_into_context,
)
from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.post_selection_interior_gap_trim import (
    BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE,
    LONG_AUDIO_SILENCE_SEC,
    split_selected_interior_performance_gaps,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _stretched_clip(start=0.0, end=12.5):
    # Seven words whose timestamps are spread over 12.5 s: no word gap >= 0.30 s
    # anywhere, exactly the shape ASR produced on the hesitant take.
    words = (
        Word("por", 0.20, 1.90),
        Word("temporada,", 2.00, 3.80),
        Word("me", 3.90, 5.70),
        Word("salía", 5.80, 7.60),
        Word("acné", 7.70, 9.40),
        Word("en", 9.50, 10.90),
        Word("la", 11.00, 12.30),
    )
    return DraftClip(
        clip_id="clip-hesitant",
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=" ".join(w.text for w in words),
        caption_text=" ".join(w.text for w in words),
        words=words,
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _diag(silences):
    return {
        "whole_video_context": {
            "sources": [{
                "source_asset_id": "src",
                "events": [
                    {"kind": AUDIO_SILENCE_EVENT_KIND, "start": s, "end": e, "confidence": 1.0}
                    for s, e in silences
                ],
            }]
        }
    }


def test_word_gap_evidence_alone_cannot_see_the_stretched_silence():
    pieces, audit = split_selected_interior_performance_gaps((_stretched_clip(),), _diag([]))
    assert len(pieces) == 1 and audit == ()


def test_long_audio_silence_inside_a_kept_take_is_cut_with_pause_edges():
    clip = _stretched_clip()
    pieces, audit = split_selected_interior_performance_gaps((clip,), _diag([(2.0, 4.36)]))
    assert len(pieces) == 2
    left, right = pieces
    assert left.end == pytest.approx(2.12) and right.start == pytest.approx(4.24)
    assert left.boundary_reason == BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE
    assert right.boundary_reason == BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE
    assert left.parent_semantic_clip_id == "clip-hesitant" == right.parent_semantic_clip_id
    assert left.clip_id != right.clip_id and left.render_fragment_id == left.clip_id
    # words partitioned by midpoint, timings clamped inside each piece
    assert [w.text for w in left.words] == ["por", "temporada,"]
    assert all(float(w.end) <= left.end + 1e-9 for w in left.words)
    assert all(float(w.start) >= right.start - 1e-9 for w in right.words)
    assert " ".join(w.text for w in left.words) == left.text
    assert audit[0]["decision"] == "split" and audit[0]["evidence_mode"] == "long_audio_silence"
    assert audit[0]["removed_gap_sec"] == pytest.approx(2.12)
    assert (left.fragment_index, left.fragment_count, right.fragment_index) == (0, 2, 1)


def test_audio_silence_below_the_qc_threshold_is_not_cut():
    pieces, audit = split_selected_interior_performance_gaps((_stretched_clip(),), _diag([(2.0, 2.0 + LONG_AUDIO_SILENCE_SEC - 0.05)]))
    assert len(pieces) == 1 and audit == ()


def test_audio_silence_touching_the_clip_edges_is_left_to_boundary_edges():
    pieces, audit = split_selected_interior_performance_gaps(
        (_stretched_clip(),), _diag([(0.1, 1.9), (10.9, 12.45)]), include_rejected_diagnostics=True,
    )
    assert len(pieces) == 1
    reasons = {row["reason"] for row in audit if row.get("evidence_mode") == "long_audio_silence"}
    assert reasons == {"audio_silence_edge_margin"}


def test_audio_silence_outside_the_clip_is_ignored():
    pieces, audit = split_selected_interior_performance_gaps((_stretched_clip(),), _diag([(20.0, 25.0)]))
    assert len(pieces) == 1 and audit == ()


def test_two_long_silences_are_both_removed_within_the_split_budget():
    clip = _stretched_clip()
    pieces, audit = split_selected_interior_performance_gaps((clip,), _diag([(2.0, 3.5), (7.9, 9.3)]))
    assert len(pieces) == 3
    assert [row["evidence_mode"] for row in audit] == ["long_audio_silence", "long_audio_silence"]
    assert [p.fragment_index for p in pieces] == [0, 1, 2]
    assert all(p.parent_semantic_clip_id == "clip-hesitant" for p in pieces)


def test_audio_silence_never_changes_which_words_survive():
    clip = _stretched_clip()
    pieces, _ = split_selected_interior_performance_gaps((clip,), _diag([(2.0, 4.36)]))
    assert [w.text for p in pieces for w in p.words] == [w.text for w in clip.words]


def test_visual_reset_path_still_applies_when_no_audio_silence_qualifies():
    from tests.test_cutsell_post_selection_interior_gap_trim import _clip, _diagnostics
    pieces, audit = split_selected_interior_performance_gaps((_clip(),), _diagnostics(with_reset=True))
    assert len(pieces) == 2 and audit[0]["evidence_mode"] == "multimodal_break"


def test_final_boundary_authority_preserves_the_removed_gap_for_fragment_siblings():
    from cutsell_worker.final_boundary_authority import _reconcile_same_source_overlaps
    clip = _stretched_clip()
    pieces, _ = split_selected_interior_performance_gaps((clip,), _diag([(2.0, 4.36)]))
    left, right = pieces
    # Envelope expansion re-extended both pieces over the removed silence
    # (a source word straddles the cut); the authority must restore the gap.
    source_words = clip.words
    expanded = [
        left.__class__(**{**left.__dict__, "end": 4.0}),
        right.__class__(**{**right.__dict__, "start": 2.5}),
    ]
    fixed, rows = _reconcile_same_source_overlaps((left, right), expanded, {"src": source_words})
    assert rows[0]["action"] == "preserve_polished_interior_gap"
    assert fixed[0].end == pytest.approx(left.end) and fixed[1].start == pytest.approx(right.start)


def _synth_tone_silence_tone(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path / "tst.wav"
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5:sample_rate=8000",
        "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=2.0",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5:sample_rate=8000",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]", "-map", "[out]", str(path),
    ])
    return str(path)


def test_detect_audio_silence_intervals_measures_the_real_dead_air(tmp_path):
    path = _synth_tone_silence_tone(tmp_path)
    intervals = detect_audio_silence_intervals(path)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start == pytest.approx(1.5, abs=0.1) and end == pytest.approx(3.5, abs=0.1)


def test_detect_audio_silence_intervals_never_raises_on_bad_input(tmp_path):
    assert detect_audio_silence_intervals(str(tmp_path / "missing.wav")) == ()
    assert detect_audio_silence_intervals("x", ffmpeg_bin="/nonexistent/ffmpeg") == ()


def test_audio_silence_events_merge_into_the_whole_video_context(tmp_path):
    path = _synth_tone_silence_tone(tmp_path)
    events = audio_silence_events({"src": path})
    assert [e.kind for e in events["src"]] == [AUDIO_SILENCE_EVENT_KIND]
    context = WholeVideoContext(
        sources=(SourceVideoContext("src", "", "", "", events=(TemporalEvent("src", 0.5, 0.6, "hand_motion_reset_candidate", 0.9, ""),)),),
        status=ProviderStatus("whole_video", True, True, "applied", None),
    )
    merged = merge_audio_silence_into_context(context, events)
    kinds = [e.kind for e in merged.sources[0].events]
    assert kinds == ["hand_motion_reset_candidate", AUDIO_SILENCE_EVENT_KIND]
    # idempotent
    again = merge_audio_silence_into_context(merged, events)
    assert len(again.sources[0].events) == 2


def test_no_video00_constants_in_the_new_engine_code():
    from pathlib import Path
    for rel in ("cutsell_worker/audio_silence.py", "cutsell_worker/post_selection_interior_gap_trim.py"):
        src = Path(rel).read_text(encoding="utf-8")
        for needle in ("VIDEO-2026", "D40F1D43", "5E01F214", "resorcina", "espinillas"):
            assert needle not in src
