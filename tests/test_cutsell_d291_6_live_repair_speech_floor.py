"""D-291.6: the live post-render physical repair never removes speech.

Defect (found on HEAD `01591b02` by reading the executed render path,
`export_job.run_export_job -> live_render_qc.render_with_post_render_qc ->
live_boundary_repair.repair_segment_for_finding`): the ONE physical repair
the technical QC loop can apply was bounded only by the routing tolerance
(0.6 s), a trim fraction and a minimum remaining duration. Two ways it
could cut real speech:

1. A measured silence that STRADDLES a join was trimmed by its WHOLE
   duration from one segment's edge, although only the part inside that
   segment's own rendered window belongs to it -- up to 0.6 s of the
   neighbouring speech at the head or tail of the segment went with it.
2. No word boundary was consulted on either edge (the trailing branch only
   trusted the renderer's silence tightener; the leading branch had no
   floor at all), so a join-instant probe finding (zero measured extent)
   trimmed 50 ms off an edge that the frozen clip's own word timings show
   is still inside a word.

Fix, in the owning authority (BoundaryEngine's live physical repair): the
trim is clamped to the defect measured INSIDE the segment's own window,
and, with the frozen draft's word timings supplied by the live loop, never
enters a word; a zero-extent finding needs word evidence of room or the
repair is refused (PHYSICAL_FAIL_UNREPAIRABLE, recorded, never a silent
cut). No new thresholds: `_MIN_TRIM_SEC` and the routing tolerance keep
their D-030/D-097.4 values; the legacy call without word evidence is
byte-identical (D-097.4's own tests).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from cutsell_worker import live_render_qc
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, Word
from cutsell_worker.live_boundary_repair import repair_segment_for_finding, speech_room_at_edge
from cutsell_worker.post_render_media_qc import ABRUPT_AUDIO_DISCONTINUITY, LINGERING_ACCIDENTAL_SILENCE
from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding, PostRenderQCResult
from cutsell_worker.render_plan import RenderSegment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    # One continuous 8 s tone: no natural silence, so the renderer's trailing
    # tightener leaves every edge where the plan put it and every finding
    # below is deliberately injected.
    path = str(tmp_path_factory.mktemp("d291_6") / "source.mp4")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=8",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", path,
    ], check=True)
    return path


def _words(spans):
    return tuple(Word(text=f"w{i}", start=s, end=e) for i, (s, e) in enumerate(spans))


def _clip(clip_id, start, end, *, words=()):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end,
        text=clip_id, caption_text=clip_id, selected=True, words=_words(words),
    )


def _draft(clips):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="d291-6", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics={},
    )


def _segment(clip, source_path):
    return RenderSegment(
        clip_id=clip.clip_id, source_asset_id=clip.source_asset_id, source_path=source_path,
        start=clip.start, end=clip.end,
    )


def _silence(start, end):
    return PostRenderFinding(
        kind=LINGERING_ACCIDENTAL_SILENCE, start=start, end=end,
        detail={"duration_sec": end - start}, routes_to="BoundaryEngine",
    )


def _join(at):
    return PostRenderFinding(kind=ABRUPT_AUDIO_DISCONTINUITY, start=at, end=at, detail={}, routes_to="BoundaryEngine")


@pytest.fixture
def two_segments(source_video):
    a = _clip("a", 0.0, 3.0)
    b = _clip("b", 3.0, 6.0)
    return a, b, (_segment(a, source_video), _segment(b, source_video))


# --------------------------------------------------- rule 1: in-window clamp

def test_a_silence_straddling_the_join_trims_only_its_part_inside_the_trailing_segment(two_segments):
    a, b, segments = two_segments
    # Output silence 2.6..3.4 straddles the join at 3.0: 0.4 s belongs to
    # `a`'s window, 0.4 s to `b`'s. Before D-291.6 the whole 0.8 s came off
    # `a`'s tail, i.e. 0.4 s of `a`'s real content.
    repaired, attempt = repair_segment_for_finding(segments, _silence(2.6, 3.4))
    assert attempt.clip_id == "a" and attempt.edge == "trailing"
    assert attempt.trim_sec == pytest.approx(0.4, abs=1e-6)
    assert attempt.measured_inside_window_sec == pytest.approx(0.4, abs=1e-6)
    assert repaired[0].end == pytest.approx(2.6, abs=1e-6)
    assert repaired[1] == segments[1]


def test_a_silence_straddling_the_join_trims_only_its_part_inside_the_leading_segment(two_segments):
    a, b, segments = two_segments
    # Output silence 2.5..3.7: too far past `a`'s end for `a`'s trailing
    # edge, so it routes to `b`'s leading edge. Only the 0.7 s inside `b`'s
    # window is `b`'s; before D-291.6 the full 1.2 s came off `b`'s head.
    repaired, attempt = repair_segment_for_finding(segments, _silence(2.5, 3.7))
    assert attempt.clip_id == "b" and attempt.edge == "leading"
    assert attempt.trim_sec == pytest.approx(0.7, abs=1e-6)
    assert repaired[1].start == pytest.approx(3.7, abs=1e-6)
    assert repaired[0] == segments[0]


def test_a_silence_fully_inside_the_window_is_trimmed_as_before(two_segments):
    a, b, segments = two_segments
    repaired, attempt = repair_segment_for_finding(segments, _silence(2.7, 3.0))
    assert attempt.clip_id == "a" and attempt.trim_sec == pytest.approx(0.3, abs=1e-6)
    assert repaired[0].end == pytest.approx(2.7, abs=1e-6)


# ----------------------------------------------------- rule 2: word floor

def test_trailing_trim_stops_at_the_last_word_end(two_segments):
    a, b, segments = two_segments
    speech = {"a": ((0.2, 1.0), (1.1, 2.8))}
    repaired, attempt = repair_segment_for_finding(segments, _silence(2.5, 3.0), protected_speech_by_clip_id=speech)
    assert attempt.clip_id == "a" and attempt.edge == "trailing"
    assert attempt.speech_room_sec == pytest.approx(0.2, abs=1e-6)
    assert attempt.trim_sec == pytest.approx(0.2, abs=1e-6)
    assert repaired[0].end == pytest.approx(2.8, abs=1e-6)


def test_leading_trim_stops_at_the_first_word_start(two_segments):
    a, b, segments = two_segments
    speech = {"b": ((3.4, 4.0), (4.1, 5.9))}
    repaired, attempt = repair_segment_for_finding(segments, _silence(2.5, 3.7), protected_speech_by_clip_id=speech)
    assert attempt.clip_id == "b" and attempt.edge == "leading"
    assert attempt.speech_room_sec == pytest.approx(0.4, abs=1e-6)
    assert attempt.trim_sec == pytest.approx(0.4, abs=1e-6)
    assert repaired[1].start == pytest.approx(3.4, abs=1e-6)


def test_a_word_at_the_edge_refuses_the_repair_instead_of_clipping_it(two_segments):
    a, b, segments = two_segments
    speech = {"a": ((0.2, 1.0), (1.1, 2.98))}  # 20 ms of room: below the minimum trim
    assert repair_segment_for_finding(segments, _silence(2.5, 3.0), protected_speech_by_clip_id=speech) is None


def test_a_fragment_inherits_its_parent_clip_words(source_video):
    a = _clip("a", 0.0, 3.0)
    fragment = RenderSegment(
        clip_id="a#0", source_asset_id="src", source_path=source_video, start=0.0, end=3.0,
        render_fragment_id="a#0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=1,
    )
    speech = {"a": ((0.2, 2.8),)}
    repaired, attempt = repair_segment_for_finding((fragment,), _silence(2.5, 3.0), protected_speech_by_clip_id=speech)
    assert attempt.trim_sec == pytest.approx(0.2, abs=1e-6)


# --------------------------------- zero-extent (join-instant) probe findings

def test_join_instant_finding_needs_word_evidence_of_room(two_segments):
    a, b, segments = two_segments
    # Word evidence for `a` says speech runs to the edge: refuse.
    assert repair_segment_for_finding(segments, _join(3.0), protected_speech_by_clip_id={"a": ((0.1, 3.0),)}) is None
    # Word evidence for `a` shows room: the D-097.4 minimum trim applies.
    repaired, attempt = repair_segment_for_finding(segments, _join(3.0), protected_speech_by_clip_id={"a": ((0.1, 2.5),)})
    assert attempt.clip_id == "a" and attempt.trim_sec == pytest.approx(0.05, abs=1e-6)
    # Word evidence supplied for the draft but none for `a` (no ASR words):
    # `a` is refused, fail closed; the join then routes to `b`'s leading
    # edge, which is repaired only because `b`'s own words show room.
    repaired, attempt = repair_segment_for_finding(segments, _join(3.0), protected_speech_by_clip_id={"b": ((3.1, 5.0),)})
    assert attempt.clip_id == "b" and attempt.edge == "leading" and attempt.trim_sec == pytest.approx(0.05, abs=1e-6)
    # No room on either side of the join: nothing is trimmed.
    assert repair_segment_for_finding(
        segments, _join(3.0), protected_speech_by_clip_id={"a": ((0.1, 3.0),), "b": ((3.0, 5.0),)},
    ) is None
    assert repair_segment_for_finding(segments, _join(3.0), protected_speech_by_clip_id={"b": ((3.0, 5.0),)}) is None


def test_legacy_call_without_word_evidence_keeps_the_d097_4_join_repair(two_segments):
    a, b, segments = two_segments
    repaired, attempt = repair_segment_for_finding(segments, _join(3.0))
    assert attempt.clip_id == "a" and attempt.edge == "trailing"
    assert attempt.trim_sec == pytest.approx(0.05, abs=1e-6)
    assert attempt.speech_room_sec is None


def test_speech_room_ignores_words_outside_the_segment():
    seg = RenderSegment(clip_id="x", source_asset_id="src", source_path="", start=10.0, end=13.0)
    speech = ((2.0, 3.0), (10.4, 11.0), (12.2, 12.7), (14.0, 15.0))
    assert speech_room_at_edge(seg, "leading", edge_time=13.0, speech=speech) == pytest.approx(0.4)
    assert speech_room_at_edge(seg, "trailing", edge_time=13.0, speech=speech) == pytest.approx(0.3)
    assert speech_room_at_edge(seg, "trailing", edge_time=13.0, speech=((2.0, 3.0),)) == pytest.approx(3.0)


# ------------------------------------------ the live loop supplies the words

def test_protected_speech_lists_only_clips_with_word_timings():
    draft = _draft([_clip("a", 0.0, 3.0, words=((0.2, 1.0), (1.1, 2.8))), _clip("b", 3.0, 6.0)])
    assert live_render_qc.protected_speech_by_clip_id(draft) == {"a": ((0.2, 1.0), (1.1, 2.8))}


def test_live_loop_refuses_to_clip_a_word_for_a_join_probe(monkeypatch, tmp_path, source_video):
    a = _clip("a", 0.0, 3.0, words=((0.2, 1.4), (1.5, 3.0)))
    b = _clip("b", 3.0, 6.0, words=((3.0, 4.0), (4.2, 5.9)))
    draft = _draft([a, b])
    segments = (_segment(a, source_video), _segment(b, source_video))
    render_calls = []
    real_render = live_render_qc.render_preview

    def fake_qc(media_path, **kwargs):
        return PostRenderQCResult(status="FAIL", findings=(_join(3.0),))

    def spy_render(segs, out, **kwargs):
        render_calls.append(tuple(segs))
        return real_render(segs, out, **kwargs)

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", fake_qc)
    monkeypatch.setattr(live_render_qc, "render_preview", spy_render)
    result = live_render_qc.render_with_post_render_qc(draft, segments, str(tmp_path / "out.mp4"), max_attempts=3)
    assert result.status != "PASS"
    assert [att.status for att in result.attempts] == ["PHYSICAL_FAIL_UNREPAIRABLE"]
    assert result.attempts[0].unrepairable_finding_count == 1
    assert len(render_calls) == 1  # no re-render with a clipped word


def test_live_loop_repairs_up_to_the_word_floor_and_records_the_evidence(monkeypatch, tmp_path, source_video):
    a = _clip("a", 0.0, 3.0, words=((0.2, 1.4), (1.5, 2.6)))
    b = _clip("b", 3.0, 6.0, words=((3.0, 4.0), (4.2, 5.9)))
    draft = _draft([a, b])
    segments = (_segment(a, source_video), _segment(b, source_video))
    qc_calls = []

    def fake_qc(media_path, **kwargs):
        qc_calls.append(media_path)
        if len(qc_calls) == 1:
            # Measured silence 2.4..3.2 straddles the join; 0.6 s is inside
            # `a`'s window but its last word ends at 2.6: only 0.4 s is room.
            return PostRenderQCResult(status="FAIL", findings=(_silence(2.4, 3.2),))
        return PostRenderQCResult(status="PASS", findings=())

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", fake_qc)
    result = live_render_qc.render_with_post_render_qc(draft, segments, str(tmp_path / "out.mp4"), max_attempts=3)
    assert result.status == "PASS"
    first = result.attempts[0]
    assert first.status == "PHYSICAL_FAIL_REPAIRED"
    applied = first.repair_applied
    assert applied["clip_id"] == "a" and applied["edge"] == "trailing"
    assert applied["measured_inside_window_sec"] == pytest.approx(0.6, abs=1e-6)
    assert applied["speech_room_sec"] == pytest.approx(0.4, abs=1e-6)
    assert applied["trim_sec"] == pytest.approx(0.4, abs=1e-6)
    assert applied["repaired_end"] == pytest.approx(2.6, abs=1e-6)
