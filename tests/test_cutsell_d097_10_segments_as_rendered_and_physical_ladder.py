"""D-097.10 (R14) -- post-render QA layers must read the segments AS RENDERED.

RAW 34047064840 (head 6fc880d; deliverable MP4): the renderer's recorded
last mechanical op (`tighten_trailing_silence`) moved 16 exits by 0.24-2.10 s,
yet the perceptual reviewer mapped source reset events onto the PRE-tighten
segment ends -- 9 of its 13 "reset debris at exit" findings pointed at
material that is not in the file (perceptual FAIL, routed to BoundaryEngine
every run). The four-way ladder already carries a physical view from render
verification (a correlation estimate that missed 2 of 25 fragments); it now
also uses the renderer's exact recorded trims, and the workflows print the
physical (final MP4) Level-1 rows as the headline. CODE EXISTS != VIDEO USED
IT applies to our own QA layers too: they must measure the video.
"""
from __future__ import annotations

from benchmarks.video00_quality_ladder import physical_engine_result, recorded_renderer_ends
from cutsell_worker import perceptual_watch_listen as pwl
from cutsell_worker.contracts import DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.universal_clean_cut_validation import segments_as_rendered


def _seg(clip_id, start, end, fragment=None):
    return RenderSegment(clip_id=clip_id, source_asset_id="src", source_path="/x.mp4", start=start, end=end,
                         render_fragment_id=fragment)


def _draft(diagnostics):
    return DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                         selected=(), alternates=(), discarded=(), diagnostics=diagnostics)


# ------------------------------------------------------- segments as rendered

def test_recorded_trailing_trims_shorten_the_reviewed_segments():
    segments = (_seg("a", 13.78, 23.28), _seg("b", 35.46, 46.42), _seg("c", 50.0, 55.0))
    trims = [
        {"clip_id": "a", "render_fragment_id": None, "original_end": 23.28, "tightened_end": 22.424, "trim_sec": 0.856},
        {"clip_id": "b", "render_fragment_id": None, "original_end": 46.42, "tightened_end": 45.254, "trim_sec": 1.166},
    ]
    rendered, applied = segments_as_rendered(segments, trims)
    assert applied == 2
    assert [(s.clip_id, s.start, s.end) for s in rendered] == [("a", 13.78, 22.424), ("b", 35.46, 45.254), ("c", 50.0, 55.0)]


def test_a_trim_never_lengthens_or_empties_a_segment_and_matches_fragments():
    segments = (_seg("a", 10.0, 12.0), _seg("a", 20.0, 24.0, fragment="a__psig1"), _seg("b", 30.0, 33.0))
    trims = [
        {"clip_id": "a", "render_fragment_id": None, "tightened_end": 12.5},        # would lengthen: ignored
        {"clip_id": "a", "render_fragment_id": "a__psig1", "tightened_end": 23.1},  # fragment-specific
        {"clip_id": "b", "render_fragment_id": None, "tightened_end": 29.0},        # would empty: ignored
        {"clip_id": "zzz", "tightened_end": "not-a-number"},
    ]
    rendered, applied = segments_as_rendered(segments, trims)
    assert applied == 1
    assert [(s.clip_id, s.end) for s in rendered] == [("a", 12.0), ("a", 23.1), ("b", 33.0)]
    assert segments_as_rendered(segments, ()) == (segments, 0)


def test_reset_debris_after_the_rendered_exit_is_not_a_finding():
    diag = {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": [
        {"kind": "hand_motion_reset_candidate", "start": 22.869, "end": 22.935, "confidence": 1.0},
    ]}]}}
    pre_tighten = (_seg("a", 13.78, 23.28),)
    defect_shape = pwl._reset_debris_at_edges(_draft(diag), pre_tighten, [(0.0, 9.5)])
    assert defect_shape.status == pwl.EVALUATED_FAIL and defect_shape.findings[0].detail["edge"] == "exit"
    rendered, _ = segments_as_rendered(pre_tighten, [{"clip_id": "a", "tightened_end": 22.424}])
    reviewed = pwl._reset_debris_at_edges(_draft(diag), rendered, [(0.0, 8.644)])
    assert reviewed.status == pwl.EVALUATED_PASS and reviewed.findings == ()


def test_reset_debris_inside_the_rendered_exit_is_still_a_finding():
    diag = {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": [
        {"kind": "hand_motion_reset_candidate", "start": 52.671, "end": 52.738, "confidence": 1.0},
    ]}]}}
    rendered, applied = segments_as_rendered((_seg("a", 48.97, 52.75),), [])
    assert applied == 0
    reviewed = pwl._reset_debris_at_edges(_draft(diag), rendered, [(0.0, 3.78)])
    assert reviewed.status == pwl.EVALUATED_FAIL and reviewed.findings[0].routes_to == pwl.ROUTE_BOUNDARY


# ------------------------------------------------------- physical ladder view

def _engine_with_trims(trims, selected=(("a", 0.0, 10.0), ("b", 20.0, 30.0))):
    return {
        "selected": [{"clip_id": c, "start": s, "end": e, "text": c} for c, s, e in selected],
        "discarded": [], "alternates": [], "diagnostics": {},
        "live_render_qc": {"status": "PASS", "attempts": [
            {"status": "FAIL", "renderer_trailing_trims": [{"clip_id": "a", "tightened_end": 5.0}]},  # an earlier attempt: ignored
            {"status": "PASS", "renderer_trailing_trims": trims},
        ]},
    }


def test_recorded_renderer_ends_read_the_last_attempt_only():
    engine = _engine_with_trims([{"clip_id": "a", "tightened_end": 9.2}, {"clip_id": "b", "tightened_end": "x"}])
    assert recorded_renderer_ends(engine) == {"a": 9.2}
    assert recorded_renderer_ends({"selected": []}) == {}


def test_physical_view_falls_back_to_the_recorded_trims_without_an_mp4():
    engine = _engine_with_trims([{"clip_id": "a", "tightened_end": 9.2}])
    physical = physical_engine_result(engine, None)
    assert [(r["clip_id"], r["start"], r["end"]) for r in physical["selected"]] == [("a", 0.0, 9.2), ("b", 20.0, 30.0)]
    assert physical["physical_source"] == {"recorded_renderer_trims": 1, "render_verification": 0, "plan": 1}


def test_recorded_trims_take_precedence_over_the_correlation_estimate_and_cover_missing_fragments():
    engine = _engine_with_trims([{"clip_id": "a", "tightened_end": 9.2}])
    rv = {"render_duration_sec": 19.0, "fragments": [
        {"clip_id": "a", "raw_start": 0.0, "raw_end": 10.0, "render_start": 0.0, "found": True, "physical_raw_end": 9.4},
        {"clip_id": "b", "raw_start": 20.0, "raw_end": 30.0, "render_start": 9.2, "found": True, "physical_raw_end": 29.8},
    ]}
    physical = physical_engine_result(engine, rv)
    assert [(r["clip_id"], r["end"]) for r in physical["selected"]] == [("a", 9.2), ("b", 29.8)]
    assert physical["physical_source"] == {"recorded_renderer_trims": 1, "render_verification": 1, "plan": 0}


def test_no_physical_view_without_any_render_truth():
    engine = {"selected": [{"clip_id": "a", "start": 0.0, "end": 10.0, "text": "a"}], "discarded": [], "alternates": [], "diagnostics": {}}
    assert physical_engine_result(engine, None) is None
