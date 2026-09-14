"""D-097 §4 perceptual System Watch+Listen v1 + CLEAN RAW gate (QA layer).

The reviewer diagnoses and routes on the real rendered MP4; capability
statuses are explicit; NOT_IMPLEMENTED / UNCERTAIN / ERROR never become
PASS; the gate never passes on absent evidence. Synthetic media only.
"""
import shutil
import subprocess

import pytest

from benchmarks.clean_raw_gate import GATE_FAIL, GATE_INCOMPLETE, GATE_PASS, build_gate_report, format_report
from cutsell_worker import perceptual_watch_listen as pwl
from cutsell_worker import universal_clean_cut_validation as harness
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.render_plan import RenderSegment


def _clip(clip_id, start, end, text):
    return DraftClip(clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text, caption_text=text, selected=True)


def _draft(clips, diagnostics=None):
    return DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                         selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics or {})


def _seg(clip_id, start, end, parent=None):
    return RenderSegment(clip_id=clip_id, source_asset_id="src", source_path="/x.mp4", start=start, end=end, parent_semantic_clip_id=parent)


# --- status algebra ------------------------------------------------------------------

def test_review_is_never_pass_while_a_capability_is_not_implemented():
    caps = [
        pwl.CapabilityReport("a", pwl.EVALUATED_PASS, "mp4_measured"),
        pwl.CapabilityReport("b", pwl.NOT_IMPLEMENTED, "none"),
    ]
    assert pwl.overall_status(caps) == pwl.REVIEW_UNCERTAIN


def test_review_fails_on_any_evaluated_fail_and_passes_only_when_all_pass():
    assert pwl.overall_status([pwl.CapabilityReport("a", pwl.EVALUATED_PASS, "x"), pwl.CapabilityReport("b", pwl.EVALUATED_FAIL, "x")]) == pwl.REVIEW_FAIL
    assert pwl.overall_status([pwl.CapabilityReport("a", pwl.EVALUATED_PASS, "x")]) == pwl.REVIEW_PASS
    assert pwl.overall_status([pwl.CapabilityReport("a", pwl.ERROR, "x")]) == pwl.REVIEW_UNCERTAIN
    assert pwl.overall_status([]) == pwl.REVIEW_UNCERTAIN


# --- capabilities without media --------------------------------------------------------

def test_reset_debris_at_entry_is_routed_to_boundary_and_absent_evidence_is_uncertain():
    diag = {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": [
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "hand_motion_reset_candidate", "start": 12.00, "end": 12.20, "confidence": 0.93},  # interior: not an edge
    ]}]}}
    report = pwl._reset_debris_at_edges(_draft([], diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL and report.method == "source_evidence_mapped"
    assert len(report.findings) == 1
    assert report.findings[0].routes_to == pwl.ROUTE_BOUNDARY and report.findings[0].detail["edge"] == "entry"
    absent = pwl._reset_debris_at_edges(_draft([], {}), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert absent.status == pwl.UNCERTAIN


def test_repeated_audience_content_between_different_semantic_clips_routes_to_best_take():
    draft = _draft([
        _clip("a", 0.0, 4.0, "me hice un test de sangre completo para ver todo"),
        _clip("b", 10.0, 14.0, "me hice un test de sangre completo para ver todo esto"),
        _clip("c", 20.0, 24.0, "otra idea totalmente distinta sobre la rutina"),
    ])
    segments = (_seg("a", 0.0, 4.0), _seg("b", 10.0, 14.0), _seg("c", 20.0, 24.0))
    report = pwl._repeated_audience_content(draft, segments, [(0, 4), (4, 8), (8, 12)])
    assert report.status == pwl.EVALUATED_FAIL
    assert report.findings[0].routes_to == pwl.ROUTE_BEST_TAKE
    assert (report.findings[0].detail["left_clip_id"], report.findings[0].detail["right_clip_id"]) == ("a", "b")
    # two physical pieces of ONE semantic clip are never "repeated content"
    same = pwl._repeated_audience_content(draft, (_seg("a__l", 0.0, 2.0, parent="a"), _seg("a__r", 2.0, 4.0, parent="a")), [(0, 2), (2, 4)])
    assert same.status == pwl.UNCERTAIN or same.findings == ()


# --- real MP4 ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tone_gap_tone(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("pwl") / "cand.mp4"
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5:sample_rate=48000",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=2.0",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5:sample_rate=48000",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=5",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]", "-map", "[a]", "-map", "3:v",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
    ])
    return str(path)


def test_review_measures_dead_air_on_the_real_mp4_and_routes_it(tone_gap_tone):
    draft = _draft([_clip("a", 0.0, 5.0, "una idea completa")])
    review = pwl.review_rendered_candidate(tone_gap_tone, draft, (_seg("a", 0.0, 5.0),), [(0.0, 5.0)])
    dead_air = next(c for c in review.capabilities if c.capability == "interior_dead_air_mp4")
    assert dead_air.status == pwl.EVALUATED_FAIL and dead_air.method == "mp4_measured"
    assert dead_air.findings[0].kind == pwl.INTERIOR_DEAD_AIR and dead_air.findings[0].routes_to == pwl.ROUTE_BOUNDARY
    assert review.status == pwl.REVIEW_FAIL
    payload = review.as_dict()
    assert payload["gate_mode"] == pwl.GATE_MODE_ADVISORY_V1 and payload["blocking"] is False
    assert payload["human_watch_listen_required"] is True
    assert payload["capability_status_counts"][pwl.NOT_IMPLEMENTED] == len(pwl.NOT_IMPLEMENTED_CAPABILITIES)
    assert payload["routing"][pwl.ROUTE_BOUNDARY] >= 1


def test_speech_energy_at_a_join_inside_the_tone_is_flagged_uncertain(tone_gap_tone):
    draft = _draft([_clip("a", 0.0, 1.0, "x"), _clip("b", 1.0, 5.0, "y")])
    review = pwl.review_rendered_candidate(tone_gap_tone, draft, (_seg("a", 0.0, 1.0), _seg("b", 1.0, 5.0)), [(0.0, 1.0), (1.0, 5.0)])
    energy = next(c for c in review.capabilities if c.capability == "cut_adjacent_speech_energy_mp4")
    assert energy.status == pwl.UNCERTAIN  # a loud tone sits on both sides of the 1.0 s join
    assert {f.detail["side"] for f in energy.findings} == {"outgoing", "incoming"}


def test_reviewer_errors_are_reported_never_passed(tmp_path):
    draft = _draft([_clip("a", 0.0, 5.0, "x")])
    review = pwl.review_rendered_candidate(str(tmp_path / "missing.mp4"), draft, (_seg("a", 0.0, 5.0),), [(0.0, 5.0)])
    assert review.status in {pwl.REVIEW_UNCERTAIN, pwl.REVIEW_FAIL}
    assert review.status != pwl.REVIEW_PASS


# --- harness wiring ---------------------------------------------------------------------

def test_harness_marks_a_technically_clean_candidate_pending_human_watch_listen():
    class _QC:
        status = "PASS"
        deliverable = True
        delivery_status = "DELIVERABLE"
        output_path = "/p.mp4"
        plan_id = "plan"
        plan_version = 1
        semantic_hash = "h"
        attempts = ()

    diag = harness._live_render_qc_diagnostics(_QC(), skipped_reason=None, perceptual_status="UNCERTAIN")
    assert diag["deliverable"] is True
    assert diag["delivery_status"] == "DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:perceptual=UNCERTAIN"
    assert diag["human_watch_listen_required"] is True
    incomplete = harness._live_render_qc_diagnostics(_QC(), skipped_reason=None, story_completeness="incomplete_no_usable_realization", perceptual_status="UNCERTAIN")
    assert incomplete["deliverable"] is False and incomplete["delivery_status"].startswith("NOT_DELIVERABLE_INCOMPLETE_STORY_REVIEW")


# --- CLEAN RAW gate ----------------------------------------------------------------------

def _result(**overrides):
    base = {
        "stage_status": {"story_completeness": "complete", "no_usable_realization_family_count": 0,
                         "take_segmentation": {"status": "complete", "polarity_rejoin_count": 1}},
        "diagnostics": {"boundary_engine_pass": {"stage": "post_freeze", "interior_split_count": 2, "audio_entry_trim_count": 1, "audio_exit_trim_count": 3},
                        "hybrid_editorial_chunks": [{"diagnostics": [{"protected_polarity_fragments": [{"clip_id": "p"}]}]}]},
        "live_render_qc": {"status": "PASS", "deliverable": True, "delivery_status": "DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN:perceptual=UNCERTAIN",
                           "attempts": [{"status": "PASS", "findings": [], "renderer_trailing_trims": [{"trim_sec": 0.4}], "dead_air_reconciliation": []}]},
        "perceptual_watch_listen": {"status": "UNCERTAIN", "gate_mode": "advisory_v1", "capability_status_counts": {"NOT_IMPLEMENTED": 4}, "routing": {}},
    }
    base.update(overrides)
    return base


def _ladder(regions):
    return {"regions": regions, "summary": {"cutai_parity": {"level1_selection_seconds": 0.0}}}


def test_gate_passes_only_with_complete_clean_evidence():
    report = build_gate_report(_result(), _ladder([{"level": "LEVEL_3", "kind": "consensus_keep", "duration_sec": 5.0}]))
    assert report["gate"]["status"] == GATE_PASS
    assert report["metrics"]["polarity_rejoin_count"] == 1 and report["metrics"]["protected_polarity_fragment_count"] == 1
    assert report["metrics"]["renderer_trailing_trim_seconds"] == 0.4
    assert "CLEAN RAW GATE: PASS" in format_report(report)


def test_gate_is_incomplete_without_the_ladder_or_the_perceptual_review():
    assert build_gate_report(_result(), None)["gate"]["status"] == GATE_INCOMPLETE
    no_perceptual = _result(perceptual_watch_listen={})
    report = build_gate_report(no_perceptual, _ladder([]))
    assert report["gate"]["status"] == GATE_INCOMPLETE and "perceptual_watch_listen" in report["gate"]["missing_evidence"]


def test_gate_fails_on_failed_material_dead_air_or_incomplete_story():
    failed = _ladder([{"level": "LEVEL_1", "kind": "false_keep", "refinement": "failed_or_process_material_retained", "duration_sec": 2.5}])
    report = build_gate_report(_result(), failed)
    assert report["gate"]["status"] == GATE_FAIL and report["gate"]["blocking"] == ["failed_material_regions:1 (2.5 s)"]
    dead_air = _result(live_render_qc={"status": "NEEDS_HUMAN_REVIEW", "deliverable": False, "attempts": [{"findings": [{"kind": "LINGERING_ACCIDENTAL_SILENCE"}]}]})
    assert build_gate_report(dead_air, _ladder([]))["gate"]["status"] == GATE_FAIL
    incomplete = _result(stage_status={"story_completeness": "incomplete_no_usable_realization"})
    assert "story:incomplete_no_usable_realization" in build_gate_report(incomplete, _ladder([]))["gate"]["blocking"]


def test_gate_uses_physical_regions_when_the_ladder_aligned_the_rendered_mp4():
    ladder = {"regions": [{"level": "LEVEL_1", "kind": "false_keep", "refinement": "redundant_realization_both_kept", "duration_sec": 2.0}],
              "physical_regions": [{"level": "LEVEL_1", "kind": "false_keep", "refinement": "loose_exit_edge", "duration_sec": 0.4}]}
    report = build_gate_report(_result(), ladder)
    assert report["metrics"]["ladder_source"] == "physical_regions"
    assert report["metrics"]["failed_material_regions"] == 0 and report["metrics"]["loose_edge_seconds"] == 0.4
    assert report["gate"]["status"] == GATE_PASS
