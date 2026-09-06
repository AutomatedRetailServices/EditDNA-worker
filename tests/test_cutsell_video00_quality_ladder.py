"""D-095: Video00 quality ladder (RAW vs Cut.ai vs Human Gold vs CutSell) -- QA-only tooling."""
from __future__ import annotations

import json

import pytest

from benchmarks.video00_quality_ladder import (
    AUTH_ATTEMPT,
    AUTH_BEST_TAKE,
    AUTH_BOUNDARY,
    AUTH_CLUSTERER,
    AUTH_COMPOSITE,
    AUTH_REALIZATION,
    CONSENSUS_DELETE,
    CONSENSUS_KEEP,
    FAILED_MATERIAL_RETAINED,
    FALSE_DELETE,
    FALSE_KEEP,
    GOLD_KEEPS_CUTAI_DROPS,
    GOLD_REMOVES_CUTAI_KEEPS,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LOST_FAMILY_COMPETITION,
    MATCHES_GOLD_BEYOND_CUTAI,
    MATCHES_GOLD_OVER_CUTAI,
    MISSING_DELIVERY,
    NO_CANDIDATE,
    REDUNDANT_REALIZATION,
    RESTORED_BY_RESOLVER,
    TAKE_CHOICE_AGAINST_REFERENCES,
    UNGROUPED_RETRY,
    ReferenceCut,
    Span,
    build_region_map,
    classify_triple,
    engine_candidates,
    normalize_spans,
    parent_clip_id,
    render_markdown,
    write_csv,
)


def _engine(selected=(), discarded=(), alternates=(), *, groups=(), ideas=(), restored=(), freeze=None):
    def rows(items):
        return [{"clip_id": c, "start": s, "end": e, "text": t} for (c, s, e, t) in items]
    diag = {
        "take_judge_groups": [
            {
                "group_id": gid,
                "selected_clip_id": winner,
                "local_selected_clip_id": winner,
                "semantic_override_applied": False,
                "semantic_candidates": [{"clip_id": m} for m in members],
            }
            for gid, winner, members in groups
        ],
        "canonical_edit_plan": {
            "ideas": [
                {"idea_id": iid, "winning_clip_ids": list(w), "discarded_clip_ids": list(d), "is_composite": comp,
                 "coverage_status": "complete", "authoritative_resolution_status": "RESOLVED_WINNER"}
                for iid, w, d, comp in ideas
            ]
        },
        "authoritative_story_placement": [{"clip_id": r} for r in restored],
        "selection_boundary_contract": freeze or {"plan_id": "plan_t", "plan_version": 1, "status": "verified"},
    }
    return {"selected": rows(selected), "discarded": rows(discarded), "alternates": rows(alternates), "diagnostics": diag}


def _regions(report, scope="selection"):
    return [r for r in report["regions"] if r["scope"] == scope]


def _region_at(report, start, end):
    for r in report["regions"]:
        if abs(r["raw_start"] - start) < 1e-6 and abs(r["raw_end"] - end) < 1e-6:
            return r
    raise AssertionError(f"no region {start}-{end}: {[(r['raw_start'], r['raw_end']) for r in report['regions']]}")


# --- pure helpers -----------------------------------------------------------

def test_normalize_spans_merges_touching_and_overlapping():
    assert normalize_spans([Span(5, 8), Span(0, 5), Span(7, 9), Span(12, 13)]) == (Span(0, 9), Span(12, 13))


def test_parent_clip_id_strips_boundary_fragment_suffix():
    assert parent_clip_id("clip_abc__psigle546") == "clip_abc"
    assert parent_clip_id("clip_abc") == "clip_abc"


@pytest.mark.parametrize(
    "triple, expected",
    [
        ((True, True, True), (LEVEL_3, CONSENSUS_KEEP)),
        ((False, False, False), (LEVEL_3, CONSENSUS_DELETE)),
        ((True, True, False), (LEVEL_1, MISSING_DELIVERY)),
        ((False, False, True), (LEVEL_1, FALSE_KEEP)),
        ((True, False, True), (LEVEL_2, GOLD_REMOVES_CUTAI_KEEPS)),
        ((False, True, False), (LEVEL_2, GOLD_KEEPS_CUTAI_DROPS)),
        ((True, False, False), (LEVEL_3, MATCHES_GOLD_OVER_CUTAI)),
        ((False, True, True), (LEVEL_3, MATCHES_GOLD_BEYOND_CUTAI)),
    ],
)
def test_classify_triple_covers_all_eight_keep_combinations(triple, expected):
    assert classify_triple(*triple) == expected


def test_engine_candidates_reads_all_three_buckets_with_parent_ids():
    engine = _engine(selected=[("a__psig1", 0, 1, "x")], discarded=[("b", 2, 3, "y")], alternates=[("c", 4, 5, "z")])
    rows = engine_candidates(engine)
    assert [(r["status"], r["clip_id"], r["parent_clip_id"]) for r in rows] == [
        ("selected", "a__psig1", "a"), ("alternate", "c", "c"), ("discarded", "b", "b"),
    ]


# --- region map -------------------------------------------------------------

def test_consensus_regions_are_level_3_and_never_attributed():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    report = build_region_map(raw_duration_sec=20.0, cutai=cutai, gold=gold,
                              engine_result=_engine(selected=[("a", 0, 10, "hook")]))
    keep = _region_at(report, 0, 10)
    delete = _region_at(report, 10, 20)
    assert (keep["level"], keep["kind"], keep["attributed_authority"]) == (LEVEL_3, CONSENSUS_KEEP, "none")
    assert (delete["level"], delete["kind"]) == (LEVEL_3, CONSENSUS_DELETE)
    assert report["summary"]["by_level"][LEVEL_1]["selection_count"] == 0


def test_missing_delivery_without_any_candidate_is_attributed_to_attempt_reconstruction():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10), (20, 30)])
    gold = ReferenceCut.from_spans("gold", [(0, 10), (20, 30)])
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold,
                              engine_result=_engine(selected=[("a", 0, 10, "hook")]))
    r = _region_at(report, 20, 30)
    assert (r["level"], r["kind"], r["refinement"], r["attributed_authority"]) == (
        LEVEL_1, MISSING_DELIVERY, NO_CANDIDATE, AUTH_ATTEMPT)


def test_missing_delivery_lost_to_family_winner_is_attributed_to_best_take():
    cutai = ReferenceCut.from_spans("cutai", [(20, 30)])
    gold = ReferenceCut.from_spans("gold", [(20, 30)])
    engine = _engine(
        selected=[("bad", 0, 10, "diagnosis attempt one")],
        discarded=[("good", 20, 30, "diagnosis attempt two")],
        groups=[("g1", "bad", ("bad", "good"))],
    )
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 20, 30)
    assert (r["kind"], r["refinement"], r["attributed_authority"]) == (
        MISSING_DELIVERY, TAKE_CHOICE_AGAINST_REFERENCES, AUTH_BEST_TAKE)
    # and the kept realization both references rejected is itself LEVEL_1 false keep
    kept = _region_at(report, 0, 10)
    assert (kept["level"], kept["kind"], kept["refinement"]) == (LEVEL_1, FALSE_KEEP, TAKE_CHOICE_AGAINST_REFERENCES)


def test_missing_delivery_discarded_inside_idea_is_attributed_to_realization_resolver():
    cutai = ReferenceCut.from_spans("cutai", [(20, 30)])
    gold = ReferenceCut.from_spans("gold", [(20, 30)])
    engine = _engine(
        selected=[("w", 0, 10, "winner")],
        discarded=[("d", 20, 30, "discarded")],
        ideas=[("tg1", ("w",), ("d",), False)],
    )
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 20, 30)
    assert (r["refinement"], r["attributed_authority"]) == (LOST_FAMILY_COMPETITION, AUTH_REALIZATION)


def test_missing_delivery_deleted_outside_any_family_is_attributed_to_pre_resolver_cleanup():
    cutai = ReferenceCut.from_spans("cutai", [(20, 30)])
    gold = ReferenceCut.from_spans("gold", [(20, 30)])
    engine = _engine(selected=[("w", 0, 10, "winner")], discarded=[("d", 20, 30, "lonely delete")])
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 20, 30)
    assert (r["refinement"], r["attributed_authority"]) == (FALSE_DELETE, AUTH_COMPOSITE)


def test_false_keep_of_lone_material_both_references_remove_is_attributed_to_recording_process_removal():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(selected=[("a", 0, 10, "hook"), ("junk", 15, 19, "eh mm let me start again")])
    report = build_region_map(raw_duration_sec=30.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 15, 19)
    assert (r["level"], r["kind"], r["refinement"], r["attributed_authority"]) == (
        LEVEL_1, FALSE_KEEP, FAILED_MATERIAL_RETAINED, AUTH_ATTEMPT)


def test_false_keep_of_ungrouped_retry_is_attributed_to_idea_clusterer():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(selected=[
        ("a", 0, 10, "la biopsia confirmó un cáncer papilar de tiroides"),
        ("a2", 15, 24, "la biopsia confirmó que era cáncer papilar de tiroides"),
    ])
    report = build_region_map(raw_duration_sec=30.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 15, 24)
    assert (r["refinement"], r["attributed_authority"]) == (UNGROUPED_RETRY, AUTH_CLUSTERER)


def test_false_keep_of_second_family_realization_is_attributed_to_best_take():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(
        selected=[("first", 0, 10, "conclusion"), ("second", 15, 25, "conclusion again")],
        groups=[("g1", "first", ("first", "second"))],
    )
    report = build_region_map(raw_duration_sec=30.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 15, 25)
    assert (r["refinement"], r["attributed_authority"]) == (REDUNDANT_REALIZATION, AUTH_BEST_TAKE)
    assert "two kept realizations" in r["attribution_rationale"]


def test_false_keep_of_resolver_restored_clip_is_attributed_to_realization_resolver():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(selected=[("a", 0, 10, "hook"), ("r", 15, 20, "restored piece")], restored=["r"])
    report = build_region_map(raw_duration_sec=30.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 15, 20)
    assert (r["refinement"], r["attributed_authority"]) == (RESTORED_BY_RESOLVER, AUTH_REALIZATION)
    assert r["cutsell_candidates"][0]["restored"] is True


def test_level_2_when_cutsell_matches_cutai_but_gold_is_stricter():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10), (20, 30)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(selected=[("a", 0, 10, "hook"), ("b", 20, 30, "second conclusion")])
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold, engine_result=engine)
    r = _region_at(report, 20, 30)
    assert (r["level"], r["kind"], r["attributed_authority"]) == (LEVEL_2, GOLD_REMOVES_CUTAI_KEEPS, "none")
    assert report["summary"]["by_level"][LEVEL_1]["selection_count"] == 0


def test_level_3_when_cutsell_agrees_with_gold_against_cutai_in_both_directions():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10), (20, 30)])
    gold = ReferenceCut.from_spans("gold", [(0, 10), (35, 40)])
    engine = _engine(selected=[("a", 0, 10, "hook"), ("c", 35, 40, "gold-only keep")])
    report = build_region_map(raw_duration_sec=50.0, cutai=cutai, gold=gold, engine_result=engine)
    assert _region_at(report, 20, 30)["kind"] == MATCHES_GOLD_OVER_CUTAI
    assert _region_at(report, 35, 40)["kind"] == MATCHES_GOLD_BEYOND_CUTAI
    assert all(r["level"] == LEVEL_3 for r in _regions(report))


def test_sub_tolerance_edge_differences_are_boundary_scope_not_selection_level_1():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10.2)])
    gold = ReferenceCut.from_spans("gold", [(0, 10.2)])
    engine = _engine(selected=[("a", 0, 10.0, "hook")])
    report = build_region_map(raw_duration_sec=20.0, cutai=cutai, gold=gold, engine_result=engine)
    edge = _region_at(report, 10.0, 10.2)
    assert edge["scope"] == "boundary"
    assert edge["attributed_authority"] == AUTH_BOUNDARY
    assert report["summary"]["by_level"][LEVEL_1]["selection_count"] == 0
    assert report["summary"]["by_level"][LEVEL_1]["boundary_count"] == 1


def test_regions_never_straddle_two_cutsell_fragments():
    cutai = ReferenceCut.from_spans("cutai", [(0, 20)])
    gold = ReferenceCut.from_spans("gold", [(0, 20)])
    engine = _engine(selected=[("a", 0, 10, "first half"), ("b", 10, 20, "second half")])
    report = build_region_map(raw_duration_sec=20.0, cutai=cutai, gold=gold, engine_result=engine)
    assert [(r["raw_start"], r["raw_end"]) for r in report["regions"]] == [(0.0, 10.0), (10.0, 20.0)]


def test_traceability_row_per_selected_fragment_with_family_idea_freeze_and_coverage():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10), (20, 30)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    rendered = ReferenceCut.from_spans("cutsell_rendered", [(0, 10), (20, 25)])
    engine = _engine(
        selected=[("a", 0, 10, "hook"), ("b__psig9", 20, 30, "conclusion")],
        groups=[("g1", "b__psig9", ("b__psig9", "b2"))],
        ideas=[("tg1", ("b",), ("b2",), False)],
        freeze={"plan_id": "plan_9", "plan_version": 3, "status": "verified"},
    )
    report = build_region_map(raw_duration_sec=40.0, cutai=cutai, gold=gold, engine_result=engine, rendered=rendered)
    rows = {t["clip_id"]: t for t in report["traceability"]}
    assert rows["a"]["worst_level"] == LEVEL_3
    b = rows["b__psig9"]
    assert b["parent_clip_id"] == "b"
    assert b["retry_family"] == "g1" and b["idea_id"] == "tg1"
    assert (b["freeze_plan_id"], b["freeze_plan_version"]) == ("plan_9", 3)
    assert b["cutai_coverage"] == 1.0 and b["gold_coverage"] == 0.0
    assert b["rendered_coverage"] == 0.5
    assert b["worst_level"] == LEVEL_2


def test_summary_reports_parity_f1_durations_and_level1_share():
    cutai = ReferenceCut.from_spans("cutai", [(0, 10), (20, 30)])
    gold = ReferenceCut.from_spans("gold", [(0, 10), (20, 30)])
    engine = _engine(selected=[("a", 0, 10, "hook"), ("junk", 40, 45, "abandoned start")])
    report = build_region_map(raw_duration_sec=50.0, cutai=cutai, gold=gold, engine_result=engine)
    s = report["summary"]
    assert s["durations_sec"] == {"raw": 50.0, "cutai_keep": 20.0, "gold_keep": 20.0, "cutsell_keep": 15.0, "cutsell_rendered": None}
    assert s["selection_parity"]["cutai_vs_gold"]["f1"] == 1.0
    assert s["selection_parity"]["cutsell_vs_cutai"]["recall"] == 0.5
    assert s["cutai_parity"]["level1_selection_seconds"] == 15.0  # 10 missing + 5 false keep
    assert s["cutai_parity"]["level1_share_of_cutai_keep"] == 0.75
    assert s["level1_by_authority"][AUTH_ATTEMPT]["count"] == 2


def test_markdown_and_csv_and_json_outputs_render(tmp_path):
    cutai = ReferenceCut.from_spans("cutai", [(0, 10)])
    gold = ReferenceCut.from_spans("gold", [(0, 10)])
    engine = _engine(selected=[("a", 0, 10, "hook")])
    report = build_region_map(raw_duration_sec=20.0, cutai=cutai, gold=gold, engine_result=engine)
    md = render_markdown(report)
    assert "## Regions (selection scope)" in md and "## Traceability" in md and "LEVEL_1 | 0 |" in md
    csv_path = tmp_path / "ladder.csv"
    write_csv(report, csv_path)
    assert "consensus_keep" in csv_path.read_text(encoding="utf-8")
    json.dumps(report)  # serializable


def test_module_is_qa_only_and_never_imported_by_production_code():
    import pathlib
    src = pathlib.Path("cutsell_worker")
    offenders = [p for p in src.rglob("*.py") if "video00_quality_ladder" in p.read_text(encoding="utf-8")]
    assert offenders == []
