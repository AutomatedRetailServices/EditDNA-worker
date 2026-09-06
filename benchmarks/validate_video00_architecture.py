"""D-034/D-035: verify the CURRENT canonical Clean Cut Core V1 architecture
actually ran for a Video00 RAW candidate.

This replaces the workflow's old "Verify unified Selection architecture"
step, which asserted legacy whole-video Unified Selection reasoner success
markers (``selection_reasoner_status == "applied"``, ``external_brain_calls_
enabled == true``, ``hybrid_requested_group_count == 0``, etc.) -- fields
that are structurally always false/absent while Clean Cut Core V1 is the
active semantic authority, since V1 deliberately deactivates that
whole-video reasoner (see CLAUDE.md's Current mission and docs/
CUTSELL_DECISIONS.md D-034). That made the old check fail on every
correct V1 run, regardless of edit quality.

This verifier instead asserts evidence that the NEW architecture actually
executed: Clean Cut Core V1 is the active authority, the whole-video
reasoner being absent/disabled is the EXPECTED state (not an error), SWAP
being absent is the EXPECTED state (D-019: KEEP/DISCARD only, no alternate
inventory), CanonicalEditPlan was built, FinalEditReviewer ran, the bounded
semantic repair loop reported a status, CoverageLedger/StoryValidator
(Final Story Coherence Validation) ran, CompositeResolver diagnostics are
present, and -- when the candidate was not freeze-blocked -- that the plan
was freeze-ready, that Selection Freeze references the exact validated
plan id/version/hash, that Boundary only ran after that freeze, and that
the live PostRenderWatchListenQC/repair service (D-030/D-035) actually
executed against the real render. When the candidate WAS freeze-blocked,
it instead asserts the semantic-failure-blocks-freeze gate actually
engaged (Boundary/render never ran on an unfrozen draft), since a blocked
run legitimately never reaches those later stages.

A check here proves the architecture ran; it makes no claim about edit
quality -- that is Selection lock / Human Gold's job
(validate_video00_selection_lock.py), not this module's.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(result_path: str) -> tuple[bool, dict]:
    result = _load(result_path)
    stage_status = result.get("stage_status") or {}
    diagnostics = result.get("diagnostics") or {}
    live_render_qc = result.get("live_render_qc") or {}

    edit_plan = diagnostics.get("canonical_edit_plan") or {}
    final_edit_reviewer = diagnostics.get("final_edit_reviewer") or {}
    repair_loop = diagnostics.get("repair_loop") or {}
    coherence = diagnostics.get("final_story_coherence_validation") or {}
    contract = diagnostics.get("selection_boundary_contract") or {}
    unified_reasoner_diag = diagnostics.get("unified_selection_reasoner")

    freeze_blocked = bool(stage_status.get("freeze_blocked_pending_coherence_review"))

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: dict | None = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail or {}})

    check(
        "clean_cut_core_v1_active",
        stage_status.get("semantic") == "clean_cut_core_v1_idea_first"
        and str(stage_status.get("selection_phase_authority") or "").startswith(
            "clean_cut_core_v1_idea_first_keep_discard"
        ),
        {
            "semantic": stage_status.get("semantic"),
            "selection_phase_authority": stage_status.get("selection_phase_authority"),
        },
    )

    # EXPECTED, not an error: the whole-video Unified Selection reasoner is
    # deactivated under Clean Cut Core V1.
    check(
        "whole_video_unified_selection_reasoner_absent_as_expected",
        stage_status.get("unified_selection_reasoner") == "disabled_clean_cut_core_v1"
        and (
            unified_reasoner_diag is None
            or (isinstance(unified_reasoner_diag, dict) and unified_reasoner_diag.get("status") == "absent")
        ),
        {
            "stage_status.unified_selection_reasoner": stage_status.get("unified_selection_reasoner"),
            "diagnostics.unified_selection_reasoner": unified_reasoner_diag,
        },
    )

    check(
        "canonical_edit_plan_created",
        bool(edit_plan.get("plan_id")) and edit_plan.get("plan_version") is not None and bool(edit_plan.get("semantic_hash")),
        {"plan_id": edit_plan.get("plan_id"), "plan_version": edit_plan.get("plan_version")},
    )

    check(
        "final_edit_reviewer_executed",
        final_edit_reviewer.get("status") in ("PASS", "FAIL"),
        {"status": final_edit_reviewer.get("status")},
    )

    check(
        "repair_loop_status_present",
        repair_loop.get("status") in ("PASS", "NEEDS_HUMAN_REVIEW") and isinstance(repair_loop.get("attempt_count"), int),
        {"status": repair_loop.get("status"), "attempt_count": repair_loop.get("attempt_count")},
    )

    check(
        "coverage_ledger_story_validator_executed",
        "freeze_blocked" in coherence,
        {"freeze_blocked": coherence.get("freeze_blocked")},
    )

    # D-019: SWAP is out of scope for Clean Cut Core V1 -- everything not
    # SELECTed is DISCARDed, never parked as an alternate. Absence is
    # EXPECTED, not an error.
    check(
        "swap_absent_as_expected",
        int(result.get("alternate_count") or 0) == 0,
        {"alternate_count": result.get("alternate_count")},
    )

    check(
        "composite_resolver_diagnostics_available",
        all(
            key in result
            for key in ("hybrid_group_diagnostic_count", "hybrid_requested_group_count", "hybrid_available_group_count")
        ),
        {
            "hybrid_group_diagnostic_count": result.get("hybrid_group_diagnostic_count"),
            "hybrid_requested_group_count": result.get("hybrid_requested_group_count"),
            "hybrid_available_group_count": result.get("hybrid_available_group_count"),
        },
    )

    if freeze_blocked:
        # A high-confidence semantic failure must block Freeze/Boundary/
        # Render entirely -- prove the gate actually engaged, rather than
        # silently freezing/rendering an unsafe draft anyway.
        check(
            "semantic_failure_correctly_blocked_freeze_and_boundary",
            contract.get("status") == "not_frozen_freeze_blocked_by_coherence_review"
            and str(stage_status.get("human_boundary_polish") or "").startswith("not_applicable_freeze_blocked"),
            {
                "selection_boundary_contract.status": contract.get("status"),
                "human_boundary_polish": stage_status.get("human_boundary_polish"),
            },
        )
        check(
            "no_render_attempted_on_a_blocked_semantic_plan",
            (live_render_qc.get("status") or "not_attempted") == "not_attempted",
            {"live_render_qc.status": live_render_qc.get("status")},
        )
    else:
        check(
            "plan_was_freeze_ready_before_freeze",
            edit_plan.get("validation_state") == "frozen_ready",
            {"validation_state": edit_plan.get("validation_state")},
        )
        check(
            "freeze_references_exact_validated_plan",
            contract.get("status") == "verified"
            and contract.get("plan_id") == edit_plan.get("plan_id")
            and contract.get("plan_version") == edit_plan.get("plan_version")
            and contract.get("plan_semantic_hash") == edit_plan.get("semantic_hash"),
            {
                "contract": {
                    "status": contract.get("status"),
                    "plan_id": contract.get("plan_id"),
                    "plan_version": contract.get("plan_version"),
                    "plan_semantic_hash": contract.get("plan_semantic_hash"),
                },
                "edit_plan": {
                    "plan_id": edit_plan.get("plan_id"),
                    "plan_version": edit_plan.get("plan_version"),
                    "semantic_hash": edit_plan.get("semantic_hash"),
                },
            },
        )
        check(
            "boundary_ran_only_after_validated_freeze",
            bool(stage_status.get("human_boundary_polish"))
            and not str(stage_status.get("human_boundary_polish") or "").startswith("not_applicable"),
            {"human_boundary_polish": stage_status.get("human_boundary_polish")},
        )
        # D-030/D-035: proof this RAW actually exercised the live render +
        # PostRenderWatchListenQC + bounded physical repair service, not
        # just a bare, unchecked ffmpeg render.
        check(
            "live_render_qc_ran_against_the_real_render",
            live_render_qc.get("status") in ("PASS", "NEEDS_HUMAN_REVIEW", "SEMANTIC_MISMATCH_INVALIDATED")
            and isinstance(live_render_qc.get("render_attempt_count"), int),
            {
                "status": live_render_qc.get("status"),
                "render_attempt_count": live_render_qc.get("render_attempt_count"),
            },
        )

    failed = [c["name"] for c in checks if not c["ok"]]
    report = {
        "schema_version": "cutsell.video00.architecture_verifier.v1",
        "architecture": "clean_cut_core_v1",
        "freeze_blocked": freeze_blocked,
        "checks": checks,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "architecture_verified": len(failed) == 0,
    }
    return len(failed) == 0, report


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_video00_architecture.py RESULT_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
