"""D-097.B -- ALL-FAILED FAMILY: no forced winner, no label-only deletion.

Product Owner (post-D-096 approval + adjustment §1): FAILED + FAILED + FAILED
must not make Best Take elect a "least bad" survivor by delivery tie-break;
but "all failed" is NOT permission to delete a family on labels alone. The
family yields no winner only when every member ALSO carries deterministic
unusability evidence (a ranker fragment penalty relative to a sibling, or
D-081 local-performance corroboration); a member without such evidence
competes normally and the label conflict is recorded (routed), never
silently trusted. A dropped family is a recorded decision: the Resolver
answers RESOLVED_NONE (never restores a failed attempt), the plan/reviewer
carry a non-blocking NO_USABLE_REALIZATION warning, StoryValidator records
the lost content instead of blocking, and the run is story-incomplete --
the candidate is never presented as a clean complete story.
Generic fixtures only.
"""
from cutsell_worker.canonical_edit_plan import (
    COVERAGE_DROPPED_NO_USABLE_REALIZATION,
    build_authoritative_plan_source,
    build_canonical_edit_plan,
)
from cutsell_worker.composer import compose_selected
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, RankedTake, SCHEMA_VERSION, TakeGroup,
)
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.final_edit_reviewer import NO_USABLE_REALIZATION, review
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.live_render_qc import LiveRenderQCResult
from cutsell_worker.pipeline import _semantic_best_take
from cutsell_worker.realization_resolver import (
    RESOLVED_NONE,
    SEMANTICALLY_RESOLVED,
    apply_authoritative_realization_resolution,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import (
    ENGINE_NO_USABLE_REALIZATION,
    NO_USABLE_REALIZATION_DISCARD_REASON,
    build_semantic_ledger_shadow,
)
from cutsell_worker.universal_clean_cut_validation import _live_render_qc_diagnostics

IDEA = "idea_generic_failed_family"


def take(clip_id, text, *, start=0.0, complete_idea=True):
    return CandidateTake(clip_id, "src", 0, start, start + 4.0, text, complete_idea=complete_idea)


def ranked(*pairs):
    return tuple(RankedTake(cid, score, reason) for cid, score, reason in pairs)


TEXT_A = "I tried to explain the routine but I lost my train of thought halfway through it"
TEXT_B = "I tried to explain the routine but"
TEXT_C = "I tried to explain the routine again and again and again and again"


# --- Best Take (draft level) -------------------------------------------------

def test_all_failed_and_all_unusable_yields_no_winner():
    members = (take("a", TEXT_A), take("b", TEXT_B, start=5.0), take("c", TEXT_C, start=10.0))
    labels = {"a": ("failed", 0.90), "b": ("failed", 0.95), "c": ("failed", 0.92)}
    r = ranked(("a", 0.62, "watch_listen_baseline"), ("b", 0.40, "watch_listen_baseline+material_prefix_fragment_penalty"),
               ("c", 0.35, "watch_listen_baseline+repetitive_restart_fragment_penalty"))
    selected, preferred, reason = _semantic_best_take(
        members, labels, "a", r,
        semantic_delete_recommended={"a": True, "b": True, "c": True},
        deterministic_unusable={"a": True, "b": True, "c": True},
    )
    assert (selected, preferred, reason) == (None, None, "no_usable_realization")


def test_all_failed_but_one_member_without_deterministic_evidence_competes_and_wins():
    members = (take("a", TEXT_A), take("b", TEXT_B, start=5.0))
    labels = {"a": ("failed", 0.90), "b": ("failed", 0.95)}
    r = ranked(("a", 0.62, "watch_listen_baseline"), ("b", 0.40, "watch_listen_baseline+material_prefix_fragment_penalty"))
    selected, _preferred, reason = _semantic_best_take(
        members, labels, "a", r,
        semantic_delete_recommended={"a": True, "b": True},
        deterministic_unusable={"a": False, "b": True},
    )
    assert selected == "a" and reason != "no_usable_realization"


def test_labels_alone_never_drop_a_family():
    members = (take("a", TEXT_A), take("b", TEXT_B, start=5.0))
    labels = {"a": ("failed", 0.99), "b": ("failed", 0.99)}
    r = ranked(("a", 0.62, "watch_listen_baseline"), ("b", 0.40, "watch_listen_baseline"))
    selected, _p, _reason = _semantic_best_take(
        members, labels, "a", r,
        semantic_delete_recommended={"a": True, "b": True},
        deterministic_unusable={},  # no objective evidence at all
    )
    assert selected == "a"


def test_not_all_failed_keeps_the_d082_ladder_unchanged():
    members = (take("a", TEXT_A), take("b", TEXT_B, start=5.0))
    labels = {"a": ("failed", 0.90), "b": ("winner", 0.95)}
    r = ranked(("a", 0.62, "watch_listen_baseline"), ("b", 0.40, "watch_listen_baseline"))
    selected, _p, reason = _semantic_best_take(
        members, labels, "a", r,
        semantic_delete_recommended={"a": True, "b": False},
        deterministic_unusable={"a": True, "b": True},
    )
    assert selected == "b" and reason == "single_semantic_winner"


def test_a_single_member_never_reaches_the_no_usable_outcome():
    members = (take("a", TEXT_A),)
    selected, _p, reason = _semantic_best_take(
        members, {"a": ("failed", 0.99)}, "a", ranked(("a", 0.5, "watch_listen_baseline")),
        semantic_delete_recommended={"a": True}, deterministic_unusable={"a": True},
    )
    assert selected == "a" and reason == "single_member_no_contest"


def test_compose_selected_ignores_a_family_with_no_winner():
    a, b, solo = take("a", TEXT_A), take("b", TEXT_B, start=5.0), take("s", "Something completely different here.", start=20.0)
    group = TakeGroup(group_id="g", semantic_key="k", candidate_ids=("a", "b"), ranked=ranked(("a", 0.6, "x"), ("b", 0.4, "x")), selected_clip_id="")
    assert [t.clip_id for t in compose_selected((a, b, solo), (group,), ())] == ["s"]


# --- authority level -----------------------------------------------------------

def _clip(clip_id, text, *, start, end, selected, idea=IDEA):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected, realization_id=f"real_{clip_id}",
        semantic_idea_id=idea, retry_family_id=idea, complete_idea=True,
    )


def _group(no_usable=True):
    return {
        "group_id": "g_failed", "selected_clip_id": "", "local_selected_clip_id": "a",
        "semantic_override_applied": False, "semantic_best_take_reason": "no_usable_realization",
        "semantic_candidates": [
            {"clip_id": "a", "label": "failed", "confidence": 0.90},
            {"clip_id": "b", "label": "failed", "confidence": 0.95},
        ],
        "ranked": [
            {"clip_id": "a", "score": 0.62, "reason": "watch_listen_baseline+restart_tail_fragment_penalty"},
            {"clip_id": "b", "score": 0.30, "reason": "watch_listen_baseline+material_prefix_fragment_penalty"},
        ],
        "no_usable_realization": no_usable,
        "member_usability": {"a": {"deterministic_unusable": True}, "b": {"deterministic_unusable": True}},
    }


def _draft():
    keep = _clip("k", "The product arrives in two days and the first week is free of charge.", start=0.0, end=5.0, selected=True, idea="idea_other")
    a = _clip("a", TEXT_A, start=10.0, end=14.0, selected=False)
    b = _clip("b", TEXT_B, start=15.0, end=17.0, selected=False)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(keep,), alternates=(), discarded=(a, b),
        diagnostics={"take_judge_groups": [_group()], "hybrid_editorial_chunks": [],
                     "final_story_coherence_validation": {"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []}},
    )


def test_deterministic_authority_never_reselects_from_a_dropped_family():
    draft = _draft()  # score gap 0.32 >= CLEAR_WINNER_MINIMUM_GAP
    after = apply_deterministic_best_take_authority(draft, swap_enabled=False)
    assert [c.clip_id for c in after.selected] == ["k"]
    assert "deterministic_best_take_authority" not in (after.diagnostics or {})


def test_ledger_records_the_drop_as_a_decision_not_an_accident():
    ledger = build_semantic_ledger_shadow(_draft())
    idea = ledger.ideas()[IDEA]
    assert idea.no_usable_realization is True
    assert idea.coverage_status == "no_usable_realization"
    assert idea.engine_resolution_status == ENGINE_NO_USABLE_REALIZATION
    for rid in ("real_a", "real_b"):
        assert ledger.realizations()[rid].discard_reason == NO_USABLE_REALIZATION_DISCARD_REASON
        assert ledger.realizations()[rid].semantic_label == "failed"


def test_resolver_confirms_the_drop_and_restores_nothing():
    draft = _draft()
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions[IDEA].decision_status == RESOLVED_NONE
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    assert applied.status == SEMANTICALLY_RESOLVED
    assert [c.clip_id for c in applied.draft.selected] == ["k"]
    assert {c.clip_id for c in applied.draft.discarded} == {"a", "b"}
    outcome = next(o for o in applied.idea_outcomes if o.semantic_idea_id == IDEA)
    assert outcome.decision_status == RESOLVED_NONE and outcome.legacy_vs_authoritative_same


def test_plan_and_reviewer_record_a_non_blocking_warning():
    draft = _draft()
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    source = build_authoritative_plan_source(applied, ledger)
    plan = build_canonical_edit_plan(applied.draft, authoritative_source=source)
    idea = next(i for i in plan.ideas if i.idea_id == "g_failed")
    assert idea.coverage_status == COVERAGE_DROPPED_NO_USABLE_REALIZATION
    result = review(plan)
    assert result.status == "PASS"
    assert [w.kind for w in result.warnings] == [NO_USABLE_REALIZATION]
    assert not [f for f in result.findings if f.blocking]
    # legacy path (no authoritative source) records the same coverage
    legacy_plan = build_canonical_edit_plan(draft)
    assert next(i for i in legacy_plan.ideas if i.idea_id == "g_failed").coverage_status == COVERAGE_DROPPED_NO_USABLE_REALIZATION


def test_story_validator_records_the_loss_without_blocking_freeze():
    validated = apply_final_story_coherence_validation(_draft())
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["missing_idea_coverage"] == []
    assert [row["group_id"] for row in diag["no_usable_realization_families"]] == ["g_failed"]
    recorded = [row for row in diag["lost_semantic_atoms"] if row.get("kind") == "LOST_IN_NO_USABLE_REALIZATION_FAMILY"]
    assert {row["clip_id"] for row in recorded} == {"a", "b"}
    assert all(row["blocking"] is False for row in recorded)


def test_incomplete_story_is_never_deliverable_even_when_technical_qc_passes():
    qc = LiveRenderQCResult(status="PASS", output_path="/tmp/x.mp4", plan_id="p", plan_version=1, semantic_hash="h", attempts=())
    complete = _live_render_qc_diagnostics(qc, skipped_reason=None, story_completeness="complete")
    assert complete["deliverable"] is True and complete["delivery_status"] == "DELIVERABLE"
    incomplete = _live_render_qc_diagnostics(qc, skipped_reason=None, story_completeness="incomplete_no_usable_realization")
    assert incomplete["deliverable"] is False
    assert incomplete["delivery_status"].startswith("NOT_DELIVERABLE_INCOMPLETE_STORY_REVIEW")
