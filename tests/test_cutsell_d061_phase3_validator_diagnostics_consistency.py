"""D-061 Phase 3 -- VALIDATOR DIAGNOSTICS CONSISTENCY.

StoryValidator/FinalEditReviewer must distinguish REAL_CONTENT_LOSS from
SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION and never silently hide the
evidence that a finding was suppressed.

- `lost_semantic_atoms` rows carry a `classification` field
  (REAL_CONTENT_LOSS / SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION) and,
  when same-idea credit suppressed the broader content-loss signal, a
  `content_loss_suppressed_by` reason -- covered directly by
  test_cutsell_d061_phase1_same_idea_paraphrase_credit.py.
- `claim_coverage_confirmations` is a new, additive, observability-only
  diagnostics list: every claim the ambiguous band resolved to covered via
  `claim_equivalence_arbiter` (the only way that band can ever resolve to
  covered) is recorded with `resolution: "claim_equivalence_arbiter_
  confirmed"`, WITHOUT being added to `lost_critical_claims` (which would
  incorrectly block Freeze -- every row in that list is unconditionally
  blocking by construction) and WITHOUT changing `freeze_blocked` at all.

Entirely generic -- no Video00 clip ids or phrases.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


class _AlwaysCoveredArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        return True, 0.9, "paraphrase confirmed"


def test_ambiguous_claim_confirmed_by_arbiter_is_recorded_not_hidden():
    winner = clip(
        "winner", 0.0, 5.0,
        "About 5 to 10 percent of cancers are hereditary, according to her doctor.",
        selected=True,
    )
    loser = clip(
        "loser", 5.0, 10.0,
        "Only 5 to 10 percent of these cases are hereditary in nature.",
        selected=False,
    )
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(loser,),
        diagnostics={"take_judge_groups": [
            {"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]},
        ]},
    )
    out = apply_final_story_coherence_validation(d, claim_equivalence_arbiter=_AlwaysCoveredArbiter())
    diag = out.diagnostics["final_story_coherence_validation"]

    # Never silently hidden: the evidence that suppressed this claim appears
    # explicitly, even though it never became a blocking finding.
    assert len(diag["claim_coverage_confirmations"]) == 1
    confirmation = diag["claim_coverage_confirmations"][0]
    assert confirmation["resolution"] == "claim_equivalence_arbiter_confirmed"
    assert confirmation["idea_id"] == "g1"
    assert confirmation["winning_clip_ids"] == ["winner"]

    # Not a blocking finding -- distinct from REAL_CONTENT_LOSS.
    assert diag["lost_critical_claims"] == []


def test_confidently_covered_claim_has_no_confirmation_entry():
    """A claim covered by raw overlap alone (coverage >= COVERAGE_THRESHOLD,
    arbiter never consulted) has nothing suppressed -- it is correctly NOT
    recorded in claim_coverage_confirmations (there is no hidden evidence to
    surface for a claim that was always, obviously, covered)."""
    winner = clip(
        "winner", 0.0, 5.0,
        "Only 5 to 10 percent of these cases are hereditary in nature, per the study.",
        selected=True,
    )
    loser = clip(
        "loser", 5.0, 10.0,
        "Only 5 to 10 percent of these cases are hereditary in nature.",
        selected=False,
    )
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(loser,),
        diagnostics={"take_judge_groups": [
            {"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]},
        ]},
    )
    out = apply_final_story_coherence_validation(d)  # no arbiter needed at all
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["lost_critical_claims"] == []
    assert diag["claim_coverage_confirmations"] == []


def test_confirmations_never_affect_freeze_blocked():
    """Confirming coverage for one claim must never mask an UNRELATED,
    genuine blocker elsewhere in the same draft -- the confirmations list
    is purely additive/observability, never wired into freeze_blocked."""
    winner = clip(
        "winner", 0.0, 5.0,
        "About 5 to 10 percent of cancers are hereditary, according to her doctor.",
        selected=True,
    )
    loser = clip(
        "loser", 5.0, 10.0,
        "Only 5 to 10 percent of these cases are hereditary in nature.",
        selected=False,
    )
    # A second, unrelated group whose entire idea vanished -- a genuine,
    # independent blocker (_missing_idea_coverage).
    orphan_loser = clip("orphan_loser", 20.0, 25.0, "This was a completely separate point about something else.", selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(loser, orphan_loser),
        diagnostics={"take_judge_groups": [
            {"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]},
            {"group_id": "g2", "ranked": [ranked_row("orphan_loser", 0.4), ranked_row("orphan_loser2", 0.3)]},
        ]},
    )
    out = apply_final_story_coherence_validation(d, claim_equivalence_arbiter=_AlwaysCoveredArbiter())
    diag = out.diagnostics["final_story_coherence_validation"]

    assert len(diag["claim_coverage_confirmations"]) == 1
    assert diag["missing_idea_coverage"]  # g2 genuinely vanished
    assert diag["freeze_blocked"] is True
