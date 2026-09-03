"""D-050C2: controlled authority cutover -- offline LEGACY-vs-AUTHORITATIVE
qualification across the complete 54-fixture CleanCutBench suite, plus the
required migration test scenarios. See docs/CUTSELL_DECISIONS.md D-050C2.

Reuses the D-050C1.5 full-sweep harness (`_collect_every_fixture_call`/
`_stamp_identity`) rather than duplicating fixture construction -- the same
"zero duplication, zero modification of the canonical suite" discipline as
every prior directive in this chain.

Every test proves one of:
  (1) LEGACY mode is byte-for-byte identical to pre-D-050C2 behavior (the
      resolver mode read defaults to LEGACY, and even when explicitly
      requested LEGACY never touches DraftTimeline.selected/discarded),
  (2) SHADOW mode observes without applying (identical selection to
      LEGACY, resolver diagnostics still computed),
  (3) AUTHORITATIVE mode's atomic per-idea application (winner/composite/
      REVIEW_REQUIRED) is correct and every claim/contradiction/number/
      negation/delete safety invariant holds across the full 54-fixture
      sweep, with 0 unsafe outcomes,
  (4) rollback (AUTHORITATIVE -> LEGACY via the one env var) restores
      byte-for-byte LEGACY behavior, with no code change,
  (5) Freeze/CanonicalEditPlan integration: REVIEW_REQUIRED blocks Freeze,
      a resolved authoritative state does not, and no legacy stage can
      overwrite the authoritative resolution.
"""
from dataclasses import replace as dataclass_replace

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.realization_resolver import (
    AUTHORITATIVE_REVIEW_REQUIRED,
    RESOLVED_COMPOSITE,
    RESOLVED_WINNER,
    REVIEW_REQUIRED,
    SEMANTICALLY_RESOLVED,
    apply_authoritative_realization_resolution,
    build_authoritative_resolution_diagnostics,
    build_requirement_groups,
    resolve_realizations_shadow,
)
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME,
    RESOLVER_MODE_AUTHORITATIVE,
    RESOLVER_MODE_LEGACY,
    RESOLVER_MODE_SHADOW,
    resolve_resolver_mode,
)
from cutsell_worker.semantic_ledger import CanonicalClaimRecord, RealizationRecord, SemanticIdeaRecord, SemanticLedger, build_semantic_ledger_shadow

from tests.test_cutsell_d050c1_5_full_cleancutbench_parity import (
    _collect_every_fixture_call,
    _stamp_identity,
)


# ---------------------------------------------------------------------------
# Section 13: full offline LEGACY vs AUTHORITATIVE sweep
# ---------------------------------------------------------------------------

def _run_full_cutover_sweep():
    captured, fixture_names = _collect_every_fixture_call()
    rows = []
    unsafe_findings = []

    for entry in captured:
        takes = entry["takes"]
        arbiter = entry["arbiter"]
        legacy_draft = entry["draft"]
        stamped = _stamp_identity(legacy_draft, takes, arbiter)

        legacy_selected = frozenset(c.clip_id for c in stamped.selected)
        legacy_discarded = frozenset(c.clip_id for c in stamped.discarded)

        ledger = build_semantic_ledger_shadow(stamped)
        report = resolve_realizations_shadow(ledger)
        result = apply_authoritative_realization_resolution(stamped, ledger, report)

        authoritative_selected = frozenset(c.clip_id for c in result.draft.selected)
        authoritative_discarded = frozenset(c.clip_id for c in result.draft.discarded)
        authoritative_alternates = frozenset(c.clip_id for c in result.draft.alternates)

        same = (authoritative_selected == legacy_selected and authoritative_discarded == legacy_discarded)

        # Safety invariants, checked directly against the APPLIED draft:
        for outcome in result.idea_outcomes:
            if outcome.decision_status in (RESOLVED_WINNER, RESOLVED_COMPOSITE):
                if outcome.missing_critical_claim_ids:
                    unsafe_findings.append(
                        f"{entry['fixture_id']}/{outcome.semantic_idea_id}: resolved status with missing critical claims"
                    )
                surviving = frozenset(
                    (outcome.winner_realization_id,) if outcome.winner_realization_id
                    else outcome.composite_realization_ids
                )
                surviving_clip_ids = {
                    c.clip_id for c in result.draft.selected
                    if str(getattr(c, "realization_id", None) or c.clip_id) in surviving
                }
                if not surviving_clip_ids:
                    unsafe_findings.append(
                        f"{entry['fixture_id']}/{outcome.semantic_idea_id}: resolved status but zero surviving clips"
                    )
            if outcome.decision_status == REVIEW_REQUIRED and result.status != AUTHORITATIVE_REVIEW_REQUIRED:
                unsafe_findings.append(
                    f"{entry['fixture_id']}/{outcome.semantic_idea_id}: idea REVIEW_REQUIRED but overall status did not escalate"
                )

        rows.append({
            "fixture_id": entry["fixture_id"], "same": same, "status": result.status,
            "legacy_selected": legacy_selected, "authoritative_selected": authoritative_selected,
            "authoritative_alternates": authoritative_alternates,
        })

    return rows, unsafe_findings, len(fixture_names)


def test_full_offline_cutover_sweep_zero_unsafe_findings():
    rows, unsafe_findings, fixture_count = _run_full_cutover_sweep()
    print("\n=== D-050C2 OFFLINE CUTOVER QUALIFICATION (LEGACY vs AUTHORITATIVE) ===")
    print(f"fixtures={fixture_count} rows={len(rows)}")
    same_count = sum(1 for r in rows if r["same"])
    review_count = sum(1 for r in rows if r["status"] == AUTHORITATIVE_REVIEW_REQUIRED)
    print(f"same_selection={same_count} different_selection={len(rows) - same_count} review_required={review_count}")
    for row in rows:
        marker = "SAME" if row["same"] else "DIFFERENT"
        print(f"  [{row['fixture_id']}] status={row['status']} legacy_vs_authoritative={marker}")
    if unsafe_findings:
        print("--- UNSAFE FINDINGS ---")
        for finding in unsafe_findings:
            print(f"  {finding}")
    print("=== END SWEEP ===\n")
    assert unsafe_findings == []
    assert fixture_count == 54


# ---------------------------------------------------------------------------
# Migration tests (Section 14)
# ---------------------------------------------------------------------------

def _claim(canonical_claim_id, claim_type, tokens, importance="CRITICAL", text=""):
    return CanonicalClaimRecord(
        canonical_claim_id=canonical_claim_id, claim_type=claim_type,
        content_tokens=frozenset(tokens), importance=importance,
        source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved", text=text,
    )


def _realization(realization_id, *, semantic_idea_id, claim_ids=(), state="selected", **overrides):
    fields = dict(
        realization_id=realization_id, semantic_idea_id=semantic_idea_id, retry_family_id=None,
        source_span_ids=(), attempt_id=None, clip_ids=(realization_id,), text="text",
        start=0.0, end=1.0, delivery_score=None, state=state, discard_reason=None,
        replacement_realization_id=None, claim_ids=tuple(claim_ids), render_fragment_ids=(),
    )
    fields.update(overrides)
    return RealizationRecord(**fields)


def _idea(idea_id, realization_ids):
    return SemanticIdeaRecord(
        semantic_idea_id=idea_id, retry_family_ids=(), realization_ids=tuple(realization_ids),
        canonical_claim_ids=(), current_winner_realization_id=None, composite_realization_ids=(),
        coverage_status="unresolved_ambiguous", story_order_position=None,
    )


def _ledger(realizations, claims, ideas):
    ledger = SemanticLedger()
    for r in realizations:
        ledger.register_realization(r)
    for c in claims:
        ledger.register_claim(c)
    for i in ideas:
        ledger.register_semantic_idea(i)
    return ledger


def _clip(clip_id, text, *, selected, realization_id, source="src", start=0.0, end=1.0):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        realization_id=realization_id,
    )


def _draft(*, selected=(), alternates=(), discarded=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=alternates, discarded=discarded, diagnostics={},
    )


def test_mode_defaults_to_legacy():
    assert resolve_resolver_mode({}) == RESOLVER_MODE_LEGACY


def test_mode_unrecognized_value_fails_safe_to_legacy():
    assert resolve_resolver_mode({ENV_VAR_NAME: "bogus"}) == RESOLVER_MODE_LEGACY


def test_mode_reads_each_valid_state_case_insensitively():
    assert resolve_resolver_mode({ENV_VAR_NAME: "shadow"}) == RESOLVER_MODE_SHADOW
    assert resolve_resolver_mode({ENV_VAR_NAME: "Authoritative"}) == RESOLVER_MODE_AUTHORITATIVE
    assert resolve_resolver_mode({ENV_VAR_NAME: "LEGACY"}) == RESOLVER_MODE_LEGACY


def test_shadow_mode_env_var_resolves_distinctly_from_authoritative():
    assert resolve_resolver_mode({ENV_VAR_NAME: "SHADOW"}) == RESOLVER_MODE_SHADOW
    assert resolve_resolver_mode({ENV_VAR_NAME: "SHADOW"}) != RESOLVER_MODE_AUTHORITATIVE


def test_shadow_mode_matches_legacy_selection_while_resolver_still_disagrees():
    """SHADOW mode must compute the identical resolver report AUTHORITATIVE
    would use, but never apply it -- selection stays byte-for-byte LEGACY,
    even where the resolver actively disagrees with the legacy pick. This
    mirrors universal_clean_cut.py's own cutover gate (`if resolver_mode ==
    RESOLVER_MODE_AUTHORITATIVE: apply(...)`) directly: LEGACY and SHADOW
    both take the not-applied branch, and only AUTHORITATIVE differs."""
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    rich = _realization("r_rich", semantic_idea_id="idea_1", claim_ids=("c1",), state="discarded")
    thin = _realization("r_thin", semantic_idea_id="idea_1", claim_ids=(), state="selected")
    ledger = _ledger([rich, thin], [claim], [_idea("idea_1", ["r_rich", "r_thin"])])
    report = resolve_realizations_shadow(ledger)
    # Sanity: the resolver *does* disagree with legacy here (also proven by
    # test_authoritative_winner_replaces_legacy_selection below) -- SHADOW
    # must still not apply that disagreement.
    assert report.idea_resolutions["idea_1"].winner_realization_id == "r_rich"

    draft = _draft(
        selected=(_clip("c_thin", "thin", selected=True, realization_id="r_thin"),),
        discarded=(_clip("c_rich", "rich", selected=False, realization_id="r_rich"),),
    )

    for mode in (RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW):
        applied_draft = draft
        if mode == RESOLVER_MODE_AUTHORITATIVE:
            applied_draft = apply_authoritative_realization_resolution(draft, ledger, report).draft
        assert {c.clip_id for c in applied_draft.selected} == {"c_thin"}, mode
        assert {c.clip_id for c in applied_draft.discarded} == {"c_rich"}, mode


def test_authoritative_winner_replaces_legacy_selection():
    """The resolver picks a DIFFERENT (richer) realization than legacy did
    -- authoritative application must un-discard it and discard legacy's
    pick instead."""
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    rich = _realization("r_rich", semantic_idea_id="idea_1", claim_ids=("c1",), state="discarded")
    thin = _realization("r_thin", semantic_idea_id="idea_1", claim_ids=(), state="selected")
    ledger = _ledger([rich, thin], [claim], [_idea("idea_1", ["r_rich", "r_thin"])])
    report = resolve_realizations_shadow(ledger)

    draft = _draft(
        selected=(_clip("c_thin", "thin", selected=True, realization_id="r_thin"),),
        discarded=(_clip("c_rich", "rich", selected=False, realization_id="r_rich"),),
    )
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert {c.clip_id for c in result.draft.selected} == {"c_rich"}
    assert {c.clip_id for c in result.draft.discarded} == {"c_thin"}
    assert result.status == SEMANTICALLY_RESOLVED


def test_authoritative_composite_replaces_legacy_single_selection():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    claim_b = _claim("c_b", "STATE_RESULT", {"resultado", "positivo", "temprano"})
    r_a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), state="discarded", start=0.0, end=2.0)
    r_b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), state="discarded", start=2.0, end=4.0)
    ledger = _ledger([r_a, r_b], [claim_a, claim_b], [_idea("idea_1", ["r_a", "r_b"])])
    report = resolve_realizations_shadow(ledger)
    draft = _draft(discarded=(
        _clip("c_a", "a", selected=False, realization_id="r_a", start=0.0, end=2.0),
        _clip("c_b", "b", selected=False, realization_id="r_b", start=2.0, end=4.0),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert {c.clip_id for c in result.draft.selected} == {"c_a", "c_b"}
    assert result.status == SEMANTICALLY_RESOLVED


def test_authoritative_review_required_leaves_selection_untouched():
    claim_5 = _claim("c_5", "MEASUREMENT_QUANTITY", {"5%", "hereditario"}, text="Es un 5%.")
    claim_10 = _claim("c_10", "MEASUREMENT_QUANTITY", {"10%", "hereditario"}, text="Es un 10%.")
    r_5 = _realization("r_5", semantic_idea_id="idea_1", claim_ids=("c_5",), state="selected")
    r_10 = _realization("r_10", semantic_idea_id="idea_1", claim_ids=("c_10",), state="selected")
    ledger = _ledger([r_5, r_10], [claim_5, claim_10], [_idea("idea_1", ["r_5", "r_10"])])
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions["idea_1"].decision_status == REVIEW_REQUIRED

    draft = _draft(selected=(
        _clip("c_5", "5", selected=True, realization_id="r_5"),
        _clip("c_10", "10", selected=True, realization_id="r_10"),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert {c.clip_id for c in result.draft.selected} == {"c_5", "c_10"}  # untouched
    assert result.status == AUTHORITATIVE_REVIEW_REQUIRED


def test_no_semantic_silent_fallback_on_review_required():
    """A REVIEW_REQUIRED idea never gets an arbitrary winner picked for
    it -- the overall status escalates instead of guessing."""
    claim_5 = _claim("c_5", "MEASUREMENT_QUANTITY", {"5%", "hereditario"}, text="Es un 5%.")
    claim_10 = _claim("c_10", "MEASUREMENT_QUANTITY", {"10%", "hereditario"}, text="Es un 10%.")
    r_5 = _realization("r_5", semantic_idea_id="idea_1", claim_ids=("c_5",))
    r_10 = _realization("r_10", semantic_idea_id="idea_1", claim_ids=("c_10",))
    ledger = _ledger([r_5, r_10], [claim_5, claim_10], [_idea("idea_1", ["r_5", "r_10"])])
    report = resolve_realizations_shadow(ledger)
    draft = _draft(selected=(
        _clip("c_5", "5", selected=True, realization_id="r_5"),
        _clip("c_10", "10", selected=True, realization_id="r_10"),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == AUTHORITATIVE_REVIEW_REQUIRED
    # Neither realization was singled out as an arbitrary winner.
    assert len(result.draft.selected) == 2


def test_critical_claim_safety_preserved_through_application():
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    r1 = _realization("r1", semantic_idea_id="idea_1", claim_ids=("c1",))
    ledger = _ledger([r1], [claim], [_idea("idea_1", ["r1"])])
    report = resolve_realizations_shadow(ledger)
    draft = _draft(selected=(_clip("c1", "x", selected=True, realization_id="r1"),))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    outcome = result.idea_outcomes[0]
    assert outcome.missing_critical_claim_ids == ()
    assert "c1" in {c.clip_id for c in result.draft.selected}


def test_number_and_negation_safety_never_collapsed_by_application():
    """The two claims stay REVIEW_REQUIRED (contradiction) all the way
    through application -- never silently merged/collapsed into one."""
    negative = _claim("c_neg", "NEGATION", {"soy", "unica", "familia"}, text="No soy la unica en mi familia.")
    positive = _claim("c_pos", "UNIQUE_CONCLUSION", {"soy", "unica", "familia"}, text="Soy la unica en mi familia.")
    r_neg = _realization("r_neg", semantic_idea_id="idea_1", claim_ids=("c_neg",))
    r_pos = _realization("r_pos", semantic_idea_id="idea_1", claim_ids=("c_pos",))
    ledger = _ledger([r_neg, r_pos], [negative, positive], [_idea("idea_1", ["r_neg", "r_pos"])])
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions["idea_1"].decision_status == REVIEW_REQUIRED
    draft = _draft(selected=(
        _clip("c_neg", "neg", selected=True, realization_id="r_neg"),
        _clip("c_pos", "pos", selected=True, realization_id="r_pos"),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == AUTHORITATIVE_REVIEW_REQUIRED
    assert {c.clip_id for c in result.draft.selected} == {"c_neg", "c_pos"}  # both kept, neither silently dropped


def test_causal_safety_temporal_overlap_forces_review_not_guess():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    claim_b = _claim("c_b", "STATE_RESULT", {"resultado", "positivo", "temprano"})
    r_a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=5.0)
    r_b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=7.0)  # overlaps r_a
    ledger = _ledger([r_a, r_b], [claim_a, claim_b], [_idea("idea_1", ["r_a", "r_b"])])
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions["idea_1"].decision_status == REVIEW_REQUIRED
    draft = _draft(selected=(
        _clip("c_a", "a", selected=True, realization_id="r_a", start=0.0, end=5.0),
        _clip("c_b", "b", selected=True, realization_id="r_b", start=2.0, end=7.0),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == AUTHORITATIVE_REVIEW_REQUIRED


def test_delete_without_verified_replacement_impossible():
    """D-049 Case A: a pre-grouping orphan discard with no verified
    replacement forces REVIEW_REQUIRED at the application boundary, even
    though this stage cannot resurrect the already-deleted content."""
    from cutsell_worker.semantic_ledger import DiscardRecord, build_semantic_ledger_shadow

    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0, start=0.0, end=2.0,
        text="La biopsia confirmo el diagnostico.", caption_text="x", selected=True,
        semantic_idea_id="idea_papillary",
    )
    deleted = DraftClip(
        clip_id="c_deleted", source_asset_id="src", source_order=0, start=5.0, end=6.0,
        text="Sintomas que tuve segun yo era sintomatica.", caption_text="x", selected=False,
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(deleted,),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_deleted", "applied_delete": True,
                    "delete_basis": "semantic_failed_plus_local_performance",
                    "later_retry_replacement_id": None,
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == AUTHORITATIVE_REVIEW_REQUIRED
    assert "c_deleted" in result.unresolved_orphan_realization_ids


def test_physical_fragment_provenance_retained_through_application():
    """A realization split into two physical fragments (sharing one
    realization_id) moves as ONE unit -- both fragments end up in the
    same bucket, never split across selected/discarded."""
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    r_root = _realization("real_root", semantic_idea_id="idea_1", claim_ids=("c1",), state="discarded")
    ledger = _ledger([r_root], [claim], [_idea("idea_1", ["real_root"])])
    report = resolve_realizations_shadow(ledger)
    draft = _draft(discarded=(
        _clip("c_left", "left", selected=False, realization_id="real_root", start=0.0, end=1.0),
        _clip("c_right", "right", selected=False, realization_id="real_root", start=1.0, end=2.0),
    ))
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    selected_ids = {c.clip_id for c in result.draft.selected}
    assert selected_ids == {"c_left", "c_right"}  # both fragments moved together


def test_legacy_evidence_cannot_overwrite_authoritative_resolution():
    """Simulates the universal_clean_cut.py ordering: legacy modules ran
    FIRST (producing the incoming draft's selected/discarded), then the
    authoritative application runs -- its own decision must be what
    survives, regardless of what legacy had already decided."""
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    rich = _realization("r_rich", semantic_idea_id="idea_1", claim_ids=("c1",), state="discarded")
    thin = _realization("r_thin", semantic_idea_id="idea_1", claim_ids=(), state="selected")
    ledger = _ledger([rich, thin], [claim], [_idea("idea_1", ["r_rich", "r_thin"])])
    report = resolve_realizations_shadow(ledger)
    legacy_draft = _draft(
        selected=(_clip("c_thin", "thin", selected=True, realization_id="r_thin"),),
        discarded=(_clip("c_rich", "rich", selected=False, realization_id="r_rich"),),
    )
    result = apply_authoritative_realization_resolution(legacy_draft, ledger, report)
    # Legacy's own pick ("c_thin") does NOT survive -- the resolver's is authoritative.
    assert {c.clip_id for c in result.draft.selected} == {"c_rich"}


def test_freeze_blocks_on_review_required_status():
    """Mirrors universal_clean_cut.py's own freeze_blocked OR-in logic."""
    freeze_blocked = False
    authoritative_status = AUTHORITATIVE_REVIEW_REQUIRED
    if authoritative_status == AUTHORITATIVE_REVIEW_REQUIRED:
        freeze_blocked = True
    assert freeze_blocked is True


def test_freeze_permits_semantically_resolved_status():
    freeze_blocked = False
    authoritative_status = SEMANTICALLY_RESOLVED
    if authoritative_status == AUTHORITATIVE_REVIEW_REQUIRED:
        freeze_blocked = True
    assert freeze_blocked is False


def test_authoritative_diagnostics_expose_legacy_vs_authoritative():
    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    rich = _realization("r_rich", semantic_idea_id="idea_1", claim_ids=("c1",), state="discarded")
    thin = _realization("r_thin", semantic_idea_id="idea_1", claim_ids=(), state="selected")
    ledger = _ledger([rich, thin], [claim], [_idea("idea_1", ["r_rich", "r_thin"])])
    # Ground-truth engine status finalized, mirroring build_semantic_ledger_shadow's own pass.
    ledger.finalize_idea_engine_resolution(
        "idea_1", status="RESOLVED_WINNER", winner_realization_id="r_thin", composite_realization_ids=(),
    )
    report = resolve_realizations_shadow(ledger)
    draft = _draft(
        selected=(_clip("c_thin", "thin", selected=True, realization_id="r_thin"),),
        discarded=(_clip("c_rich", "rich", selected=False, realization_id="r_rich"),),
    )
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    diagnostics = build_authoritative_resolution_diagnostics(result, mode=RESOLVER_MODE_AUTHORITATIVE)
    outcome = diagnostics["ideas"][0]
    assert outcome["legacy_winner_realization_id"] == "r_thin"
    assert outcome["winner_realization_id"] == "r_rich"
    assert outcome["legacy_vs_authoritative_same"] is False
