"""D-048 FIX 2 -- claim-criticality gate for the ClaimCoverageBestTake
single-candidate override.

D-047 Case 2: a retry family's only extracted CRITICAL claim was a bare
negation riding on an incidental year inside an ordinary temporal aside
("... en una temporada, en 2023, no hay que preguntar."), source-exclusive
to the one thin candidate that happened to contain that exact wording.
Because that candidate trivially "covered every critical claim" (the set
had exactly one member, and it was its own), claim_coverage_best_take.py
swapped it in as the new winner over an already-correct, substantively
richer realization that had the actual diagnosis/treatment content -- a
self-source-exclusivity trap, not a real coverage gap.

The fix does not disable ClaimCoverageBestTake or change classify_claim's
own importance labels (a negation is still always CRITICAL everywhere else
in this codebase, including StoryValidator's freeze-blocking posture,
unchanged). It adds a second, narrower gate scoped to this module's own
override decision: a missing claim only blocks the override from applying
when it is (1) source-exclusive (no OTHER sibling covers it either), (2)
itself low-information/incidental (no independently substantive marker of
its own, and its only CRITICAL signal is a bare negation/number riding on
a recognizable temporal-aside shape -- mirrors classify_number_atom's own
D-031 CONTEXTUAL rule for a bare year), and (3) the candidate is not
otherwise richer than the current winner. A single substantive,
non-source-exclusive, or genuinely-richer-candidate claim keeps the
override eligible, exactly as before.
"""
from cutsell_worker.claim_coverage_best_take import apply_claim_coverage_best_take
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), alternates=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=alternates, discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


def ranked_row(clip_id, score, reason="watch_listen_baseline"):
    return {"clip_id": clip_id, "score": score, "reason": reason}


# A richer winner with NO critical claim of its own -- mirrors the real
# incident exactly (neither sibling registered a critical claim there
# either; the group's only critical claim was the incidental one).
_RICHER_WINNER_TEXT = (
    "Después de eso empecé a sentirme mucho mejor y pude retomar mi rutina "
    "diaria con tranquilidad."
)


def _run(candidate_text, *, winner_text=_RICHER_WINNER_TEXT):
    winner = clip("winner", 0.0, 5.0, winner_text, selected=True)
    candidate = clip("candidate", 5.0, 10.0, candidate_text, selected=False)
    d = draft(
        selected=(winner,), discarded=(candidate,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("candidate", 0.5)],
        }],
    )
    return apply_claim_coverage_best_take(d)


def _winner_kept(out) -> bool:
    return [c.clip_id for c in out.selected] == ["winner"]


# 1 & 2. Contextual year/date + incidental temporal aside -- does NOT override.

def test_contextual_year_unique_to_loser_does_not_override_richer_winner():
    out = _run("Tuve molestias durante una temporada, en 2015, no sé por qué me pasaba eso.")
    assert _winner_kept(out)
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["overrides"] == []
    assert len(diag["suppressed_incidental_overrides"]) == 1
    assert diag["suppressed_incidental_overrides"][0]["suppressed_new_winner_clip_id"] == "candidate"


def test_incidental_temporal_aside_does_not_override():
    out = _run("Sentí algo extraño por un tiempo, durante 2016, no sé qué me pasaba realmente.")
    assert _winner_kept(out)


# 3. Filler/self-referential aside (same shape: incidental temporal aside
#    carrying the only critical marker) -- does NOT override.

def test_filler_self_referential_aside_does_not_override():
    out = _run("Tuve algo raro en 2018, no sé, fue rarísimo de verdad.")
    assert _winner_kept(out)


# 4. Unique substantive diagnosis DOES override when truly missing.

def test_unique_substantive_diagnosis_overrides_when_truly_missing():
    out = _run("El médico confirmó que era una gastritis y me recetó tratamiento por tres meses.")
    assert [c.clip_id for c in out.selected] == ["candidate"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert len(diag["overrides"]) == 1
    assert diag["overrides"][0]["new_winner_clip_id"] == "candidate"
    assert diag.get("suppressed_incidental_overrides", []) == []


# 5. Unique negation (NOT wrapped in an incidental temporal aside) DOES
#    override when truly missing.

def test_unique_negation_overrides_when_truly_missing():
    out = _run("Nunca me dijeron que tenía algo grave en los examenes.")
    assert [c.clip_id for c in out.selected] == ["candidate"]


# 6. Unique causal/result claim (reported via explicit result-determining
#    language) DOES override when truly missing.

def test_unique_causal_result_claim_overrides_when_truly_missing():
    out = _run("El médico determinó que la causa fue una infección seria en el intestino.")
    assert [c.clip_id for c in out.selected] == ["candidate"]


# 7. Unique treatment fact (a dosed/quantified measurement) can override.

def test_unique_treatment_fact_can_override():
    out = _run("Me recetaron 3 pastillas al día durante un mes para tratar la infección.")
    assert [c.clip_id for c in out.selected] == ["candidate"]


# 8. Unique family/hereditary fact (a unique-conclusion statistic) can override.

def test_unique_family_hereditary_fact_can_override():
    out = _run(
        "Soy la única persona de mi familia con este tipo de cáncer, y solo "
        "1 de cada 20 personas lo desarrolla."
    )
    assert [c.clip_id for c in out.selected] == ["candidate"]


# 9. Contextual claim plus an already-richer current winner -- winner is preserved.

def test_contextual_claim_with_richer_current_winner_preserves_winner():
    out = _run(
        "Tuve molestias durante una temporada, en 2019, no sé por qué me pasaba eso.",
        winner_text=(
            "El médico confirmó que era una gastritis, me recetó tratamiento y "
            "me explicó todo con calma durante la consulta."
        ),
    )
    assert _winner_kept(out)


# 10. Multiple critical claims genuinely better covered by an alternate --
#     override still works (not source-exclusive/incidental in this shape).

def test_multiple_critical_claims_better_covered_by_alternate_still_overrides():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall, nothing more to add.", selected=True)
    alt = clip(
        "alt", 5.0, 10.0,
        "El médico confirmó que era una gastritis y me recetó 3 pastillas al día.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(alt,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("alt", 0.5)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert [c.clip_id for c in out.selected] == ["alt"]


# 11. A source-exclusive but genuinely CRITICAL claim remains eligible
#     (source-exclusivity alone is never disqualifying).

def test_source_exclusive_critical_claim_remains_eligible():
    # "candidate" is the ONLY member covering this claim (source-exclusive)
    # but the claim itself is substantive (diagnosis language) -- must
    # still override.
    out = _run("El médico confirmó que era una gastritis grave que requería atención inmediata.")
    assert [c.clip_id for c in out.selected] == ["candidate"]


# 12. A source-exclusive CONTEXTUAL/incidental claim is not winner-forcing
#     (restates test 1's point explicitly against the "source-exclusive"
#     framing named in the D-048 directive).

def test_source_exclusive_contextual_claim_is_not_winner_forcing():
    out = _run("Tuve molestias durante una temporada, en 2020, no sé por qué me pasaba eso.")
    assert _winner_kept(out)


# 13. No override if the new winner would lose more critical/substantive
#     content than it restores (richer-content guard).

def test_no_override_when_new_winner_would_lose_more_content_than_it_restores():
    # The candidate's only unique claim is incidental, AND it is far
    # thinner than the current winner -- both the incidental-content and
    # the richer-content conditions point the same way.
    out = _run(
        "Durante 2021 tuve algo, no sé qué fue.",
        winner_text=(
            "Después de eso empecé a sentirme mucho mejor, pude retomar mi rutina "
            "diaria con tranquilidad y disfrutar de mi tiempo libre sin preocupaciones."
        ),
    )
    assert _winner_kept(out)


# 14. Previous correct BestTake ranking remains untouched when no critical
#     gap exists at all (no critical claims anywhere in the group).

def test_no_critical_gap_leaves_ranking_untouched():
    winner = clip("winner", 0.0, 5.0, "I walked into the room and looked around for a while.", selected=True)
    other = clip("other", 5.0, 10.0, "I stood near the door for a bit before leaving.", selected=False)
    d = draft(
        selected=(winner,), discarded=(other,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("other", 0.4)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


# --- Regression lock: the literal D-047 Case 2 incident ------------------
#
# Offline replay of the exact real shape observed in the D-046 confirmatory
# Modal Video00 result (benchmark_id video00-modal-33669148915-1),
# reconstructed from the real, unmasked diagnostics fetched via the
# read-only cutsell-video00-d044-forensic-extract.yml workflow -- not a
# live pipeline re-run. Before D-048 FIX 2, this exact group produced the
# real override recorded there: previous_winner_clip_id=fa663079e3014bb16c76
# (the gold-matching diagnosis/treatment text), new_winner_clip_id=
# 26d546bc38d3e27029b9 (the vague, source-exclusive incidental aside).

def test_d047_case2_incident_no_longer_overrides_the_diagnosis_winner():
    real_winner_id = "clip_fa663079e3014bb16c76"
    real_sibling_id = "clip_a8d998331786513e565d"
    real_incidental_id = "clip_26d546bc38d3e27029b9"
    winner = clip(
        real_winner_id, 0.0, 5.0,
        "Tuve problemas de digestión en donde me hicieron una endoscopía y "
        "dijeron que tenía gastritis. Nada severo pero tenía gastritis y me "
        "mandaron tres meses con pastillas.",
        selected=True,
    )
    sibling = clip(
        real_sibling_id, 5.0, 10.0,
        "Tuve problemas estomacales a un tiempo en donde se me hizo una "
        "endoscopía y me diagnosticaron con.",
        selected=False,
    )
    incidental = clip(
        real_incidental_id, 10.0, 15.0,
        "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(sibling, incidental),
        take_judge_groups=[{
            "group_id": "tg_54163e61976218f589",
            "ranked": [
                ranked_row(real_winner_id, 0.9),
                ranked_row(real_sibling_id, 0.6),
                ranked_row(real_incidental_id, 0.5),
            ],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == [real_winner_id]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["overrides"] == []
    assert len(diag["suppressed_incidental_overrides"]) == 1
    assert diag["suppressed_incidental_overrides"][0]["suppressed_new_winner_clip_id"] == real_incidental_id
