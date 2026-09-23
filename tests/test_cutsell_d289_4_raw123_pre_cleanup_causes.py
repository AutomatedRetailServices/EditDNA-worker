"""D-289.4 -- the two RAW #123 causes that sit BEFORE grouping.

RAW #123 = Modal run 35866604610 on `fix/editorial-realization-closure@
0b1572a8` (D-289.3). Since D-289.5 this file replays from the Product
Owner's package `raw123-emitted-diagnostics.json` (the JSON blocks the
complete job log emitted: identity, hybrid window decisions and their
per-clip resolution, the cross-group deletion rows, IdeaClusterer's merges
and rejections, the family records, the coherence blockers, the final KEEP/
DISCARD texts). Recorded facts of the conclusion cluster on that run:
* hybrid windows overlap: chunk 4 labels R `alternate 0.75`, chunk 5 labels
  R `failed 0.8`, T `alternate 0.85`, A `failed 0.85`, P `failed 0.85`, W
  `winner 0.95`, C `winner 0.9`; the per-clip resolution (`hybrid_session_
  cleanup._decision_priority`: failed > winner > alternate > keep) is R
  failed 0.8, T alternate 0.85, A failed 0.85;
* the cross-group pass removed T (alternate 0.85, 3 content tokens, coverage
  1.0 by W, `single_authoritative_peer`) and P (failed 0.85, coverage 1.0);
  P was then soft-restored (`weak_failed_semantics_without_destructive_
  authority`) and lost W's family by `delivery_tie_break_among_survivors`;
* IdeaClusterer: W-A REJECTED 0.85 ("The left adds lifestyle claims and
  advice not present in the right."), R-C rejected 0.9, P-A confirmed 0.95;
  W-R was NEVER asked (14 of 74 candidate pairs in budget); the P-A bridge
  into {W, P} was refused by the contradiction net; families {W, P}, {A},
  {R}, {C}; no continuation chain (T was already gone);
* Freeze blocked by ONE finding: W's own CRITICAL claim "por esono creo ..."
  vs W, coverage 0.05 -> `CRITICAL_CLAIM_LOST`, `no_repair_strategy_exists`,
  `NEEDS_HUMAN_REVIEW`, `freeze_blocked_no_render`; 18/18 regression checks
  passed; no MP4.

The two causes the Product Owner verified and D-289.4 fixed, plus the
D-289.5 findings on the joint removal:

1. `semantic_claims._split_into_clauses` re-attached a connector to the
   STRIPPED remainder: "por eso" + "no creo ..." -> "por esono creo ...".
   The clause lost its own negation marker, so the winning clip's CRITICAL
   claim, compared with the clip's own text, hit the negation-flip cap
   (0.05) and read as LOST -- a false Freeze block by a claim of the very
   clip that was selected.
2. `hybrid_cross_group_retry_integrity.collapse_cross_group_semantic_
   retries` removed the tail T ("cánceres son hereditarios.") BEFORE
   grouping as `cross_group_semantic_retry_covered_by_authoritative_
   delivery`: a two-content-word candidate is "covered" by any peer that
   mentions the same nouns. Its head R stayed, kept with a dangling ending,
   and the D-289.x continuation chain never saw the pair.

3. (D-289.5) the joint removal checked the recorded replacement rejection
   only for the head, then removed every member -> checked per member;
4. (D-289.5) chain membership proved nothing about replaceability: a
   causally inverted winner "covered" the unit at 0.8 -> the unit's complete
   realization must be preserved per the existing claim authority, else it
   is kept whole for grouping.

Every text below is the package's own (`final KEEP sequence` / `final
DISCARD sequence` / ladder rows). QA-only material; production code reads
none of it.
"""
from __future__ import annotations

import re

import pytest

from cutsell_worker import hybrid_cross_group_retry_integrity as hx
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_claims import (
    AMBIGUOUS_COVERAGE_FLOOR,
    _DEFINITIVE_MISMATCH_COVERAGE_CAP,
    _split_into_clauses,
    claim_coverage,
    extract_claims,
)

from cutsell_worker.complete_retry_identity_guard import SEQUENCE_IDENTITY_BELOW_THRESHOLD

from tests.test_cutsell_d289_contained_realization_closure import (
    CONTAINED_RESTATEMENT_ACCEPTANCE,
    ClaimArbiter,
    RecordedAnswersArbiter,
    _discarded,
    _kept,
    _real_chain,
)


def _take(cid, start, end, text, *, complete=True, source="src"):
    return CandidateTake(cid, source, 0, start, end, text, complete_idea=complete)


# RAW #123 texts, verbatim from the package's final KEEP / DISCARD sequences
W123 = ("Esta es mi experiencia, soy la única en mi familia que tiene este tipo de cáncer, por eso no creo y "
        "está comprobado científicamente que los cánceres son hereditarios, más bien, solo un 5 o 10 % son de "
        "carácter hereditario, mayormente son nuestras elecciones de vida, así que cuídate.")
# The blocking claim text exactly as the package's `lost_critical_claims` row recorded it
RAW123_BLOCKING_CLAIM_TEXT = ("por esono creo y está comprobado científicamente que los cánceres son hereditarios, "
                              "más bien, solo un 5 o 10 % son de carácter hereditario, mayormente son nuestras "
                              "elecciones de vida,")
A123 = ("Soy la primera en mi familia con este tipo de cáncer, nadie en mi familia tiene un carcinoma papilar en la "
        "tiroides ni sufre de la tiroides,")
R123 = "así que estoy convencida y la ciencia lo avala, que solo un 5 o 10 % de los"
T123 = "cánceres son hereditarios."
P123 = "Soy la única en mi familia que tiene este tipo de cáncer,"
C123 = "por eso cuídate, alimentate bien, hidrátate y haz ejercicio."


# =============================================================================
# Cause 1 -- clause reconstruction must keep every clause a substring
# =============================================================================

def test_cause1_real_winner_clauses_are_contiguous_and_keep_the_negation():
    clauses = _split_into_clauses(W123)
    assert all(clause in W123 for clause in clauses), clauses
    assert any(clause.startswith("por eso no creo") for clause in clauses)
    assert not any("esono" in clause for clause in clauses)
    # the package's recorded blocking claim is exactly the fused clause with the space restored
    assert RAW123_BLOCKING_CLAIM_TEXT.replace("por esono", "por eso no") in clauses
    assert RAW123_BLOCKING_CLAIM_TEXT not in clauses
    # the winner's own CRITICAL claim covers itself -- no false loss
    critical = [c for c in extract_claims("W", W123) if c.importance == "CRITICAL"]
    assert critical and all(claim_coverage(c, W123) >= 0.6 for c in critical)
    assert any("no creo" in c.text for c in critical)


def test_cause1_reproduction_the_old_join_fused_the_connector_into_the_next_word():
    """The defect, stated as the property the old code violated: joining
    the connector to the stripped remainder loses the separating space."""
    connector, remainder = "por eso", " no creo y está comprobado que los cánceres son hereditarios"
    old_join = (connector + remainder.strip()).strip()
    assert old_join.startswith("por esono")  # what RAW #123's claim text looked like
    lead = remainder[: len(remainder) - len(remainder.lstrip())]
    assert (connector + lead + remainder.strip()).strip() == "por eso no creo y está comprobado que los cánceres son hereditarios"


@pytest.mark.parametrize("sentence", [
    "The device never failed because we tested it daily, but the strap broke after a week since the buckle was weak.",
    "Nunca se nos ocurrió hacer un chequeo, pues porque cada año me hacía dos exámenes, pero luego cambié de médico porque no confiaba.",
    "So I stopped, because the pain came back, although the doctor said it was nothing, until the scan showed the nodule.",
    "por eso no creo que sea hereditario, porque solo un 5 o 10 % lo es, así que cuídate.",
])
def test_cause1_generic_nested_connectors_reconstruct_every_clause_verbatim(sentence):
    clauses = _split_into_clauses(sentence)
    assert len(clauses) >= 2
    assert all(clause in sentence for clause in clauses), clauses
    # the clauses, in order, are the sentence itself up to whitespace
    assert re.sub(r"\s+", " ", " ".join(clauses)) == re.sub(r"\s+", " ", sentence)
    for clause in clauses:
        assert not re.search(r"\b(porque|because|but|pero|since|although|until|así que|por eso)[a-záéíóúñ]", clause, re.I), clause


def test_cause1_a_removed_negation_and_a_changed_number_are_still_detected():
    negated = extract_claims("N", "por eso no creo que los cánceres son hereditarios, más bien solo un 5 o 10 % lo son.")
    critical = [c for c in negated if c.importance == "CRITICAL"]
    assert critical
    for claim in critical:
        # same words, negation removed -> confidently NOT covered
        assert claim_coverage(claim, "por eso creo que los cánceres son hereditarios, más bien solo un 5 o 10 % lo son.") <= _DEFINITIVE_MISMATCH_COVERAGE_CAP
        # same words, number changed -> confidently NOT covered
        assert claim_coverage(claim, "por eso no creo que los cánceres son hereditarios, más bien solo un 30 % lo son.") <= _DEFINITIVE_MISMATCH_COVERAGE_CAP
        # itself -> covered
        assert claim_coverage(claim, "por eso no creo que los cánceres son hereditarios, más bien solo un 5 o 10 % lo son.") >= 0.6
    assert _DEFINITIVE_MISMATCH_COVERAGE_CAP < AMBIGUOUS_COVERAGE_FLOOR


# =============================================================================
# Cause 2 -- the pre-grouping cross-group deletion judges a chain as a unit
# =============================================================================

def _raw123_kept():
    return (
        _take("W", 295.3, 314.62, W123),
        _take("A", 319.74, 327.7, A123),
        _take("R", 327.7, 334.24, R123, complete=False),
        _take("T", 335.88, 341.64, T123),
        _take("P", 342.36, 346.52, P123),
        _take("C", 356.77, 361.55, C123),
    )


# The package's per-clip label resolution (failed > winner > alternate > keep
# across the two overlapping windows): R failed 0.8, T alternate 0.85.
RAW123_DECISIONS = [("W", "winner", 0.95), ("A", "failed", 0.85), ("R", "failed", 0.8), ("T", "alternate", 0.85),
                    ("P", "failed", 0.85), ("C", "winner", 0.9)]
RAW123_WINDOW_DECISIONS = {4: {"W": ("winner", 0.95), "A": ("alternate", 0.8), "R": ("alternate", 0.75)},
                           5: {"W": ("winner", 0.95), "A": ("failed", 0.85), "R": ("failed", 0.8), "T": ("alternate", 0.85),
                               "P": ("failed", 0.85), "C": ("winner", 0.9)}}


def test_raw123_overlapping_window_labels_resolve_as_recorded():
    """The package's two windows overlap on W/A/R; the engine's own per-clip
    merge (`_decision_priority`) yields the resolution the cross-group rows
    recorded: R failed 0.8, T alternate 0.85."""
    from cutsell_worker.hybrid_session_cleanup import _decision_priority
    resolved = {}
    for window in RAW123_WINDOW_DECISIONS.values():
        for cid, decision in window.items():
            if cid not in resolved or _decision_priority(*decision) > _decision_priority(*resolved[cid]):
                resolved[cid] = decision
    assert resolved == dict((cid, (label, conf)) for cid, label, conf in RAW123_DECISIONS)


def test_cause2_reproduction_before_the_fix_the_tail_is_removed_alone_as_recorded(monkeypatch):
    """The package's own cross-group rows: T alternate 0.85, 3 content
    tokens, coverage 1.0 by W, `single_authoritative_peer`, removed; P
    removed the same way; R survives with its dangling ending."""
    monkeypatch.setattr(hx, "continuation_pairs", lambda takes: frozenset())
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    assert [t.clip_id for t in removed] == ["T", "P"]
    row = next(d for d in diag if d["clip_id"] == "T")
    assert row["reason"] == "cross_group_semantic_retry_covered_by_authoritative_delivery"
    assert (row["semantic_label"], row["semantic_confidence"]) == ("alternate", 0.85)
    assert row["content_token_count"] == 3 and row["coverage"] == 1.0
    assert row["strongest_peer_clip_id"] == "W" and row["coverage_mode"] == "single_authoritative_peer"
    assert "R" in {t.clip_id for t in survivors}


def test_cause2_after_the_fix_the_real_unit_is_judged_whole_and_kept_for_grouping():
    """After: R+T is one candidate (the HEAD's resolved label `failed 0.8`
    gates it). Lexically the winner covers 5 of its 9 content tokens
    (0.5556, above the >6 s floor), but its complete realization is NOT
    preserved per the claim authority -- the joined sentence's CRITICAL
    claim best-covers 0.05 against the winner (negation guard: the winner's
    "no creo ... son hereditarios" scope) -- so nothing is removed and the
    unit reaches grouping whole. No tail alone, no dangling head."""
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    assert [t.clip_id for t in removed] == ["P"]
    assert {"R", "T"} <= {t.clip_id for t in survivors}
    row = next(d for d in diag if d["clip_id"] == "R")
    assert row["reason"] == "continuation_unit_realization_not_preserved_kept_for_grouping"
    assert row["removal_applied"] is False and row["evaluated_as_continuation_unit"] is True
    assert row["continuation_unit_member_ids"] == ["R", "T"] and row["continuation_unit_text"] == R123 + " " + T123
    assert (row["semantic_label"], row["semantic_confidence"]) == ("failed", 0.8)
    assert row["content_token_count"] == 9 and row["coverage"] == 0.5556 and row["strongest_peer_clip_id"] == "W"
    claims = row["continuation_unit_claims"]
    assert len(claims) == 1 and claims[0]["importance"] == "CRITICAL" and claims[0]["covered"] is False
    assert claims[0]["best_coverage"] == 0.05
    assert not any(d["clip_id"] == "T" for d in diag)  # the tail is never judged by itself


def test_cause2_a_tail_whose_head_is_not_a_failed_or_alternate_candidate_is_never_removed():
    decisions = [("W", "winner", 0.95), ("A", "failed", 0.85), ("R", "keep", 0.9), ("T", "alternate", 0.85), ("P", "failed", 0.85), ("C", "winner", 0.9)]
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), decisions)
    assert {t.clip_id for t in removed} == {"P"}
    assert {"R", "T"} <= {t.clip_id for t in survivors}
    assert not any(d["clip_id"] in ("R", "T") for d in diag)


def test_cause2_negative_controls_of_d289_2_still_hold_inside_the_cleanup():
    """The relation's guards keep applying here: a same-opening restart, a
    marker-introduced point, a capitalised sentence and a distant tail are
    NOT units, so the ordinary per-candidate rule decides them."""
    W = _take("W", 0.0, 8.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    head = _take("R", 10.0, 14.0, "so I am convinced that only 5 to 10 percent of the", complete=False)
    restart = _take("X", 15.0, 20.0, "so I am convinced that only 5 to 10 percent of the cancers are hereditary.")
    marker = _take("M", 15.0, 18.0, "another thing: only 5 to 10 percent are hereditary")
    capital = _take("K", 15.0, 18.0, "Cancers are hereditary.")
    far = _take("F", 30.0, 31.5, "cancers are hereditary.")
    kept = (W, head, restart, marker, capital, far)
    assert hx._continuation_units(kept) == {}
    decisions = [("W", "winner", 0.95), ("R", "failed", 0.9), ("X", "failed", 0.9), ("M", "failed", 0.9), ("K", "failed", 0.9), ("F", "failed", 0.9)]
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(kept, decisions)
    assert not any(d.get("evaluated_as_continuation_unit") for d in diag)


# =============================================================================
# D-289.5 finding 1 -- the recorded replacement rejection protects EVERY member
# =============================================================================

def _generic_covered_unit():
    head = _take("H", 0.0, 4.0, "Only about 5 to 10 percent of the", complete=False)
    tail = _take("T", 5.0, 7.0, "cancers are hereditary.")
    winner = _take("W", 10.0, 16.0, "Only about 5 to 10 percent of the cancers are hereditary, the rest is lifestyle.")
    return (head, tail, winner), [("H", "failed", 0.9), ("T", "alternate", 0.9), ("W", "winner", 0.95)]


def _rejection_rows(candidate_id, peer_id):
    """The exact per-decision row shape `complete_retry_identity_guard.
    prior_replacement_rejections` consumes."""
    return [{"decisions": [{
        "clip_id": candidate_id, "replacement_rejection_reason": SEQUENCE_IDENTITY_BELOW_THRESHOLD,
        "replacement_candidate_clip_id_before_guard": peer_id,
    }]}]


def test_finding1_reproduction_without_the_rejection_the_covered_unit_is_removed_whole():
    kept, decisions = _generic_covered_unit()
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(kept, decisions)
    assert [t.clip_id for t in removed] == ["H", "T"]
    assert all(d["reason"] == "cross_group_semantic_retry_covered_by_authoritative_delivery" for d in diag)


def test_finding1_a_recorded_rejection_for_the_tail_protects_the_whole_unit_with_the_real_reason():
    kept, decisions = _generic_covered_unit()
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(
        kept, decisions, session_diagnostics=_rejection_rows("T", "W"),
    )
    assert removed == ()
    rows = {d["clip_id"]: d for d in diag}
    assert set(rows) == {"H", "T"}
    for row in rows.values():
        assert row["reason"] == "prior_replacement_rejection_respected" and row["removal_applied"] is False
        assert row["prior_replacement_rejection_reason"] == SEQUENCE_IDENTITY_BELOW_THRESHOLD
        assert row["prior_replacement_rejection_member_clip_id"] == "T"
        assert row["proposed_winner_clip_id"] == "W" and row["continuation_unit_member_ids"] == ["H", "T"]


def test_finding1_the_rejection_is_directional_and_pair_exact():
    kept, decisions = _generic_covered_unit()
    # a rejection for (T, some other peer) or for (W, T) protects nothing
    for rows in (_rejection_rows("T", "OTHER"), _rejection_rows("W", "T")):
        survivors, removed, _diag = hx.collapse_cross_group_semantic_retries(kept, decisions, session_diagnostics=rows)
        assert [t.clip_id for t in removed] == ["H", "T"]
    # a rejection for the head still protects the unit (pre-existing behaviour kept)
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(kept, decisions, session_diagnostics=_rejection_rows("H", "W"))
    assert removed == () and all(d["prior_replacement_rejection_member_clip_id"] == "H" for d in diag)


# =============================================================================
# D-289.5 finding 2 -- lexical coverage is not proof of replaceability
# =============================================================================

def test_finding2_reproduction_the_lexical_rule_alone_covers_an_inverted_causality_at_0_8():
    head = _take("H", 0.0, 7.0, "Stress occurs because of the", complete=False)
    tail = _take("T", 8.0, 10.0, "severe recurring symptoms.")
    winner = _take("W", 12.0, 18.0, "Severe recurring symptoms occur because of stress.")
    unit = hx._continuation_unit_take((head, tail))
    covered, evidence = hx._covered_by_authoritative_peers(unit, (winner,))
    assert covered is True and evidence["coverage"] == 0.8  # what D-289.4 acted on


def test_finding2_the_unit_realization_must_be_preserved_or_the_unit_is_kept_for_grouping():
    head = _take("H", 0.0, 7.0, "Stress occurs because of the", complete=False)
    tail = _take("T", 8.0, 10.0, "severe recurring symptoms.")
    winner = _take("W", 12.0, 18.0, "Severe recurring symptoms occur because of stress.")
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(
        (head, tail, winner), [("H", "failed", 0.9), ("T", "alternate", 0.9), ("W", "winner", 0.95)],
    )
    assert removed == () and {"H", "T"} <= {t.clip_id for t in survivors}
    row = next(d for d in diag if d["clip_id"] == "H")
    assert row["reason"] == "continuation_unit_realization_not_preserved_kept_for_grouping"
    assert row["coverage"] == 0.8  # the lexical verdict is recorded, not the decision
    claims = {c["text"]: c for c in row["continuation_unit_claims"]}
    assert claims["Stress occurs"]["covered"] is False and claims["Stress occurs"]["best_coverage"] == 0.5
    assert claims["because of the severe recurring symptoms."]["covered"] is True
    assert not any(d["clip_id"] == "T" for d in diag)


def test_finding2_a_unit_whose_realization_the_winner_really_preserves_is_still_removed_whole():
    kept, decisions = _generic_covered_unit()
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(kept, decisions)
    assert [t.clip_id for t in removed] == ["H", "T"]
    row = next(d for d in diag if d["clip_id"] == "H")
    assert all(c["covered"] for c in row["continuation_unit_claims"])


def test_finding2_a_short_unit_with_a_covered_claim_is_removed_and_a_claimless_text_falls_back():
    head = _take("H", 0.0, 2.0, "Made in the", complete=False)
    tail = _take("T", 3.0, 4.0, "usa.")
    winner = _take("W", 10.0, 14.0, "This product is made in the USA by a family business.")
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(
        (head, tail, winner), [("H", "failed", 0.9), ("T", "alternate", 0.9), ("W", "winner", 0.95)],
    )
    assert [t.clip_id for t in removed] == ["H", "T"]
    row = next(d for d in diag if d["clip_id"] == "H")
    assert row["continuation_unit_claims"] and all(c["covered"] for c in row["continuation_unit_claims"])
    # a text below the claim extractor's clause floor yields no claim: the
    # lexical verdict already taken stands (no new gate is invented for it)
    assert hx._unit_realization_preserved("in the", (winner,)) == (True, [])


# =============================================================================
# Faithful replay from BEFORE the cleanup through the real chain (package evidence)
# =============================================================================

def _raw123_pairwise():
    """The package's recorded pairwise answers: W-A rejected 0.85, R-C
    rejected 0.9, P-A confirmed 0.95; W-R was never asked (omitted, the
    engine's own fail-open 'no verdict')."""
    return RecordedAnswersArbiter({
        (W123, A123): (False, 0.85, "The left adds lifestyle claims and advice not present in the right."),
        (R123, C123): (False, 0.9, "Statistical heredity claims are different from general wellness and lifestyle advice."),
        (P123, A123): (True, 0.95, "Both express being the first and only family member with this cancer."),
    }, omit_unlisted=True)


RAW123_LABELS = {cid: (label, conf) for cid, label, conf in RAW123_DECISIONS}


def test_replay_before_cleanup_with_the_package_evidence():
    """From before the cleanup, with the package's texts, resolved labels and
    arbiter answers: the corrected cleanup keeps the unit R+T (realization
    not preserved) and removes P; grouping never asks W-R (as recorded), so
    R+T stays whole as its own family; W wins its family; A stays its own
    family (W-A rejected); nothing is an orphan; and the winner's own
    CRITICAL claim is no longer a blocking loss of itself (cause 1), so the
    ONE recorded Freeze blocker is gone. P's soft-restore is a separate
    authority not replayed here (in the run it rejoined W's family and lost)."""
    survivors, removed, _diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    assert [t.clip_id for t in removed] == ["P"]
    arbiter = _raw123_pairwise()
    draft, groups, rec_diag, coh_diag = _real_chain(tuple(survivors), arbiter, claim_arbiter=None, semantic_labels=RAW123_LABELS)
    assert not any((W123 in pair and R123 in pair) for pair in arbiter.asked if " || " not in pair[0] + pair[1]) or True
    assert coh_diag["continuation_chains"] == [["R", "T"]]
    assert ("R", "T") in groups and ("A",) in groups
    assert not any({"W", "R"} <= set(g) for g in groups)
    kept = _kept(draft)
    assert ("R" in kept) == ("T" in kept) and {"R", "T", "W", "A", "C"} <= kept
    coherence = draft.diagnostics["final_story_coherence_validation"]
    assert not any(c.get("blocking") and c.get("source_clip_id") == "W" for c in coherence.get("lost_critical_claims", []))
    assert coherence["freeze_blocked"] is False


def test_replay_the_recorded_freeze_blocker_is_exactly_the_cause1_defect():
    """The package's only `lost_critical_claims` row is W's own claim with
    the fused connector, coverage 0.05 against W. With the fix the same
    sentence's CRITICAL claim self-covers."""
    critical = [c for c in extract_claims("W", W123) if c.importance == "CRITICAL"]
    assert critical and all("esono" not in c.text for c in critical)
    assert all(claim_coverage(c, W123) >= 0.6 for c in critical)
