"""D-289.4 -- the two RAW #123 causes that sit BEFORE grouping.

RAW #123 = Modal run 35866604610 on `fix/editorial-realization-closure@
0b1572a8` (D-289.3). The Product Owner's package with the full log and the
real diagnostics did NOT reach this environment (uploads/attach mounts hold
nothing newer than the RAW #122 JSON), so this file replays from what the
run's own workflow log tail exposes -- the complete quality-ladder region
map with every candidate's text, timing and status -- plus the two causes
the Product Owner verified on the full package:

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

Texts below are RAW #123's own (the ladder rows carry them verbatim up to
140 characters; the winner's tail beyond that is reconstructed from RAW
#122's sentence with RAW #123's punctuation and is marked as such). RAW
#123's hybrid labels are NOT available here: the winner/aside/head labels
are RAW #122's recorded ones and the tail's `failed 0.8` is the minimal
assumption consistent with the recorded reason code (a removal requires
`failed`/`alternate` >= 0.75). Both are stated in the tests that use them.
QA-only material; production code reads none of it.
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


# RAW #123 texts (ladder rows; the winner beyond "... que los" is reconstructed)
W123_LOGGED_PREFIX = ("Esta es mi experiencia, soy la única en mi familia que tiene este tipo de cáncer, por eso no creo y "
                      "está comprobado científicamente que los ")
W123 = (W123_LOGGED_PREFIX + "cánceres son hereditarios, más bien solo un 5 o 10 % son de carácter hereditario, "
        "mayormente son nuestras elecciones de vida, así que cuídate.")
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


# W/A/R labels: RAW #122's recorded ones. T `failed 0.8`: ASSUMED (the
# recorded reason code requires failed/alternate >= 0.75; RAW #123's own
# label is in the package this environment did not receive).
RAW123_DECISIONS = [("W", "winner", 0.95), ("A", "alternate", 0.8), ("R", "failed", 0.9), ("T", "failed", 0.8), ("P", "alternate", 0.75)]


def test_cause2_reproduction_before_the_fix_the_tail_is_removed_alone(monkeypatch):
    monkeypatch.setattr(hx, "continuation_pairs", lambda takes: frozenset())
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    assert [t.clip_id for t in removed] == ["T", "P"]
    row = next(d for d in diag if d["clip_id"] == "T")
    assert row["reason"] == "cross_group_semantic_retry_covered_by_authoritative_delivery"
    assert row["coverage"] == 1.0 and row["strongest_peer_clip_id"] == "W" and row["coverage_mode"] == "single_authoritative_peer"
    assert "R" in {t.clip_id for t in survivors}  # the head survives with its dangling ending


def test_cause2_after_the_fix_the_chain_is_judged_as_one_sentence_and_moves_as_a_unit():
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    removed_ids = [t.clip_id for t in removed]
    assert set(removed_ids) >= {"R", "T"} and "W" not in removed_ids and "C" not in removed_ids
    rows = {d["clip_id"]: d for d in diag if d["clip_id"] in ("R", "T")}
    assert set(rows) == {"R", "T"}
    for row in rows.values():
        assert row["evaluated_as_continuation_unit"] is True
        assert row["continuation_unit_member_ids"] == ["R", "T"]
        assert row["continuation_unit_text"] == R123 + " " + T123
        assert row["semantic_label"] == "failed" and row["semantic_confidence"] == 0.9  # the HEAD's label decides
        assert row["content_token_count"] == 9 and row["coverage"] == 0.5556 and row["critical_preserved"] is True
    # never one without the other
    assert ("R" in removed_ids) == ("T" in removed_ids)


def test_cause2_a_tail_whose_head_is_not_a_failed_or_alternate_candidate_is_never_removed():
    decisions = [("W", "winner", 0.95), ("A", "alternate", 0.8), ("R", "keep", 0.9), ("T", "failed", 0.8), ("P", "alternate", 0.75)]
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), decisions)
    assert {t.clip_id for t in removed} == {"P"}
    assert {"R", "T"} <= {t.clip_id for t in survivors}
    assert not any(d["clip_id"] == "T" for d in diag)


def test_cause2_a_unit_the_winner_does_not_cover_reaches_grouping_whole():
    """With the head's own uncovered material (a claim of its own), the unit
    is not covered by the winner: nothing is removed, both members reach
    grouping, and the row says so."""
    head = _take("R", 327.7, 334.24, "así que después de leer tres estudios independientes me convencí de que solo un 5 o 10 % de los", complete=False)
    kept = tuple(head if t.clip_id == "R" else t for t in _raw123_kept())
    survivors, removed, diag = hx.collapse_cross_group_semantic_retries(kept, RAW123_DECISIONS)
    assert {"R", "T"} <= {t.clip_id for t in survivors}
    row = next(d for d in diag if d["clip_id"] == "R")
    assert row["reason"] == "continuation_unit_not_covered_kept_for_grouping" and row["removal_applied"] is False
    assert row["continuation_unit_member_ids"] == ["R", "T"]


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
# Replay from BEFORE the cleanup through the real chain (RAW #123 texts)
# =============================================================================

def _raw123_pairwise():
    """RAW #123's pairwise answers are in the unavailable package; RAW #122's
    recorded W-A / W-R / A-R verdicts are applied to RAW #123's texts, with
    the recorded 0.9 probe decline -- stated as such."""
    return RecordedAnswersArbiter({
        (W123, A123): (True, 0.9, "Both discuss being the only family member with the condition."),
        (W123, R123): (True, 0.85, "Second text starts restating the hereditary statistics argument."),
        (A123, R123): (True, 0.8, "Overlapping themes of familial uniqueness and hereditary stats."),
    }, declined_probe_texts=(R123,), probe_confidence=0.9)


RAW123_LABELS = {"W": ("winner", 0.95), "A": ("alternate", 0.8), "R": ("failed", 0.9), "T": ("failed", 0.8), "P": ("alternate", 0.75)}


def test_replay_before_cleanup_head_labelled_failed_the_unit_is_a_covered_retry_and_never_an_orphan():
    survivors, removed, _diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), RAW123_DECISIONS)
    assert {t.clip_id for t in removed} == {"R", "T", "P"}
    draft, groups, _rec, coh_diag = _real_chain(tuple(survivors), _raw123_pairwise(), claim_arbiter=None, semantic_labels=RAW123_LABELS)
    kept = _kept(draft)
    assert "W" in kept and "C" in kept
    assert not ({"R", "T"} & kept)  # no dangling head, no orphan tail
    assert coh_diag.get("continuation_chains") == []
    # the winner's own CRITICAL claim is not a lost claim of itself (cause 1)
    coherence = draft.diagnostics["final_story_coherence_validation"]
    assert not any(c.get("blocking") and c.get("source_clip_id") == "W" for c in coherence.get("lost_critical_claims", []))
    assert not any(a.get("blocking") and a.get("clip_id") == "W" for a in coherence.get("lost_semantic_atoms", []))


@pytest.mark.parametrize("claim_verdict", [True, False, None], ids=["confirming", "declining", "absent"])
def test_replay_before_cleanup_head_not_failed_the_unit_reaches_grouping_whole(claim_verdict):
    """If RAW #123's head label was NOT failed/alternate, the cleanup keeps
    the unit and grouping takes over. On RAW #123's OWN punctuation the
    winner is one comma-run sentence, so D-085's deterministic contradiction
    net (`detect_text_contradiction`, sentence-scoped negation) reads "no
    creo ... cánceres son hereditarios" against the unit's "cánceres son
    hereditarios" as a polarity conflict and refuses the bridge BEFORE any
    claim arbiter is consulted -- whatever that arbiter would answer. The
    unit therefore stays whole as its own family (both kept, never one
    alone). This is the primitive's honest verdict on this transcript, not
    something to force; recorded as a D-289.4 residual (the same texts with
    RAW #122's sentence breaks are the accepted-bridge case D-289.2 proved)."""
    decisions = [("W", "winner", 0.95), ("A", "alternate", 0.8), ("R", "keep", 0.9), ("T", "keep", 0.8), ("P", "alternate", 0.75)]
    labels = {"W": ("winner", 0.95), "A": ("alternate", 0.8), "R": ("keep", 0.9), "T": ("keep", 0.8)}
    survivors, removed, _diag = hx.collapse_cross_group_semantic_retries(_raw123_kept(), decisions)
    assert {t.clip_id for t in removed} == {"P"} and {"R", "T"} <= {t.clip_id for t in survivors}
    claims = None if claim_verdict is None else ClaimArbiter(claim_verdict)
    draft, groups, _rec, coh_diag = _real_chain(tuple(survivors), _raw123_pairwise(), claim_arbiter=claims, semantic_labels=labels)
    assert coh_diag["continuation_chains"] == [["R", "T"]]
    rows = [r for r in coh_diag["edge_trace"] if r.get("bridge_sensitive") and "R" in (r.get("left_clip_id"), r.get("right_clip_id"))]
    assert rows and all(r["reason_rejected"] == "cross_component_contradiction" and r["component_cohesion_evaluated"] is False for r in rows)
    assert not any(r.get("accepted_by") == CONTAINED_RESTATEMENT_ACCEPTANCE for r in coh_diag["edge_trace"])
    if claims is not None:
        assert claims.asked == []  # the net refused before the arbiter was ever asked
    kept = _kept(draft)
    assert ("R" in kept) == ("T" in kept)  # always a unit
    assert {"R", "T", "W", "C"} <= kept and ("R", "T") in groups
