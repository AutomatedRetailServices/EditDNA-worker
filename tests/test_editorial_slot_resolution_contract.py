"""Contract tests for complete-realization editorial slot resolution.

These tests intentionally define the required decision semantics before the resolver
implementation is wired into the production pipeline.
"""

from dataclasses import dataclass


@dataclass
class Realization:
    required_idea_coverage: float
    factual_safety: float
    completeness: float
    redundancy: float
    delivery_quality: float
    narrative_fit: float
    brevity: float
    complete: bool = True


def editorial_score(r: Realization) -> tuple:
    """Lexicographic authority used by the contract.

    Required ideas and factual safety outrank stylistic delivery. Redundancy is a
    penalty once sufficiency is established. Brevity is only a late tiebreaker.
    """

    return (
        r.required_idea_coverage,
        r.factual_safety,
        r.completeness,
        -r.redundancy,
        r.delivery_quality,
        r.narrative_fit,
        r.brevity,
    )


def resolve_complete_realizations(candidates):
    complete = [c for c in candidates if c.complete]
    if len(complete) >= 2:
        return [max(complete, key=editorial_score)]
    return candidates


def test_two_complete_same_slot_compete_instead_of_cokeep():
    first = Realization(1, 1, 1, 0.1, 0.9, 1.0, 0.8)
    retry = Realization(1, 1, 1, 0.8, 0.95, 0.9, 0.5)
    assert resolve_complete_realizations([first, retry]) == [first]


def test_unique_supporting_wording_does_not_force_second_complete_take():
    concise_complete = Realization(1, 1, 1, 0.0, 0.9, 1.0, 0.9)
    elaborated_retry = Realization(1, 1, 1, 0.7, 0.92, 0.9, 0.4)
    assert resolve_complete_realizations([concise_complete, elaborated_retry]) == [concise_complete]


def test_incomplete_candidates_are_not_forced_into_single_winner_contract():
    a = Realization(0.6, 1, 0.6, 0.0, 0.9, 0.9, 0.9, complete=False)
    b = Realization(0.7, 1, 0.7, 0.0, 0.9, 0.9, 0.9, complete=False)
    assert resolve_complete_realizations([a, b]) == [a, b]
