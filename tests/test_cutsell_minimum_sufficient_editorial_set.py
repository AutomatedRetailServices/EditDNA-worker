from cutsell_worker import pipeline
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
)
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence


CONCLUSION_A = (
    "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer "
    "está comprobado científicamente que los cánceres son hereditarios solo un porcentaje "
    "son de carácter hereditario mayormente son nuestras elecciones de vida"
)
CONCLUSION_B = (
    "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un "
    "carcinoma papilar en la tiroides y la ciencia avala que solo un porcentaje de los cánceres son hereditarios"
)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=(),
        signals=None,
        complete_idea=complete,
    )


class _EditorialSlotArbiter:
    def __init__(self):
        self.request = None

    def check(self, request):
        self.request = request
        decisions = []
        for index, pair in enumerate(request.pairs):
            texts = {pair.left_text, pair.right_text}
            same = CONCLUSION_A in texts and CONCLUSION_B in texts
            decisions.append(IdeaEquivalenceDecision(
                pair_index=index,
                same_idea=same,
                confidence=0.97 if same else 0.95,
                reason="same conclusion slot" if same else "different story beat",
            ))
        return IdeaEquivalenceResult(
            decisions=tuple(decisions),
            provider="test",
            model="test",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=100,
        )


def test_late_same_slot_conclusions_reach_arbiter_under_pair_budget_and_merge():
    distractors = tuple(
        _take(
            f"d{i}",
            250.0 + i * 5.0,
            254.0 + i * 5.0,
            f"story beat number {i} about a separate unrelated subject and event",
        )
        for i in range(8)
    )
    conclusion_a = _take("conclusion-a", 295.36, 314.60, CONCLUSION_A, complete=True)
    conclusion_b = _take("conclusion-b", 319.38, 334.24, CONCLUSION_B, complete=True)
    takes = (*distractors, conclusion_a, conclusion_b)
    groups = tuple((take.clip_id,) for take in takes)
    arbiter = _EditorialSlotArbiter()

    reconciled, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)

    assert arbiter.request is not None
    requested_pairs = [{pair.left_text, pair.right_text} for pair in arbiter.request.pairs]
    assert {CONCLUSION_A, CONCLUSION_B} in requested_pairs
    merged = next(group for group in reconciled if "conclusion-a" in group)
    assert "conclusion-b" in merged
    assert diagnostics["merged_pair_count"] >= 1


def test_incomplete_semantic_winner_cannot_override_complete_local_winner():
    complete_local = _take("complete-local", 0.0, 18.0, CONCLUSION_A, complete=True)
    incomplete_semantic = _take("incomplete-semantic", 20.0, 32.0, CONCLUSION_B, complete=False)
    decisions = {
        "complete-local": ("alternate", 0.78),
        "incomplete-semantic": ("winner", 0.96),
    }

    selected, preferred = pipeline._semantic_best_take(
        (complete_local, incomplete_semantic),
        decisions,
        "complete-local",
    )

    assert selected == "complete-local"
    assert preferred is None


def test_complete_semantic_winner_can_still_override_complete_local_winner():
    complete_local = _take("complete-local", 0.0, 18.0, CONCLUSION_A, complete=True)
    complete_semantic = _take("complete-semantic", 20.0, 36.0, CONCLUSION_B, complete=True)
    decisions = {
        "complete-local": ("alternate", 0.78),
        "complete-semantic": ("winner", 0.96),
    }

    selected, preferred = pipeline._semantic_best_take(
        (complete_local, complete_semantic),
        decisions,
        "complete-local",
    )

    assert selected == "complete-semantic"
    assert preferred == "complete-semantic"
