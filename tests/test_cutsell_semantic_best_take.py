from cutsell_worker.contracts import CandidateTake
from cutsell_worker.pipeline import _semantic_best_take


def _take(clip_id: str, start: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 3.0,
        text=f"delivery {clip_id}",
    )


def test_clear_semantic_winner_overrides_marginal_local_choice_inside_retry_group():
    local = _take("local", 0.0)
    semantic = _take("semantic", 10.0)
    selected, preferred, reason = _semantic_best_take(
        (local, semantic),
        {
            "local": ("alternate", 0.85),
            "semantic": ("winner", 0.92),
        },
        "local",
    )
    assert selected == "semantic"
    assert preferred == "semantic"
    assert reason == "single_semantic_winner"


def test_semantic_winner_at_point_eight_five_is_enough_when_unique():
    local = _take("local", 0.0)
    semantic = _take("semantic", 10.0)
    selected, _, _ = _semantic_best_take(
        (local, semantic),
        {
            "local": ("alternate", 0.75),
            "semantic": ("winner", 0.85),
        },
        "local",
    )
    assert selected == "semantic"


def test_ambiguous_multiple_semantic_winners_fail_open_to_local_ranker():
    # D-082: neither candidate here carries a CRITICAL claim (no numbers/
    # negations) and no `ranked` scores are supplied, so every deterministic
    # escalation step finds nothing to decide on -- this is the genuine
    # "nothing else to consult" case, and the function still falls open to
    # the local ranker's own top choice, same as before D-082.
    a = _take("a", 0.0)
    b = _take("b", 10.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b),
        {"a": ("winner", 0.90), "b": ("winner", 0.88)},
        "a",
    )
    assert selected == "a"
    assert preferred is None
    assert reason == "local_fallback"
