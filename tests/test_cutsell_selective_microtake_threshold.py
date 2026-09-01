from cutsell_worker.clean_cut_provider import (
    CleanCutJudgement,
    CleanCutProviderResult,
    apply_provider_judgements,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus


def _take(clip_id: str = "micro") -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=0.6,
        text="At",
        complete_idea=False,
    )


def _selective_result(take: CandidateTake, confidence: float) -> CleanCutProviderResult:
    return CleanCutProviderResult(
        (CleanCutJudgement(take.clip_id, "delete", confidence, "candidate recording error"),),
        ProviderStatus("openai", True, True, "applied", "selective_microtake_review"),
    )


def test_selective_microtake_point_90_fails_open():
    take = _take()
    kept, deleted, diagnostics = apply_provider_judgements(
        (take,), _selective_result(take, 0.90)
    )

    assert kept == (take,)
    assert deleted == ()
    assert diagnostics[0]["delete_threshold"] == 0.94
    assert diagnostics[0]["applied_delete"] is False


def test_selective_microtake_point_93_still_fails_open():
    take = _take()
    kept, deleted, diagnostics = apply_provider_judgements(
        (take,), _selective_result(take, 0.93)
    )

    assert kept == (take,)
    assert deleted == ()
    assert diagnostics[0]["delete_threshold"] == 0.94
    assert diagnostics[0]["applied_delete"] is False


def test_selective_microtake_point_94_can_apply():
    take = _take()
    kept, deleted, diagnostics = apply_provider_judgements(
        (take,), _selective_result(take, 0.94)
    )

    assert kept == ()
    assert deleted == (take,)
    assert diagnostics[0]["delete_threshold"] == 0.94
    assert diagnostics[0]["applied_delete"] is True


def test_generic_delete_still_requires_point_94():
    take = _take()
    result = CleanCutProviderResult(
        (CleanCutJudgement(take.clip_id, "delete", 0.90, "uncertain"),),
        ProviderStatus("openai", True, True, "applied", "general_review"),
    )

    kept, deleted, diagnostics = apply_provider_judgements((take,), result)

    assert kept == (take,)
    assert deleted == ()
    assert diagnostics[0]["delete_threshold"] == 0.94
    assert diagnostics[0]["applied_delete"] is False
