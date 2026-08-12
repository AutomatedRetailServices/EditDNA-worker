from cutsell_worker.clean_cut_provider import (
    CleanCutJudgement,
    CleanCutProviderResult,
    apply_provider_judgements,
    safe_clean_cut_judge,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus


def take(clip_id: str, text: str) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="source",
        source_order=0,
        start=0.0,
        end=2.0,
        text=text,
    )


class GoodProvider:
    def judge(self, takes):
        return CleanCutProviderResult(
            (
                CleanCutJudgement(takes[0].clip_id, "delete", 0.97, "obvious restart"),
                CleanCutJudgement(takes[1].clip_id, "mixed", 0.99, "blooper prefix plus valid speech"),
                CleanCutJudgement(takes[2].clip_id, "delete", 0.90, "not certain enough"),
                CleanCutJudgement(takes[3].clip_id, "keep", 0.99, "valid creator speech"),
            ),
            ProviderStatus("fake", True, True, "applied"),
        )


class BrokenProvider:
    def judge(self, takes):
        return CleanCutProviderResult(
            (CleanCutJudgement("wrong-id", "delete", 1.0, "bad payload"),),
            ProviderStatus("fake", True, True, "applied"),
        )


def test_only_high_confidence_whole_delete_is_applied():
    takes = (
        take("a", "okay stop"),
        take("b", "fuck sorry and now I use this every day"),
        take("c", "what"),
        take("d", "this product is fucking amazing"),
    )
    result = safe_clean_cut_judge(GoodProvider(), takes)
    kept, deleted, diagnostics = apply_provider_judgements(takes, result)
    assert [item.clip_id for item in deleted] == ["a"]
    assert [item.clip_id for item in kept] == ["b", "c", "d"]
    assert next(item for item in diagnostics if item["clip_id"] == "b")["applied_delete"] is False
    assert next(item for item in diagnostics if item["clip_id"] == "c")["applied_delete"] is False


def test_malformed_provider_fails_open_and_keeps_everything():
    takes = (take("a", "valid speech"), take("b", "also valid"))
    result = safe_clean_cut_judge(BrokenProvider(), takes)
    assert result.status.status == "provider_error"
    assert result.status.reason == "ValueError: clean cut judge returned invalid clip id"
    kept, deleted, diagnostics = apply_provider_judgements(takes, result)
    assert kept == takes
    assert deleted == ()
    assert diagnostics == ()


def test_no_provider_is_not_requested_and_keeps_everything():
    takes = (take("a", "valid speech"),)
    result = safe_clean_cut_judge(None, takes)
    assert result.status.status == "not_requested"
    kept, deleted, _ = apply_provider_judgements(takes, result)
    assert kept == takes
    assert deleted == ()
