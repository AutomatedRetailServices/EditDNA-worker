from cutsell_worker.clean_cut_provider import (
    CleanCutJudgement,
    CleanCutProviderResult,
    apply_provider_judgements,
    safe_clean_cut_judge,
)
from cutsell_worker.clean_cut_retry import OneShotCleanCutContractRetry
from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.providers import ProviderStatus


def take(clip_id: str, text: str, *, start: float = 0.0, end: float = 2.0, with_words: bool = False) -> CandidateTake:
    words = ()
    if with_words:
        parts = text.split()
        span = max(0.01, (end - start) / max(1, len(parts)))
        words = tuple(
            Word(token, start + index * span, start + (index + 1) * span, 0.99)
            for index, token in enumerate(parts)
        )
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="source",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
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


class KeepProvider:
    def judge(self, takes):
        return CleanCutProviderResult(
            tuple(CleanCutJudgement(item.clip_id, "keep", 0.99, "baseline keep") for item in takes),
            ProviderStatus("fake", True, True, "applied"),
        )


class BrokenProvider:
    def judge(self, takes):
        return CleanCutProviderResult(
            (CleanCutJudgement("wrong-id", "delete", 1.0, "bad payload"),),
            ProviderStatus("fake", True, True, "applied"),
        )


class FlakyOmissionProvider:
    def __init__(self):
        self.calls = 0

    def judge(self, takes):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("clean cut judge omitted ambiguous microtake")
        return CleanCutProviderResult(
            tuple(CleanCutJudgement(item.clip_id, "keep", 0.99, "valid after strict retry") for item in takes),
            ProviderStatus("fake", True, True, "applied"),
        )


class LargeBatchOmissionProvider:
    def __init__(self):
        self.call_sizes = []

    def judge(self, takes):
        self.call_sizes.append(len(takes))
        if len(takes) > 14:
            raise ValueError("clean cut judge omitted ambiguous microtake")
        return CleanCutProviderResult(
            tuple(CleanCutJudgement(item.clip_id, "keep", 0.99, "valid batched response") for item in takes),
            ProviderStatus("fake", True, True, "applied"),
        )


def test_only_high_confidence_whole_delete_is_applied():
    takes = (
        take("a", "okay stop"),
        take("b", "blooper prefix and now I use this every day"),
        take("c", "what"),
        take("d", "this product is amazing"),
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


def test_value_error_gets_one_strict_retry_before_provider_failure():
    takes = (take("a", "valid short speech"), take("b", "another short phrase"))
    provider = FlakyOmissionProvider()
    result = safe_clean_cut_judge(provider, takes)
    assert provider.calls == 2
    assert result.status.status == "applied"
    assert [item.clip_id for item in result.judgements] == ["a", "b"]


def test_large_contract_omission_recovers_in_validated_context_batches():
    takes = tuple(take(f"clip-{index}", f"speech {index}") for index in range(25))
    underlying = LargeBatchOmissionProvider()
    provider = OneShotCleanCutContractRetry(underlying)
    result = provider.judge(takes)
    assert underlying.call_sizes[0] == 25
    assert all(size <= 14 for size in underlying.call_sizes[1:])
    assert result.status.status == "applied"
    assert "batched_contract_recovery" in (result.status.reason or "")
    assert [item.clip_id for item in result.judgements] == [item.clip_id for item in takes]


def test_repeated_false_start_uses_mixed_trim_when_word_boundaries_are_available():
    bad = take("bad", "this is the this is the shade", start=10.0, end=12.1, with_words=True)
    good = take("good", "this is the shade", start=12.4, end=14.0, with_words=True)
    provider = OneShotCleanCutContractRetry(KeepProvider())
    result = provider.judge((bad, good))
    by_id = {item.clip_id: item for item in result.judgements}
    assert by_id["bad"].action == "mixed"
    assert by_id["bad"].confidence >= 0.97
    assert by_id["bad"].reason == "repeated_prefix_trim"
    assert by_id["bad"].keep_start_word_index == 3
    assert by_id["bad"].keep_end_word_index == 6
    kept, deleted, diagnostics = apply_provider_judgements((bad, good), result)
    assert kept[0].text.lower() == "this is the shade"
    assert any(item["applied_mixed_trim"] for item in diagnostics if item["clip_id"] == "bad")


def test_self_critique_is_deleted_only_when_opening_repeats_cleanly_nearby():
    bad = take("bad", "Churro protein shake. Do you know I don't like the beginning?", start=0.0, end=2.54)
    good = take("good", "Churro protein shake. You'll be getting Churro.", start=6.84, end=9.36)
    provider = OneShotCleanCutContractRetry(KeepProvider())
    result = provider.judge((bad, good))
    by_id = {item.clip_id: item for item in result.judgements}
    assert by_id["bad"].action == "delete"
    assert by_id["bad"].reason == "recording_self_critique_with_retry"
    assert by_id["good"].action == "keep"


def test_self_critique_without_matching_retry_stays_kept():
    bad = take("bad", "I don't like the beginning", start=0.0, end=1.5)
    unrelated = take("next", "This product is amazing", start=2.0, end=3.5)
    provider = OneShotCleanCutContractRetry(KeepProvider())
    result = provider.judge((bad, unrelated))
    by_id = {item.clip_id: item for item in result.judgements}
    assert by_id["bad"].action == "keep"


def test_intentional_repetition_without_retry_stays_kept():
    takes = (
        take("reaction", "they are so so super cute", start=10.0, end=12.0, with_words=True),
        take("next", "I wear them every day", start=12.4, end=14.0, with_words=True),
    )
    provider = OneShotCleanCutContractRetry(KeepProvider())
    result = provider.judge(takes)
    by_id = {item.clip_id: item for item in result.judgements}
    assert by_id["reaction"].action == "keep"
    assert by_id["next"].action == "keep"


def test_triple_word_repetition_without_matching_retry_stays_kept():
    takes = (
        take("repeat", "little little little little little", start=10.0, end=11.5, with_words=True),
        take("next", "this one is perfect", start=12.0, end=13.5, with_words=True),
    )
    provider = OneShotCleanCutContractRetry(KeepProvider())
    result = provider.judge(takes)
    by_id = {item.clip_id: item for item in result.judgements}
    assert by_id["repeat"].action == "keep"


def test_no_provider_is_not_requested_and_keeps_everything():
    takes = (take("a", "valid speech"),)
    result = safe_clean_cut_judge(None, takes)
    assert result.status.status == "not_requested"
    kept, deleted, _ = apply_provider_judgements(takes, result)
    assert kept == takes
    assert deleted == ()
