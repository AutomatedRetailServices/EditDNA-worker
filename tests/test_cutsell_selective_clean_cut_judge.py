from cutsell_worker.clean_cut_provider import (
    CleanCutJudgement,
    CleanCutProviderResult,
    apply_provider_judgements,
    safe_clean_cut_judge,
)
from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.providers import ProviderStatus


def _take(clip_id, start, end, text):
    tokens = text.split()
    duration = max(0.001, end - start)
    step = duration / max(1, len(tokens))
    words = tuple(
        Word(token, start + index * step, min(end, start + (index + 1) * step), 0.9)
        for index, token in enumerate(tokens)
    )
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        complete_idea=len(tokens) >= 6,
    )


class DeleteEverythingProvider:
    def __init__(self):
        self.calls = []

    def judge(self, takes):
        self.calls.append(tuple(take.clip_id for take in takes))
        return CleanCutProviderResult(
            tuple(
                CleanCutJudgement(take.clip_id, "delete", 0.99, "test")
                for take in takes
            ),
            ProviderStatus("fake", True, True, "applied"),
        )


def test_selective_judge_uses_neighbors_as_context_but_only_microtake_is_applicable():
    left = _take("left", 0.0, 4.0, "this is a valid long creator sentence here")
    micro = _take("micro", 4.1, 5.0, "duffel")
    right = _take("right", 5.1, 9.1, "this is another valid long creator sentence here")
    takes = (left, micro, right)
    provider = DeleteEverythingProvider()

    result = safe_clean_cut_judge(provider, takes)

    assert provider.calls == [("left", "micro", "right")]
    assert [item.clip_id for item in result.judgements] == ["micro"]

    kept, deleted, diagnostics = apply_provider_judgements(takes, result)
    assert [take.clip_id for take in kept] == ["left", "right"]
    assert [take.clip_id for take in deleted] == ["micro"]
    assert [item["clip_id"] for item in diagnostics] == ["micro"]


def test_selective_judge_does_not_call_provider_when_no_ambiguous_microtakes_exist():
    takes = (
        _take("left", 0.0, 4.0, "this is a valid long creator sentence here"),
        _take("right", 4.2, 8.2, "another complete creator sentence remains safely untouched"),
    )
    provider = DeleteEverythingProvider()

    result = safe_clean_cut_judge(provider, takes)

    assert provider.calls == []
    assert result.judgements == ()
    assert result.status.status == "not_requested_no_ambiguous_microtakes"


def test_selective_judge_reviews_valid_short_reactions_instead_of_hard_deleting_them():
    reaction = _take("reaction", 1.0, 1.6, "Yeah")

    class KeepProvider:
        def judge(self, takes):
            return CleanCutProviderResult(
                (CleanCutJudgement("reaction", "keep", 0.99, "intentional reaction"),),
                ProviderStatus("fake", True, True, "applied"),
            )

    result = safe_clean_cut_judge(KeepProvider(), (reaction,))
    kept, deleted, diagnostics = apply_provider_judgements((reaction,), result)

    assert [take.clip_id for take in kept] == ["reaction"]
    assert deleted == ()
    assert diagnostics[0]["action"] == "keep"
