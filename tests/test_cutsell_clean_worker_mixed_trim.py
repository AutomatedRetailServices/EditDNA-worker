from cutsell_worker.clean_cut_provider import (
    CleanCutJudgement,
    CleanCutProviderResult,
    apply_provider_judgements,
    safe_clean_cut_judge,
)
from cutsell_worker.contracts import CandidateTake, ProcessingRequest, SemanticLabel, SemanticRole, SourceAsset, Word
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.providers import ProviderStatus


def _take(text="This works great oh my god"):
    words = (
        Word("This", 1.00, 1.25, 0.99),
        Word("works", 1.26, 1.55, 0.99),
        Word("great", 1.56, 1.90, 0.99),
        Word("oh", 1.95, 2.10, 0.99),
        Word("my", 2.11, 2.24, 0.99),
        Word("god", 2.25, 2.50, 0.99),
    )
    return CandidateTake(
        clip_id="parent",
        source_asset_id="src",
        source_order=0,
        start=1.0,
        end=2.5,
        text=text,
        words=words,
    )


def _result(judgement):
    return CleanCutProviderResult(
        (judgement,),
        ProviderStatus("fake", True, True, "applied"),
    )


def test_high_confidence_mixed_trim_snaps_to_real_words_and_preserves_source():
    take = _take()
    judgement = CleanCutJudgement("parent", "mixed", 0.99, "valid speech then recording reaction", 0, 2)
    kept, discarded, diagnostics = apply_provider_judgements((take,), _result(judgement))

    assert len(kept) == 1
    assert len(discarded) == 1
    child = kept[0]
    rejected = discarded[0]
    assert child.clip_id != take.clip_id
    assert child.source_asset_id == take.source_asset_id
    assert child.start == 1.00
    assert child.end == 1.90
    assert child.text == "This works great"
    assert [word.text for word in child.words] == ["This", "works", "great"]
    assert rejected.source_asset_id == take.source_asset_id
    assert rejected.start == 1.95
    assert rejected.end == 2.50
    assert rejected.text == "oh my god"
    assert diagnostics[0]["applied_mixed_trim"] is True
    assert diagnostics[0]["kept_clip_id"] == child.clip_id
    assert diagnostics[0]["discarded_clip_ids"] == [rejected.clip_id]


def test_mixed_trim_can_discard_prefix_and_suffix_without_crossing_source():
    take = _take()
    judgement = CleanCutJudgement("parent", "mixed", 0.99, "valid middle span", 1, 3)
    kept, discarded, _ = apply_provider_judgements((take,), _result(judgement))

    assert kept[0].text == "works great oh"
    assert [item.text for item in discarded] == ["This", "my god"]
    assert all(item.source_asset_id == "src" for item in (*kept, *discarded))


def test_mixed_trim_fails_open_without_word_timings():
    take = CandidateTake("parent", "src", 0, 1.0, 2.0, "valid then blooper")
    judgement = CleanCutJudgement("parent", "mixed", 0.99, "mixed", 0, 1)
    kept, discarded, diagnostics = apply_provider_judgements((take,), _result(judgement))
    assert kept == (take,)
    assert discarded == ()
    assert diagnostics[0]["applied_mixed_trim"] is False


def test_mixed_trim_fails_open_for_low_confidence_invalid_or_tiny_keep():
    take = _take()
    cases = (
        CleanCutJudgement("parent", "mixed", 0.96, "below threshold", 0, 2),
        CleanCutJudgement("parent", "mixed", 0.99, "bad range", 4, 99),
        CleanCutJudgement("parent", "mixed", 0.99, "one word only", 0, 0),
        CleanCutJudgement("parent", "mixed", 0.99, "whole candidate", 0, 5),
    )
    for judgement in cases:
        kept, discarded, diagnostics = apply_provider_judgements((take,), _result(judgement))
        assert kept == (take,)
        assert discarded == ()
        assert diagnostics[0]["applied_mixed_trim"] is False


def test_malformed_mixed_boundary_payload_fails_open_at_provider_boundary():
    class Provider:
        def judge(self, takes):
            return _result(CleanCutJudgement("parent", "mixed", 0.99, "half boundary", 0, None))

    take = _take()
    judged = safe_clean_cut_judge(Provider(), (take,))
    assert judged.judgements == ()
    assert judged.status.status == "provider_error"
    kept, discarded, _ = apply_provider_judgements((take,), judged)
    assert kept == (take,)
    assert discarded == ()


def test_pipeline_preserves_semantic_role_on_word_trimmed_child_and_discarded_fragment():
    take = _take()

    class Provider:
        def judge(self, takes):
            return _result(CleanCutJudgement("parent", "mixed", 0.99, "valid benefit then reaction", 0, 2))

    request = ProcessingRequest(
        project_id="project",
        user_id="user",
        sources=(SourceAsset("src", "project", "user", "video.mp4", 0, 10.0, "s3://bucket/video.mp4"),),
    )
    label = SemanticLabel("parent", SemanticRole.BENEFITS, 0.91, "benefit statement")
    result = build_flow_b_draft(request, (take,), (label,), clean_cut_provider=Provider())

    assert len(result.draft.selected) == 1
    assert result.draft.selected[0].text == "This works great"
    assert result.draft.selected[0].semantic_role == SemanticRole.BENEFITS
    assert len(result.draft.discarded) == 1
    assert result.draft.discarded[0].text == "oh my god"
    assert result.draft.discarded[0].semantic_role == SemanticRole.BENEFITS
    assert result.draft.diagnostics["clean_cut_judge_mixed_trimmed_count"] == 1
