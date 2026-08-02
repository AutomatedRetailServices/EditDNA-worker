import importlib.util
import logging
import sys
import types

import pytest


def stub(name, **attrs):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        module = types.ModuleType(name); module.__dict__.update(attrs); sys.modules[name] = module


stub("requests"); stub("boto3"); stub("clip"); stub("faster_whisper", WhisperModel=object)
from worker import pipeline
from worker.models import openai_provider as provider
from worker.models.openai_client import OpenAIProviderError, OpenAIResponseValidationError
from worker.take_judge_v2 import (
    TakeJudgeCandidate, TakeJudgeCandidateScore, TakeJudgeV2Result,
    delivery_features, sample_candidate_frames, temporal_timestamps,
)


def candidate(cid="a", start=10.0, end=20.0, **extra):
    value = {"id": cid, "start": start, "end": end, "text": "clear product message.",
             "slot": "HOOK", "semantic_score": .9, "meta": {"keep": True}}
    value.update(extra)
    return value


def result(winner="a", confidence=.9, abstain=False):
    scores = tuple(TakeJudgeCandidateScore(x, .8, .8, .8, .8, .8, "safe") for x in ("a", "b"))
    return TakeJudgeV2Result(winner, scores, confidence, abstain, "safe")


def test_temporal_sampling_one_multiple_and_boundary_safe():
    assert temporal_timestamps(10, 20, 1) == (15.0,)
    values = temporal_timestamps(10, 20, 3)
    assert len(values) == 3 and all(10 < value < 20 for value in values)
    assert values == tuple(sorted(values))


def test_short_clip_skips_duplicate_milliseconds():
    values = temporal_timestamps(1.0, 1.0001, 20)
    assert len(values) == 1 and 1.0 <= values[0] <= 1.0001


def test_correct_source_partial_failure_and_frame_limit(tmp_path):
    calls = []
    def extract(source, timestamp, path):
        calls.append((source, timestamp))
        if len(calls) == 2:
            return False
        with open(path, "wb") as image: image.write(b"jpeg")
        return True
    sample = sample_candidate_frames(candidate(source_local="second.mp4"), 3, "first.mp4", extract)
    assert [call[0] for call in calls] == ["second.mp4"] * 3
    assert sample.requested_frame_count == 3 and sample.attempted_count == 3
    assert len(sample.successful_frame_timestamps) == len(sample.image_content) == 2
    assert sample.failed_count == 1


def test_no_usable_frames_is_valid():
    sample = sample_candidate_frames(candidate(), 2, "input.mp4", lambda *args: False)
    assert sample.image_content == () and sample.failed_count == 2


@pytest.mark.parametrize("text,expected", [
    ("This product works very well.", (0, 0, False)),
    ("um uh like actually product", (4, 0, True)),
    ("this this works", (0, 1, True)),
])
def test_delivery_text_signals(text, expected):
    features = delivery_features(candidate(start=0, end=2, text=text))
    assert (features.filler_count, features.repeated_word_count, features.incomplete_phrase) == expected


def test_delivery_rate_pauses_absent_words_and_zero_duration():
    normal = delivery_features(candidate(start=0, end=2, text="one two three four", words=[
        {"start": 0, "end": .2}, {"start": 1.5, "end": 1.8}]))
    absent = delivery_features(candidate(start=1, end=1, text="one", words=None))
    assert normal.words_per_second == 2 and normal.excessive_pause
    assert absent.words_per_second == 0 and not absent.excessive_pause



def test_sentence_clips_preserve_only_overlapping_normalized_words():
    segments = [{"start": 0, "end": 5, "text": "One. Two.", "words": [
        {"start": 0.2, "end": 0.6, "word": " One."},
        {"start": 3.0, "end": 3.4, "word": " Two."},
    ]}]
    clips = pipeline.sentence_boundary_micro_cuts(segments)
    assert [[word["word"] for word in clip["words"]] for clip in clips] == [[" One."], [" Two."]]
    assert clips[0]["words"][0] == {"start": .2, "end": .6, "word": " One."}


def test_make_base_clip_excludes_words_outside_interval():
    clip = pipeline.make_base_clip("a", 1, 2, "inside", words=[
        {"start": .1, "end": .9, "word": "before"},
        {"start": 1.2, "end": 1.5, "word": "inside"},
        {"start": 2.1, "end": 2.5, "word": "after"},
    ])
    assert [word["word"] for word in clip["words"]] == ["inside"]


def test_merged_clips_order_and_deduplicate_words():
    duplicate = {"start": 1.0, "end": 1.2, "word": " shared"}
    first = pipeline.make_base_clip("a", 0, 1.3, "this continues", words=[
        duplicate, {"start": .1, "end": .3, "word": " this"}])
    second = pipeline.make_base_clip("b", 1, 2, "into a complete phrase.", words=[
        duplicate, {"start": 1.3, "end": 1.6, "word": " phrase."}])
    merged = pipeline.merge_incomplete_phrases([first, second])
    assert len(merged) == 1
    assert [word["word"] for word in merged[0]["words"]] == [" this", " shared", " phrase."]
    assert [word["start"] for word in merged[0]["words"]] == sorted(word["start"] for word in merged[0]["words"])


def test_real_clip_construction_exposes_pause_to_delivery_features():
    clips = pipeline.sentence_boundary_micro_cuts([{"start": 0, "end": 3, "text": "one two.", "words": [
        {"start": 0, "end": .2, "word": " one"},
        {"start": 1.5, "end": 1.8, "word": " two."},
    ]}])
    assert delivery_features(clips[0]).excessive_pause is True


def test_real_clip_construction_without_words_remains_safe():
    clips = pipeline.sentence_boundary_micro_cuts([{"start": 0, "end": 2, "text": "safe fallback."}])
    assert "words" not in clips[0]
    assert delivery_features(clips[0]).excessive_pause is False

def valid_json(**changes):
    data = {"winner_id": "a", "confidence": .8, "abstain": False, "reason": "safe",
            "candidate_scores": [{"candidate_id": cid, "delivery_score": .8,
              "visual_performance_score": .7, "clarity_score": .9,
              "sales_effectiveness_score": .8, "overall_score": (.9 if cid == "a" else .8), "reason": "safe"}
             for cid in ("a", "b")]}
    data.update(changes)
    import json
    return json.dumps(data)


def provider_candidates():
    features = delivery_features(candidate())
    return [TakeJudgeCandidate(cid, "HOOK", "text", 1, features, (), 0) for cid in ("a", "b")]


@pytest.mark.parametrize("raw", [
    valid_json(winner_id="unknown"),
    valid_json(candidate_scores=[]),
    valid_json(candidate_scores=[{"candidate_id":"a","delivery_score":2,"visual_performance_score":.5,"clarity_score":.5,"sales_effectiveness_score":.5,"overall_score":.5}]),
    "malformed",
])
def test_v2_validation_abstains_on_malformed_unknown_missing_or_bad_score(monkeypatch, raw):
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: raw)
    with pytest.raises(OpenAIResponseValidationError):
        provider.judge_takes_v2("gpt-4o-mini", "HOOK", provider_candidates(), [])


def test_v2_valid_winner_abstention_and_duplicate_ids(monkeypatch):
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: valid_json())
    assert provider.judge_takes_v2("m", "HOOK", provider_candidates(), []).winner_id == "a"
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: valid_json(winner_id=None, abstain=True))
    assert provider.judge_takes_v2("m", "HOOK", provider_candidates(), []).abstain
    with pytest.raises(OpenAIResponseValidationError):
        provider.judge_takes_v2("m", "HOOK", [provider_candidates()[0]] * 2, [])



def _scores_json(a_score, b_score, winner="a"):
    import json
    data = json.loads(valid_json(winner_id=winner))
    data["candidate_scores"][0]["overall_score"] = a_score
    data["candidate_scores"][1]["overall_score"] = b_score
    return json.dumps(data)


def test_v2_accepts_unique_highest_winner(monkeypatch):
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: _scores_json(.9, .8))
    result_value = provider.judge_takes_v2("m", "HOOK", provider_candidates(), [])
    assert result_value.winner_id == "a" and not result_value.abstain


def test_v2_rejects_winner_below_another_score(monkeypatch):
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: _scores_json(.7, .9))
    with pytest.raises(OpenAIResponseValidationError):
        provider.judge_takes_v2("m", "HOOK", provider_candidates(), [])


def test_v2_tied_top_scores_abstain(monkeypatch):
    monkeypatch.setattr(provider, "_chat", lambda *args, **kwargs: _scores_json(.9, .9))
    result_value = provider.judge_takes_v2("m", "HOOK", provider_candidates(), [])
    assert result_value.abstain and result_value.winner_id is None


def test_inconsistent_provider_output_never_removes_candidate(monkeypatch):
    group, outside, _ = configure_pipeline(monkeypatch, OpenAIResponseValidationError("safe"))
    pipeline.run_take_judge(group + [outside], "session", "input")
    assert all(item["meta"]["keep"] for item in group + [outside])

def configure_pipeline(monkeypatch, outcome):
    group = [candidate("a", 0, 2), candidate("b", 3, 5)]
    outside = candidate("outside", 8, 10)
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", True)
    monkeypatch.setattr(pipeline, "is_openai_available", lambda: True)
    monkeypatch.setattr(pipeline, "find_sibling_groups", lambda clips: [group])
    monkeypatch.setattr(pipeline, "sample_candidate_frames", lambda c, *args: types.SimpleNamespace(
        successful_frame_timestamps=(), image_content=()))
    calls = []
    def judge(*args, **kwargs):
        calls.append(1)
        if isinstance(outcome, Exception): raise outcome
        return outcome
    monkeypatch.setattr(pipeline, "judge_takes_v2", judge)
    return group, outside, calls


@pytest.mark.parametrize("outcome,removed", [(result(), ["b"]), (result(None, abstain=True), []), (result(confidence=.69), [])])
def test_selection_winner_abstain_and_low_confidence(monkeypatch, outcome, removed):
    group, outside, calls = configure_pipeline(monkeypatch, outcome)
    pipeline.run_take_judge(group + [outside], "/private/session", "/private/input.mp4")
    assert [item["id"] for item in group if not item["meta"]["keep"]] == removed
    assert group[0]["meta"]["keep"] and outside["meta"]["keep"] and len(calls) == 1


def test_provider_failure_keeps_all_and_logs_are_private(monkeypatch, caplog):
    group, outside, _ = configure_pipeline(monkeypatch, OpenAIProviderError("safe"))
    group[0]["text"] = "SECRET_TRANSCRIPT"
    with caplog.at_level(logging.WARNING): pipeline.run_take_judge(group + [outside], "/private/session", "/private/input.mp4")
    assert all(item["meta"]["keep"] for item in group + [outside])
    assert "SECRET_TRANSCRIPT" not in caplog.text and "base64" not in caplog.text and "/private" not in caplog.text


def test_group_and_take_limits_and_one_call_per_group(monkeypatch):
    groups = [[candidate(f"{letter}{i}", i * 3, i * 3 + 2) for i in range(4)] for letter in "abc"]
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", True); monkeypatch.setattr(pipeline, "is_openai_available", lambda: True)
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_MAX_GROUPS", 2); monkeypatch.setattr(pipeline, "TAKE_JUDGE_MAX_TAKES", 2)
    monkeypatch.setattr(pipeline, "find_sibling_groups", lambda clips: groups)
    monkeypatch.setattr(pipeline, "sample_candidate_frames", lambda c, *args: types.SimpleNamespace(successful_frame_timestamps=(), image_content=()))
    seen = []
    def judge(model, slot, candidates, frames, **kwargs):
        seen.append([c.candidate_id for c in candidates]); return TakeJudgeV2Result(None, (), .9, True, "tie")
    monkeypatch.setattr(pipeline, "judge_takes_v2", judge)
    pipeline.run_take_judge(sum(groups, []), "session", "input")
    assert seen == [["a0", "a1"], ["b0", "b1"]]


def test_defaults_remain_disabled_and_model_unchanged():
    from worker.models.config import load_model_config
    config = load_model_config().take_judge
    assert not config.enabled and config.model_name == "gpt-4o-mini" and config.frame_count == 1 and config.min_confidence == .70
