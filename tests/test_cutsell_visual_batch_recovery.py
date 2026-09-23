from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.visual_analysis import VisualObservation
from cutsell_worker.visual_openai import OpenAIVisualProvider


def _take(clip_id: str, start: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 1.0,
        text=f"take {clip_id}",
    )


def _observation(clip_id: str) -> VisualObservation:
    return VisualObservation(
        clip_id=clip_id,
        face_visibility=0.8,
        eye_contact=0.8,
        framing_quality=0.8,
        product_visibility=0.0,
        motion_stability=0.8,
        continuity=0.8,
        visual_fumble=0.0,
        expression_naturalness=0.8,
        gesture_naturalness=0.8,
        delivery_energy=0.8,
        distraction_risk=0.0,
    )


class SplitRecoveryProvider(OpenAIVisualProvider):
    def __init__(self):
        super().__init__(batch_size=4)
        self.calls = []

    def _analyze_batch(self, takes, frames_by_clip):
        self.calls.append(tuple(t.clip_id for t in takes))
        if len(takes) > 1:
            raise ValueError("malformed multi-take response")
        return (_observation(takes[0].clip_id),)


class SingleRetryProvider(OpenAIVisualProvider):
    def __init__(self):
        super().__init__(batch_size=1)
        self.attempts = 0

    def _analyze_batch(self, takes, frames_by_clip):
        self.attempts += 1
        if self.attempts == 1:
            raise ValueError("first malformed response")
        return (_observation(takes[0].clip_id),)


def test_visual_batch_failure_splits_until_each_take_is_recovered():
    provider = SplitRecoveryProvider()
    takes = (_take("a", 0.0), _take("b", 1.0), _take("c", 2.0), _take("d", 3.0))

    result = provider.analyze(takes, ())

    assert result.status == ProviderStatus("openai", True, True, "applied")
    assert tuple(item.clip_id for item in result.observations) == ("a", "b", "c", "d")
    assert provider.calls[0] == ("a", "b", "c", "d")
    assert ("a",) in provider.calls
    assert ("d",) in provider.calls


def test_visual_single_take_gets_one_fresh_retry_before_failing():
    provider = SingleRetryProvider()
    result = provider.analyze((_take("a", 0.0),), ())

    assert result.status.status == "applied"
    assert tuple(item.clip_id for item in result.observations) == ("a",)
    assert provider.attempts == 2
