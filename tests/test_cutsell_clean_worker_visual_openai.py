from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.frame_sampling import FrameSample
from cutsell_worker.visual_openai import OpenAIVisualProvider


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def _take():
    return CandidateTake(
        clip_id="clip-1",
        source_asset_id="src-1",
        source_order=0,
        start=0.0,
        end=2.0,
        text="This serum is amazing",
    )


def test_openai_visual_adapter_uses_image_input_and_maps_scores(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-bytes")
    client = FakeClient(
        '{"clips":[{"id":"clip-1","face_visibility":0.9,"eye_contact":0.8,'
        '"framing_quality":0.7,"product_visibility":0.6,"motion_stability":0.85,'
        '"continuity":0.75,"visual_fumble":0.1}]}'
    )
    provider = OpenAIVisualProvider(client_factory=lambda: client)
    result = provider.analyze(
        (_take(),),
        (FrameSample("clip-1", "src-1", 1.0, str(image)),),
    )
    assert result.status.status == "applied"
    assert result.observations[0].face_visibility == 0.9
    call = client.responses.calls[0]
    assert call["model"] == "gpt-4o-mini"
    image_parts = [
        part
        for message in call["input"]
        for part in message["content"]
        if part["type"] == "input_image"
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert image_parts[0]["detail"] == "low"


def test_openai_visual_adapter_rejects_omitted_take():
    client = FakeClient('{"clips":[]}')
    provider = OpenAIVisualProvider(client_factory=lambda: client)
    try:
        provider.analyze((_take(),), ())
    except ValueError as exc:
        assert "omitted" in str(exc)
    else:
        raise AssertionError("visual provider must not silently omit takes")
