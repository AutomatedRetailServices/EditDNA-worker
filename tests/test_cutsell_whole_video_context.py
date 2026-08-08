from types import SimpleNamespace

from cutsell_worker.contracts import SourceAsset, TranscriptSegment
from cutsell_worker.source_sampling import sample_source_frames
from cutsell_worker.whole_video_analysis import safe_whole_video_analyze
from cutsell_worker.whole_video_openai import OpenAIWholeVideoProvider


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


def _source():
    return SourceAsset(
        source_asset_id="src-1",
        project_id="p",
        user_id="u",
        original_name="raw.mp4",
        source_order=0,
        duration_sec=30.0,
        uri="s3://bucket/raw.mp4",
    )


def test_whole_source_sampling_covers_full_video_and_is_bounded(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    frames = sample_source_frames(
        "raw.mp4",
        source_asset_id="src-1",
        duration_sec=30.0,
        output_dir=str(tmp_path),
        runner=runner,
    )
    assert 8 <= len(frames) <= 48
    assert frames[0].relative_position < 0.05
    assert frames[-1].relative_position > 0.95
    assert all(frame.source_asset_id == "src-1" for frame in frames)


def test_whole_video_provider_returns_auditable_temporal_context(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    client = FakeClient(
        '{"sources":[{"source_asset_id":"src-1","summary":"Creator retries hook then demos product",'
        '"dominant_style":"demo","creator_intent":"show product result",'
        '"events":[{"start":2.0,"end":4.0,"kind":"retry","confidence":0.95,"description":"visible restart"},'
        '{"start":10.0,"end":18.0,"kind":"product_demo","confidence":0.9,"description":"product demonstration"}]}]}'
    )
    provider = OpenAIWholeVideoProvider(client_factory=lambda: client)
    from cutsell_worker.source_sampling import SourceFrameSample
    context = provider.analyze(
        (_source(),),
        (TranscriptSegment("src-1", 1.0, 5.0, "let me start again"),),
        (SourceFrameSample("src-1", 3.0, str(image), 0.1),),
    )
    assert context.status.status == "applied"
    assert context.sources[0].dominant_style == "demo"
    assert [event.kind for event in context.sources[0].events] == ["retry", "product_demo"]
    assert "product_demo" in context.compact_text()
    call = client.responses.calls[0]
    image_parts = [
        part
        for message in call["input"]
        for part in message["content"]
        if part["type"] == "input_image"
    ]
    assert image_parts and image_parts[0]["detail"] == "low"


def test_whole_video_provider_failure_is_non_destructive():
    class Broken:
        def analyze(self, sources, transcripts, samples):
            raise TimeoutError("unavailable")
    result = safe_whole_video_analyze(Broken(), (_source(),), (), ())
    assert result.sources == ()
    assert result.status.status == "provider_error"
    assert result.status.reason == "TimeoutError"
