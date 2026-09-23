from types import SimpleNamespace

from cutsell_worker.contracts import SourceAsset, TranscriptSegment
from cutsell_worker.source_sampling import sample_source_frames
from cutsell_worker.whole_video_analysis import safe_whole_video_analyze
from cutsell_worker.whole_video_openai import OpenAIWholeVideoProvider


class FakeResponses:
    def __init__(self, output_text):
        if isinstance(output_text, (list, tuple)):
            self.output_texts = list(output_text)
        else:
            self.output_texts = [output_text]
        self.calls = []

    def create(self, **kwargs):
        index = min(len(self.calls), len(self.output_texts) - 1)
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_texts[index])


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


def _valid_whole_video_json():
    return (
        '{"sources":[{"source_asset_id":"src-1","summary":"Creator retries hook then demos product",'
        '"dominant_style":"demo","creator_intent":"show product result","edit_mode":"sales","sales_intent":0.96,'
        '"main_topic":"portable blender","product_or_subject":"blender","story_logic":"visual hook then demo and result",'
        '"events":[{"start":2.0,"end":4.0,"kind":"retry","confidence":0.95,"description":"visible restart"},'
        '{"start":10.0,"end":18.0,"kind":"product_demo","confidence":0.9,"description":"product demonstration"}]}]}'
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
    assert 12 <= len(frames) <= 120
    assert len(frames) >= 40  # 30 sec source is observed at ~0.75 sec cadence
    assert frames[0].relative_position < 0.05
    assert frames[-1].relative_position > 0.95
    assert all(frame.source_asset_id == "src-1" for frame in frames)


def test_whole_video_provider_returns_auditable_temporal_context(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    client = FakeClient(_valid_whole_video_json())
    provider = OpenAIWholeVideoProvider(client_factory=lambda: client)
    from cutsell_worker.source_sampling import SourceFrameSample
    context = provider.analyze(
        (_source(),),
        (TranscriptSegment("src-1", 1.0, 5.0, "let me start again"),),
        (SourceFrameSample("src-1", 3.0, str(image), 0.1),),
    )
    assert context.status.status == "applied"
    assert context.status.reason == ""
    assert context.sources[0].dominant_style == "demo"
    assert context.sources[0].edit_mode == "sales"
    assert context.sources[0].sales_intent == 0.96
    assert context.dominant_edit_mode == "sales"
    assert [event.kind for event in context.sources[0].events] == ["retry", "product_demo"]
    assert "product_demo" in context.compact_text()
    assert "story_logic=visual hook then demo and result" in context.compact_text()
    call = client.responses.calls[0]
    image_parts = [
        part
        for message in call["input"]
        for part in message["content"]
        if part["type"] == "input_image"
    ]
    assert image_parts and image_parts[0]["detail"] == "low"


def test_malformed_whole_video_json_gets_one_format_only_repair(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    malformed = _valid_whole_video_json()[:-1]  # remove final closing brace
    client = FakeClient((malformed, _valid_whole_video_json()))
    provider = OpenAIWholeVideoProvider(client_factory=lambda: client)
    from cutsell_worker.source_sampling import SourceFrameSample

    context = provider.analyze(
        (_source(),),
        (TranscriptSegment("src-1", 1.0, 5.0, "let me start again"),),
        (SourceFrameSample("src-1", 3.0, str(image), 0.1),),
    )

    assert context.status.status == "applied"
    assert context.status.reason == "json_format_repaired"
    assert len(client.responses.calls) == 2
    repair_prompt = client.responses.calls[1]["input"][0]["content"]
    assert "Repair ONLY JSON syntax/formatting" in repair_prompt
    assert "Do not reinterpret the video" in repair_prompt


def test_bad_whole_video_repair_still_fails_open_with_detail(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    client = FakeClient(("{bad", "still bad"))
    provider = OpenAIWholeVideoProvider(client_factory=lambda: client)
    from cutsell_worker.source_sampling import SourceFrameSample

    context = safe_whole_video_analyze(
        provider,
        (_source(),),
        (),
        (SourceFrameSample("src-1", 3.0, str(image), 0.1),),
    )

    assert context.sources == ()
    assert context.status.status == "provider_error"
    assert "JSONDecodeError" in context.status.reason
    assert len(client.responses.calls) == 2


def test_whole_video_provider_failure_is_non_destructive():
    class Broken:
        def analyze(self, sources, transcripts, samples):
            raise TimeoutError("unavailable")
    result = safe_whole_video_analyze(Broken(), (_source(),), (), ())
    assert result.sources == ()
    assert result.status.status == "provider_error"
    assert result.status.reason == "TimeoutError: unavailable"
