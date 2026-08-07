from types import SimpleNamespace

import pytest

import cutsell_worker.validation as validation


class FakeS3:
    def list_objects_v2(self, **kwargs):
        assert kwargs["Bucket"] == "bucket"
        assert kwargs["Prefix"] == "Editdna bloopers videos/"
        return {
            "Contents": [
                {"Key": "Editdna bloopers videos/._Bloppers1.MP4", "Size": 4096},
                {"Key": "Editdna bloopers videos/.DS_Store", "Size": 100000},
                {"Key": "Editdna bloopers videos/a.mov", "Size": 100000},
                {"Key": "Editdna bloopers videos/readme.txt", "Size": 100000},
                {"Key": "Editdna bloopers videos/tiny.mp4", "Size": 10},
                {"Key": "Editdna bloopers videos/b.mp4", "Size": 200000},
            ]
        }


def test_validation_inventory_is_bounded_video_only_and_ignores_macos_metadata(monkeypatch):
    monkeypatch.setattr(
        validation,
        "load_runtime_config",
        lambda: SimpleNamespace(s3_bucket="bucket", aws_region="us-east-1"),
    )
    items = validation.list_validation_videos(s3=FakeS3(), limit=2)
    assert [item["key"] for item in items] == [
        "Editdna bloopers videos/a.mov",
        "Editdna bloopers videos/b.mp4",
    ]


def test_validation_rejects_appledouble_as_explicit_video_key():
    assert validation._is_real_video_key("Editdna bloopers videos/._Bloppers1.MP4") is False


@pytest.mark.parametrize("prefix", ["/absolute/", "../escape/", "unsafe\\path/"])
def test_validation_rejects_unsafe_prefix(prefix):
    with pytest.raises(ValueError):
        validation._safe_prefix(prefix)


@pytest.mark.parametrize(
    "start,end",
    [(-1.0, 2.0), (5.0, 5.0), (6.0, 5.0)],
)
def test_validation_window_rejects_invalid_ranges(start, end):
    with pytest.raises(ValueError, match="0 <= start < end"):
        validation._extract_validation_window(
            "source.mp4", "window.mp4", start_sec=start, end_sec=end, runner=lambda *a, **k: None
        )


def test_validation_window_rejects_over_180_seconds():
    with pytest.raises(ValueError, match="exceeds bounded duration"):
        validation._extract_validation_window(
            "source.mp4",
            "window.mp4",
            start_sec=10.0,
            end_sec=191.0,
            runner=lambda *a, **k: None,
        )


def test_validation_window_builds_exact_bounded_ffmpeg_command():
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    result = validation._extract_validation_window(
        "source.mp4",
        "window.mp4",
        start_sec=208.0,
        end_sec=220.0,
        runner=fake_runner,
    )
    assert result == "window.mp4"
    command, kwargs = calls[0]
    assert command[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert command[command.index("-ss") + 1] == "208.000"
    assert command[command.index("-t") + 1] == "12.000"
    assert ["-map", "0:v:0"] == command[command.index("-map") : command.index("-map") + 2]
    second_map = command.index("-map", command.index("-map") + 1)
    assert command[second_map : second_map + 2] == ["-map", "0:a?"]
    assert command[-1] == "window.mp4"
    assert kwargs == {"capture_output": True, "check": True}
