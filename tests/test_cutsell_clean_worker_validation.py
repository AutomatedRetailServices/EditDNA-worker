from types import SimpleNamespace

import pytest

import cutsell_worker.validation as validation


class FakeS3:
    def list_objects_v2(self, **kwargs):
        assert kwargs["Bucket"] == "bucket"
        assert kwargs["Prefix"] == "Editdna bloopers videos/"
        return {
            "Contents": [
                {"Key": "Editdna bloopers videos/a.mov", "Size": 100000},
                {"Key": "Editdna bloopers videos/readme.txt", "Size": 100000},
                {"Key": "Editdna bloopers videos/tiny.mp4", "Size": 10},
                {"Key": "Editdna bloopers videos/b.mp4", "Size": 200000},
            ]
        }


def test_validation_inventory_is_bounded_and_video_only(monkeypatch):
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


@pytest.mark.parametrize("prefix", ["/absolute/", "../escape/", "unsafe\\path/"])
def test_validation_rejects_unsafe_prefix(prefix):
    with pytest.raises(ValueError):
        validation._safe_prefix(prefix)
