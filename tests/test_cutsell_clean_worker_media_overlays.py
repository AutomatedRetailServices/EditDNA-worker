from types import SimpleNamespace

import cutsell_worker.media_overlay_render as compositor
import cutsell_worker.overlay_uploads as uploads
from cutsell_worker.contracts import MediaOverlay, TextOverlay
from cutsell_worker.media_overlay_render import LocalMediaOverlay
from cutsell_worker.overlay_edits import add_media_overlay, remove_media_overlay, update_media_overlay
from cutsell_worker.serde import draft_from_dict


def _draft():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "p1",
        "strategy": "mixed",
        "selected": [{
            "clip_id": "c1", "source_asset_id": "s1", "source_order": 0,
            "start": 0.0, "end": 5.0, "text": "hello", "caption_text": "hello", "selected": True,
        }],
        "alternates": [], "discarded": [], "diagnostics": {},
    }


class FakeS3:
    def generate_presigned_post(self, **kwargs):
        self.kwargs = kwargs
        return {"url": "https://upload.invalid", "fields": {"key": kwargs["Key"]}}


def test_overlay_upload_supports_photo_and_video_with_project_scope(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", lambda: SimpleNamespace(s3_bucket="bucket", aws_region="us-east-1"))
    client = FakeS3()
    result = uploads.create_overlay_presigned_upload(
        project_id="p1", user_id="u1", original_name="product.png",
        content_type="image/png", size_bytes=1234, client=client,
    )
    assert result["kind"] == "photo"
    assert result["uri"].startswith("s3://bucket/cutsell/overlay-assets/")
    assert result["fields"]["key"].startswith("cutsell/overlay-assets/")


def test_overlay_edits_support_position_resize_trim_and_mute():
    added = add_media_overlay(
        _draft(), kind="video", uri="s3://bucket/cutsell/overlay-assets/a/video.mp4",
        start=1.0, end=4.0, x=0.7, y=0.3, width=0.5,
        source_start=2.0, source_end=5.0, mute_audio=True,
    )
    item = added["media_overlays"][0]
    overlay_id = item["overlay_id"]
    assert item["source_start"] == 2.0 and item["source_end"] == 5.0

    updated = update_media_overlay(added, overlay_id, x=0.4, width=0.7, source_start=2.5, source_end=5.5, mute_audio=False)
    item = updated["media_overlays"][0]
    assert item["x"] == 0.4 and item["width"] == 0.7
    assert item["mute_audio"] is False

    removed = remove_media_overlay(updated, overlay_id)
    assert removed["media_overlays"] == []


def test_serde_preserves_photo_video_overlay_track():
    payload = _draft()
    payload["media_overlays"] = [{
        "overlay_id": "ov1", "kind": "photo", "uri": "s3://bucket/path/image.png",
        "start": 0.5, "end": 2.0, "x": 0.5, "y": 0.5, "width": 0.4,
        "source_start": 0.0, "source_end": None, "mute_audio": True,
    }]
    parsed = draft_from_dict(payload)
    assert parsed.media_overlays[0].kind == "photo"
    assert parsed.media_overlays[0].width == 0.4


def test_compositor_builds_photo_video_overlay_and_audio_mix(monkeypatch, tmp_path):
    photo = LocalMediaOverlay(
        MediaOverlay("photo1", "photo", "s3://x/photo.png", 0.5, 2.0, x=0.5, y=0.2, width=0.3),
        str(tmp_path / "photo.png"),
    )
    video = LocalMediaOverlay(
        MediaOverlay("video1", "video", "s3://x/video.mp4", 1.0, 4.0, x=0.7, y=0.6, width=0.4, source_start=2.0, source_end=5.0, mute_audio=False),
        str(tmp_path / "video.mp4"),
    )
    monkeypatch.setattr(compositor, "probe_media", lambda _path: SimpleNamespace(has_audio=True))
    command = compositor.build_final_overlay_command(
        "joined.mp4", "out.mp4",
        media_overlays=(photo, video),
        text_overlays=(TextOverlay("txt", "SALE", 0.2, 1.0),),
        ass_path=str(tmp_path / "text.ass"),
    )
    joined = " ".join(command)
    assert "-loop 1" in joined
    assert "overlay=" in joined
    assert "scale=324:-1" in joined
    assert "scale=432:-1" in joined
    assert "adelay=1000|1000" in joined
    assert "amix=inputs=2" in joined
    assert "ass='" in joined
