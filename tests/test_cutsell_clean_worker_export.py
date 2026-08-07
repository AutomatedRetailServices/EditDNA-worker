from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import cutsell_worker.export_job as export_job
import cutsell_worker.exports as exports


def _draft():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "project-1",
        "strategy": "mixed",
        "selected": [{
            "clip_id": "clip-1",
            "source_asset_id": "src-1",
            "source_order": 0,
            "start": 1.0,
            "end": 2.0,
            "text": "hello",
            "caption_text": "hello",
            "semantic_role": "OTHER",
            "selected": True,
        }],
        "alternates": [],
        "discarded": [],
        "diagnostics": {},
    }


class FakeJob:
    def __init__(self):
        self.meta = {}
        self.saved = []
    def save_meta(self):
        self.saved.append(dict(self.meta))


def test_export_job_renders_edited_draft_without_rerunning_ai(monkeypatch, tmp_path):
    fake_job = FakeJob()
    rq_module = ModuleType("rq")
    rq_module.get_current_job = lambda: fake_job
    monkeypatch.setitem(sys.modules, "rq", rq_module)

    validated = []
    monkeypatch.setattr(
        export_job,
        "validate_product_source_uri",
        lambda uri, **kwargs: validated.append((uri, kwargs)) or ("bucket", "key"),
    )
    monkeypatch.setattr(
        export_job,
        "download_source",
        lambda uri, destination: Path(destination).write_bytes(b"source") or destination,
    )
    monkeypatch.setattr(export_job, "build_render_plan", lambda draft, local_paths: ("plan",))

    rendered = []
    def fake_render(plan, output):
        rendered.append((plan, output))
        Path(output).write_bytes(b"mp4")
        return output
    monkeypatch.setattr(export_job, "render_preview", fake_render)
    monkeypatch.setattr(
        export_job,
        "store_export",
        lambda output, **kwargs: {
            "export_uri": "s3://bucket/cutsell/exports/file.mp4",
            "download_url": "https://download.invalid/file.mp4",
            "expires_in": 3600,
            "size_bytes": Path(output).stat().st_size,
        },
    )

    result = export_job.run_export_job({
        "project_id": "project-1",
        "user_id": "user-1",
        "draft": _draft(),
        "sources": [{
            "source_asset_id": "src-1",
            "original_name": "one.mov",
            "uri": "s3://bucket/cutsell/uploads/u/p/one.mov",
        }],
    })

    assert result["state"] == "finished"
    assert result["selected_count"] == 1
    assert result["download_url"].startswith("https://download.invalid/")
    assert len(validated) == 1
    assert validated[0][1] == {"project_id": "project-1", "user_id": "user-1"}
    assert len(rendered) == 1
    assert fake_job.meta["stage"] == "finished"
    assert fake_job.meta["progress_percent"] == 100


class FakeS3:
    def __init__(self):
        self.uploads = []
        self.urls = []
    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.uploads.append((source, bucket, key, ExtraArgs))
    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.urls.append((operation, Params, ExpiresIn))
        return "https://download.invalid/export.mp4"


def test_store_export_scopes_object_and_returns_temporary_download(monkeypatch, tmp_path):
    video = tmp_path / "final.mp4"
    video.write_bytes(b"video-bytes")
    monkeypatch.setattr(
        exports,
        "load_runtime_config",
        lambda: SimpleNamespace(s3_bucket="bucket", aws_region="us-east-1"),
    )
    client = FakeS3()
    result = exports.store_export(
        str(video),
        project_id="project-1",
        user_id="user-1",
        client=client,
    )
    assert result["export_uri"].startswith("s3://bucket/cutsell/exports/")
    assert result["download_url"] == "https://download.invalid/export.mp4"
    assert result["size_bytes"] == len(b"video-bytes")
    assert client.uploads[0][3] == {"ContentType": "video/mp4"}
    assert client.urls[0][0] == "get_object"
