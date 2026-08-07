from pathlib import Path

from main import app


def test_render_and_docker_use_real_fastapi_entrypoint():
    render = Path("render.yaml").read_text()
    dockerfile = Path("Dockerfile").read_text()
    assert "uvicorn main:app" in render
    assert "uvicorn main:app" in dockerfile
    assert "uvicorn app:app" not in render
    assert "uvicorn app:app" not in dockerfile


def test_main_mounts_v1_draft_routes():
    paths = app.openapi()["paths"]
    assert "/v1/healthz" in paths
    assert "/v1/draft-edits/swap" in paths
    assert "/v1/draft-edits/remove" in paths
    assert "/v1/draft-edits/restore" in paths
    assert "/v1/draft-edits/reorder" in paths
    assert "/v1/draft-edits/captions" in paths
