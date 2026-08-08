from pathlib import Path


def test_cutsell_render_blueprint_is_api_only_and_isolated_from_legacy_services():
    text = Path("render.cutsell.yaml").read_text()
    assert "name: cutsell-api" in text
    assert "uvicorn cutsell_app.main:app" in text
    assert "requirements.cutsell.api.txt" in text
    assert "CUTSELL_AUTH_REQUIRED" in text
    assert 'value: "1"' in text
    assert "CUTSELL_CLEAN_CUT_JUDGE" in text
    assert 'value: "0"' in text
    assert "name: cutsell-worker" not in text
    assert "rq.cli worker" not in text
    assert "name: editdna-web" not in text
    assert "name: editdna-worker" not in text
    assert "autoDeploy: false" in text
    assert "RunPod GPU infrastructure" in text


def test_cutsell_container_uses_clean_api_entrypoint():
    text = Path("Dockerfile.cutsell").read_text()
    assert "cutsell_app.main:app" in text
    assert "COPY cutsell_worker" in text
    assert "COPY cutsell_app" in text
    assert "requirements.cutsell.api.txt" in text
    assert "requirements.txt" not in text.replace("requirements.cutsell.api.txt", "")
    assert "main:app" not in text.replace("cutsell_app.main:app", "")


def test_cutsell_api_dependencies_do_not_pull_gpu_ml_stack():
    text = Path("requirements.cutsell.api.txt").read_text().lower()
    for forbidden in (
        "faster-whisper",
        "opencv",
        "sentence-transformers",
        "scikit-learn",
        "moviepy",
        "imageio-ffmpeg",
        "clip.git",
    ):
        assert forbidden not in text
    for required in ("fastapi", "uvicorn", "boto3", "redis", "rq"):
        assert required in text


def test_cutsell_gpu_worker_has_isolated_container_and_immutable_build_workflow():
    dockerfile = Path("Dockerfile.cutsell.worker").read_text()
    workflow = Path(".github/workflows/cutsell-worker-build.yml").read_text()
    assert "madiator2011/better-pytorch:cuda12.4-torch2.6.0" in dockerfile
    assert "COPY cutsell_worker" in dockerfile
    assert "rq.cli worker" in dockerfile
    assert "ghcr.io/automatedretailservices/cutsell-worker:${{ github.sha }}" in workflow
    assert "Dockerfile.cutsell.worker" in workflow
    assert "docker push" in workflow
