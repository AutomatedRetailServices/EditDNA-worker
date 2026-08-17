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
    assert "workflow_dispatch" in workflow
    assert "push:" not in workflow


def test_paid_staging_worker_workflows_are_manual_and_explicitly_guarded():
    create = Path(".github/workflows/cutsell-runpod-staging-worker.yml").read_text()
    delete = Path(".github/workflows/cutsell-runpod-staging-worker-delete.yml").read_text()

    for text in (create, delete):
        assert "workflow_dispatch" in text
        assert "push:" not in text
        assert "schedule:" not in text

    assert "APPROVE_PAID_STAGING_WORKER" in create
    assert "cutsell-staging-worker already exists" in create
    assert 'CUTSELL_CLEAN_CUT_JUDGE: "0"' in create
    assert 'CUTSELL_ASR_MODEL: "medium"' in create
    assert "APPROVE_DELETE_STAGING_WORKER" in delete
    assert "Pod ID is not the CutSell staging worker" in delete
    assert "DELETE" in delete


def test_feature_runpod_paid_deploy_is_manual_and_requires_explicit_run_token():
    deploy = Path(".github/workflows/cutsell-feature-runpod-one-shot-deploy.yml").read_text()
    assert "workflow_dispatch:" in deploy
    assert "approval_token:" in deploy
    assert 'test "$APPROVAL_TOKEN" = "RUN' in deploy
    assert "Paid RunPod deploy not explicitly approved" in deploy
    assert "push:" not in deploy
    assert "schedule:" not in deploy
    assert "GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}" in deploy
    assert 'CUTSELL_HYBRID_LLM_ENABLED:"1"' in deploy
    assert 'CUTSELL_HYBRID_PRIMARY_MODEL:"gemini-3.5-flash-lite"' in deploy
    assert 'CUTSELL_HYBRID_MAX_SESSION_USD:"0.0075"' in deploy
    assert 'CUTSELL_HYBRID_TEST_BUDGET_USD:"0.50"' in deploy
    assert 'has("OPENAI_API_KEY")' in deploy


def test_hybrid_unseen_benchmark_is_manual_and_requires_explicit_run_token():
    benchmark = Path(".github/workflows/cutsell-unseen-clean-cut-benchmark.yml").read_text()
    assert "workflow_dispatch:" in benchmark
    assert "approval_token:" in benchmark
    assert 'test "$APPROVAL_TOKEN" = "RUN' in benchmark
    assert "Paid Hybrid benchmark not explicitly approved" in benchmark
    assert "push:" not in benchmark
    assert "schedule:" not in benchmark
    assert "'expected_external_brain_calls_enabled': True" in benchmark
    assert "'hybrid_provider': 'google'" in benchmark
    assert "'hybrid_primary_model': 'gemini-3.5-flash-lite'" in benchmark
    assert "report.get('external_brain_calls_enabled') is not True" in benchmark
    assert "requested <= 0" in benchmark
    assert "availability_ratio < 0.80" in benchmark
