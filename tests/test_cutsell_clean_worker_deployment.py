from pathlib import Path


def test_cutsell_render_blueprint_is_isolated_from_legacy_services():
    text = Path("render.cutsell.yaml").read_text()
    assert "name: cutsell-api" in text
    assert "name: cutsell-worker" in text
    assert "uvicorn cutsell_app.main:app" in text
    assert "rq.cli worker" in text
    assert "cutsell'" in text
    assert "name: editdna-web" not in text
    assert "name: editdna-worker" not in text
    assert "autoDeploy: false" in text


def test_cutsell_container_uses_clean_api_entrypoint():
    text = Path("Dockerfile.cutsell").read_text()
    assert "cutsell_app.main:app" in text
    assert "COPY cutsell_worker" in text
    assert "COPY cutsell_app" in text
    assert "main:app" not in text.replace("cutsell_app.main:app", "")
