from pathlib import Path
import ast


FORBIDDEN_HEAVY_IMPORTS = {
    "clip",
    "cv2",
    "moviepy",
    "PIL",
    "sentence_transformers",
    "sklearn",
    "torch",
    "torchvision",
}


def test_clean_worker_does_not_import_legacy_heavy_ml_stack():
    violations = []
    for path in Path("cutsell_worker").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                if name in FORBIDDEN_HEAVY_IMPORTS:
                    violations.append(f"{path}:{getattr(node, 'lineno', '?')} imports {name}")
    assert not violations, "Legacy heavy imports leaked into clean CutSell worker: " + "; ".join(violations)


def _requirement_names() -> set[str]:
    names = set()
    for raw in Path("requirements.cutsell.worker.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if not line:
            continue
        name = line
        for separator in ["==", ">=", "<=", "~=", "!=", ">", "<", "["]:
            if separator in name:
                name = name.split(separator, 1)[0]
        names.add(name.strip())
    return names


def test_worker_requirements_do_not_replace_base_torch_cuda_stack_or_install_openai():
    names = _requirement_names()
    forbidden = {
        "torch", "torchvision", "clip", "sentence-transformers",
        "opencv-python", "opencv-python-headless", "moviepy", "openai",
    }
    assert not (names & forbidden)
    for required in {"faster-whisper", "boto3", "redis", "rq"}:
        assert required in names


def test_worker_dockerfile_uses_clean_worker_requirements_and_removes_openai_sdk():
    text = Path("Dockerfile.cutsell.worker").read_text(encoding="utf-8")
    assert "requirements.cutsell.worker.txt" in text
    assert "requirements.txt" not in text.replace("requirements.cutsell.worker.txt", "")
    assert "pip uninstall -y openai" in text
