"""D-043 (CutSell Modal GPU execution -- first live validation): pure
config/validation tests for the Modal backend. No `modal` package
dependency -- see modal_gpu_config.py's own module docstring."""
from __future__ import annotations

import pytest

import modal_gpu_config as cfg


def test_approved_pool_is_l4_only():
    assert cfg.APPROVED_MODAL_GPU_TYPES == ("L4",)


def test_excluded_pool_names_the_expensive_gpus():
    for excluded in ("A100", "A100-80GB", "H100", "H200", "L40S"):
        assert excluded in cfg.EXCLUDED_MODAL_GPU_TYPES
        assert excluded not in cfg.APPROVED_MODAL_GPU_TYPES


def test_require_modal_gpu_type_accepts_l4():
    cfg.require_modal_gpu_type("L4")  # must not raise


@pytest.mark.parametrize("excluded", ["A100", "A100-80GB", "H100", "H200", "L40S"])
def test_require_modal_gpu_type_rejects_excluded_types_by_name(excluded):
    with pytest.raises(ValueError, match=excluded):
        cfg.require_modal_gpu_type(excluded)


def test_require_modal_gpu_type_rejects_unknown_type():
    with pytest.raises(ValueError):
        cfg.require_modal_gpu_type("some-made-up-gpu")


def test_require_modal_timeout_accepts_default():
    cfg.require_modal_timeout(cfg.DEFAULT_MODAL_TIMEOUT_S)  # must not raise


def test_require_modal_timeout_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        cfg.require_modal_timeout(0)
    with pytest.raises(ValueError):
        cfg.require_modal_timeout(-5)


def test_require_modal_timeout_rejects_above_ceiling():
    with pytest.raises(ValueError):
        cfg.require_modal_timeout(cfg.MAX_MODAL_SMOKE_TEST_TIMEOUT_S + 1)


def test_require_modal_timeout_accepts_exactly_the_ceiling():
    cfg.require_modal_timeout(cfg.MAX_MODAL_SMOKE_TEST_TIMEOUT_S)  # must not raise


def test_require_modal_token_env_passes_with_both_set():
    cfg.require_modal_token_env({"MODAL_TOKEN_ID": "id-fake", "MODAL_TOKEN_SECRET": "secret-fake"})


def test_require_modal_token_env_names_missing_token_id():
    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
        cfg.require_modal_token_env({"MODAL_TOKEN_SECRET": "secret-fake"})


def test_require_modal_token_env_names_missing_token_secret():
    with pytest.raises(RuntimeError, match="MODAL_TOKEN_SECRET"):
        cfg.require_modal_token_env({"MODAL_TOKEN_ID": "id-fake"})


def test_require_modal_token_env_names_both_when_both_missing():
    with pytest.raises(RuntimeError) as exc_info:
        cfg.require_modal_token_env({})
    assert "MODAL_TOKEN_ID" in str(exc_info.value)
    assert "MODAL_TOKEN_SECRET" in str(exc_info.value)


def test_require_modal_token_env_rejects_blank_values():
    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
        cfg.require_modal_token_env({"MODAL_TOKEN_ID": "   ", "MODAL_TOKEN_SECRET": "secret-fake"})


def test_cutsell_base_image_matches_dockerfile():
    # D-043 image/runtime audit: this must stay byte-for-byte identical to
    # Dockerfile.cutsell.serverless's own FROM line -- any drift here is a
    # silent CUDA/torch version change, which the standing directive
    # requires reporting before doing, not committing quietly.
    with open("Dockerfile.cutsell.serverless") as f:
        dockerfile_from_line = f.readline().strip()
    assert dockerfile_from_line == f"FROM {cfg.CUTSELL_BASE_IMAGE}"


def test_default_timeout_is_within_the_ceiling():
    assert 0 < cfg.DEFAULT_MODAL_TIMEOUT_S <= cfg.MAX_MODAL_SMOKE_TEST_TIMEOUT_S


# --- D-043 full Video00 execution phase --------------------------------

def test_video00_timeout_mirrors_runpod_serverless_raws_own_bound():
    # cutsell-video00-raw-v5-auto-microtrim.yml's own poll deadline is
    # SECONDS+5400 -- this must stay the same number, not a new invention,
    # for the exact same six-minute source video.
    assert cfg.DEFAULT_MODAL_VIDEO00_TIMEOUT_S == 5400
    assert cfg.MAX_MODAL_VIDEO00_TIMEOUT_S == 5400


def test_require_modal_video00_timeout_accepts_the_default():
    cfg.require_modal_video00_timeout(cfg.DEFAULT_MODAL_VIDEO00_TIMEOUT_S)  # must not raise


def test_require_modal_video00_timeout_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        cfg.require_modal_video00_timeout(0)
    with pytest.raises(ValueError):
        cfg.require_modal_video00_timeout(-1)


def test_require_modal_video00_timeout_rejects_above_ceiling():
    with pytest.raises(ValueError):
        cfg.require_modal_video00_timeout(cfg.MAX_MODAL_VIDEO00_TIMEOUT_S + 1)


def test_require_modal_video00_timeout_is_independent_of_smoke_test_ceiling():
    # Widening the Video00 ceiling must never silently widen the minimal
    # smoke test's own (much smaller) ceiling, and vice versa.
    assert cfg.MAX_MODAL_VIDEO00_TIMEOUT_S > cfg.MAX_MODAL_SMOKE_TEST_TIMEOUT_S
    with pytest.raises(ValueError):
        cfg.require_modal_timeout(cfg.DEFAULT_MODAL_VIDEO00_TIMEOUT_S)


def test_cutsell_apt_packages_match_dockerfile():
    # D-043 Section 3: never a second, hand-typed dependency list drifting
    # from Dockerfile.cutsell.serverless's own apt-get install block.
    with open("Dockerfile.cutsell.serverless") as f:
        dockerfile_text = f.read()
    start = dockerfile_text.index("apt-get install -y --no-install-recommends")
    end = dockerfile_text.index("&& rm -rf /var/lib/apt/lists/*", start)
    block = dockerfile_text[start:end]
    packages = tuple(
        line.strip().rstrip("\\").strip()
        for line in block.splitlines()[1:]
        if line.strip().strip("\\").strip()
    )
    assert packages == cfg.CUTSELL_APT_PACKAGES


def test_cutsell_requirements_file_path_matches_dockerfile():
    with open("Dockerfile.cutsell.serverless") as f:
        dockerfile_text = f.read()
    assert cfg.CUTSELL_REQUIREMENTS_FILE in dockerfile_text
    import os
    assert os.path.isfile(cfg.CUTSELL_REQUIREMENTS_FILE)


def test_cutsell_runpod_pip_spec_matches_dockerfile():
    with open("Dockerfile.cutsell.serverless") as f:
        dockerfile_text = f.read()
    assert cfg.CUTSELL_RUNPOD_PIP_SPEC in dockerfile_text
