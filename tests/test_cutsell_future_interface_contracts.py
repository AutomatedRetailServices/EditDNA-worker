"""Interface-only contracts from D-024 (PostRenderWatchListenQC, the Sales/
TikTok Shop StyleProfile extension points, Finishing). None of these is
implemented or activated -- these tests only pin the agreed shape so a
future implementation has something stable to build against, and guard
against the active pipeline accidentally importing/activating them.
"""
import dataclasses

from cutsell_worker.finishing_contract import FinishingProvider, FinishingResult, FinishingSpec
from cutsell_worker.post_render_watch_listen_qc import (
    CLIPPED_WORD,
    PostRenderFinding,
    PostRenderQCResult,
    PostRenderWatchListenQCProvider,
)
from cutsell_worker.style_profile_contract import (
    NATURAL_CLEAN,
    RESERVED_ANNOTATION_KEYS,
    TIGHT_SOCIAL,
    TIKTOK_SHOP,
    StyleProfile,
)


def test_post_render_qc_result_shape():
    finding = PostRenderFinding(kind=CLIPPED_WORD, start=1.0, end=1.2, detail={}, routes_to="BoundaryEngine")
    result = PostRenderQCResult(status="FAIL", findings=(finding,))
    assert result.status == "FAIL"
    assert result.findings[0].routes_to == "BoundaryEngine"


def test_post_render_qc_provider_is_a_protocol_no_concrete_implementation_shipped():
    # No concrete provider exists in this codebase -- the Protocol exists
    # only as a future contract.
    assert getattr(PostRenderWatchListenQCProvider, "_is_protocol", False) is True
    assert getattr(FinishingProvider, "_is_protocol", False) is True
    assert getattr(StyleProfile, "_is_protocol", False) is True


def test_finishing_spec_defaults_never_mutate_semantics_by_construction():
    spec = FinishingSpec()
    assert spec.delivery_codec == "h264_aac"
    assert spec.fast_start is True
    assert spec.metadata == {}
    result = FinishingResult(status="PASS", output_path=None, decode_verified=False, detail={})
    assert result.status == "PASS"


def test_style_profile_reserved_annotation_keys_match_canonical_edit_plan_field():
    from cutsell_worker.canonical_edit_plan import EditPlanClip

    # EditPlanClip.annotations is the dormant extension point; this pins
    # that the reserved vocabulary is at least consistent in naming intent,
    # without ever populating it from Clean Cut Core V1.
    clip = EditPlanClip(
        clip_id="a", source_asset_id="src", source_order=0, start=0.0, end=1.0,
        text="hi", idea_id=None, is_composite_piece=False,
        protected_leading_word=None, protected_trailing_word=None,
    )
    assert clip.annotations == {}
    assert RESERVED_ANNOTATION_KEYS
    assert len({NATURAL_CLEAN, TIGHT_SOCIAL, TIKTOK_SHOP}) == 3


def test_no_active_pipeline_module_imports_the_dormant_contracts():
    # Grep-style guard: these three modules must not be imported by the
    # active Clean Cut Core V1 pipeline entry points. A future activation
    # is a deliberate decision, not an accidental import.
    import ast
    import pathlib

    forbidden = {"post_render_watch_listen_qc", "style_profile_contract", "finishing_contract"}
    active_modules = ["universal_clean_cut.py", "pipeline.py", "composite_resolver.py", "brain_runtime.py"]
    root = pathlib.Path(__file__).resolve().parents[1] / "cutsell_worker"
    for name in active_modules:
        tree = ast.parse((root / name).read_text())
        imported = {
            alias.name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {
            node.module.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not (imported & forbidden), f"{name} imports a dormant D-024 contract: {imported & forbidden}"
