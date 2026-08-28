from dataclasses import replace

import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, SemanticRole, Word
from cutsell_worker.selection_boundary_contract import (
    enforce_selection_contract,
    freeze_selection_contract,
)


def _clip(clip_id, start, end, text):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=(Word(text, start + 0.01, end - 0.01),),
        semantic_role=SemanticRole.STORY,
        selected=True,
    )


def _draft(selected):
    return DraftTimeline(selected=tuple(selected), discarded=(), alternates=(), diagnostics={})


def test_boundary_split_preserves_frozen_semantic_stream():
    frozen = freeze_selection_contract(_draft((_clip("p", 10.0, 14.0, "uno dos tres cuatro"),)))
    left = _clip("p__left", 10.0, 11.8, "uno dos")
    right = _clip("p__right", 12.2, 14.0, "tres cuatro")
    verified = enforce_selection_contract(replace(frozen, selected=(left, right)))
    assert verified.diagnostics["selection_boundary_contract"]["status"] == "verified"
    assert verified.diagnostics["selection_boundary_contract"]["final_selected_fragment_count"] == 2


def test_boundary_timing_only_change_preserves_contract():
    original = _clip("p", 10.0, 14.0, "uno dos tres")
    frozen = freeze_selection_contract(_draft((original,)))
    trimmed = replace(original, start=10.2, end=13.8)
    verified = enforce_selection_contract(replace(frozen, selected=(trimmed,)))
    assert verified.diagnostics["selection_boundary_contract"]["status"] == "verified"


def test_boundary_content_loss_hard_fails():
    frozen = freeze_selection_contract(_draft((_clip("p", 10.0, 14.0, "uno dos tres cuatro"),)))
    corrupted = _clip("p__bad", 10.0, 13.0, "uno dos cuatro")
    with pytest.raises(RuntimeError, match="Boundary changed frozen Selection semantic content"):
        enforce_selection_contract(replace(frozen, selected=(corrupted,)))


def test_boundary_reorder_hard_fails():
    frozen = freeze_selection_contract(_draft((
        _clip("a", 10.0, 12.0, "uno dos"),
        _clip("b", 13.0, 15.0, "tres cuatro"),
    )))
    with pytest.raises(RuntimeError, match="Boundary changed frozen Selection semantic content"):
        enforce_selection_contract(replace(frozen, selected=(
            _clip("b", 10.0, 12.0, "tres cuatro"),
            _clip("a", 13.0, 15.0, "uno dos"),
        )))
