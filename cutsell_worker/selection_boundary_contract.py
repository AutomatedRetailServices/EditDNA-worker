"""Hard contract between Selection and Boundary.

Selection owns semantic membership and spoken content. Boundary may change only exact
source timing and split/coalesce structure while preserving the ordered spoken token
stream. The freeze is benchmark-agnostic and is enforced on every Flow B draft.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)


def _canon(value: str) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def semantic_token_stream(selected) -> tuple[str, ...]:
    clips = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    out: list[str] = []
    for clip in clips:
        out.extend(_canon(token) for token in _TOKEN_RE.findall(str(clip.text or "")))
    return tuple(token for token in out if token)


def _digest(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def freeze_selection_contract(draft, *, plan=None):
    """Freeze the current ``draft.selected`` as the semantic contract
    Boundary must not disturb.

    ``plan`` (D-025, optional): the CanonicalEditPlan FinalEditReviewer
    already returned PASS for. When given, its ``plan_id``/``plan_version``/
    ``semantic_hash`` are recorded onto this freeze's own diagnostics so a
    reader can tell which reviewed plan a given freeze corresponds to --
    "Selection Freeze must reference a specific validated plan," not just
    freeze whatever ``draft.selected`` happens to contain at call time.
    This does NOT hard-assert byte-identical hash equality against the
    plan: ``enforce_complete_idea_boundaries`` legitimately runs between
    FinalEditReviewer's PASS and this call and can restore source-proven
    leading/trailing words, which changes the token stream without
    changing meaning -- a hard equality check here would misfire on that
    expected, documented step. The mismatch (if any) is still recorded
    (``matches_reviewed_plan``) for observability, not enforcement.
    """
    diagnostics = dict(draft.diagnostics or {})
    tokens = semantic_token_stream(draft.selected)
    digest = _digest(tokens)
    contract = {
        "schema_version": "cutsell.selection_boundary_contract.v1",
        "semantic_token_count": len(tokens),
        "semantic_sha256": digest,
        "selected_parent_count_at_freeze": len(tuple(draft.selected)),
        "status": "frozen",
    }
    if plan is not None:
        contract["plan_id"] = plan.plan_id
        contract["plan_version"] = plan.plan_version
        contract["plan_semantic_hash"] = plan.semantic_hash
        contract["matches_reviewed_plan"] = (digest == plan.semantic_hash)
    diagnostics["selection_boundary_contract"] = contract
    return replace(draft, diagnostics=diagnostics)


def enforce_selection_contract(draft):
    diagnostics = dict(draft.diagnostics or {})
    frozen = diagnostics.get("selection_boundary_contract") or {}
    expected = str(frozen.get("semantic_sha256") or "")
    if not expected:
        raise RuntimeError("Selection/Boundary contract missing freeze; refusing unsafe final timeline")
    tokens = semantic_token_stream(draft.selected)
    actual = _digest(tokens)
    if actual != expected:
        raise RuntimeError(
            "Boundary changed frozen Selection semantic content; refusing unsafe final timeline "
            f"expected={expected[:12]} actual={actual[:12]}"
        )
    diagnostics["selection_boundary_contract"] = {
        **dict(frozen),
        "final_semantic_token_count": len(tokens),
        "final_selected_fragment_count": len(tuple(draft.selected)),
        "status": "verified",
    }
    return replace(draft, diagnostics=diagnostics)


def install_selection_freeze() -> None:
    from . import pipeline
    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_selection_freeze_contract", False):
        return

    def build_with_selection_freeze(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, draft=freeze_selection_contract(result.draft))

    build_with_selection_freeze._cutsell_selection_freeze_contract = True
    pipeline.build_flow_b_draft = build_with_selection_freeze


def install_boundary_selection_invariant() -> None:
    from . import pipeline
    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_boundary_selection_invariant", False):
        return

    def build_with_boundary_invariant(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, draft=enforce_selection_contract(result.draft))

    build_with_boundary_invariant._cutsell_boundary_selection_invariant = True
    pipeline.build_flow_b_draft = build_with_boundary_invariant
