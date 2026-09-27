"""Experimental whole-video editorial authority for CutSell.

V2 deliberately has one semantic decision seam.  Perception and candidate
construction happen upstream; the whole-video reasoner then settles the full
candidate universe.  Once that plan is accepted, only physical Boundary work
may run.  Provider failure is fail-closed because silently falling back to the
legacy rule chain would make an A/B run impossible to interpret.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import re
import unicodedata
from typing import Callable

from .contracts import ProcessingResult
from .selection_boundary_contract import (
    enforce_selection_contract,
    freeze_selection_contract,
)
from .unified_selection_reasoner import (
    UnifiedSelectionReasoner,
    apply_unified_selection_reasoner,
)


SCHEMA_VERSION = "cutsell.editorial_engine_v2.v1"
ENV_FLAG = "CUTSELL_EDITORIAL_ENGINE_V2"

PHASES = (
    "perceive_complete_source",
    "understand_story_and_attempts",
    "reconstruct_attempts",
    "clean_cut_and_best_take",
    "compose_story",
    "resolve_keep_discard",
    "freeze_selection",
    "execute_boundaries",
    "post_render_review",
)


def _true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def enabled(explicit: bool | None = None) -> bool:
    return _true(os.environ.get(ENV_FLAG, "0")) if explicit is None else bool(explicit)


def _candidate_ids(draft) -> tuple[str, ...]:
    return tuple(sorted({
        str(clip.clip_id)
        for clip in (*tuple(draft.selected), *tuple(draft.alternates), *tuple(draft.discarded))
    }))


def _discard_signature(draft) -> str:
    values = sorted(str(clip.clip_id) for clip in tuple(draft.discarded))
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)


def _ordered_semantic_signature(draft) -> str:
    """Seal V2 story order as well as content; legacy Freeze sorts by source."""
    tokens = []
    for clip in tuple(draft.selected):
        for token in _TOKEN_RE.findall(str(clip.text or "")):
            raw = unicodedata.normalize("NFKD", token.casefold())
            tokens.append("".join(ch for ch in raw if not unicodedata.combining(ch)))
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def _require_whole_video_evidence(draft) -> dict:
    diagnostics = dict(getattr(draft, "diagnostics", None) or {})
    whole = diagnostics.get("whole_video_context") or {}
    status = (whole.get("status") or {}).get("status")
    av_status = whole.get("audiovisual_input_status")
    if status not in {"ok", "available", "complete", "success", "applied"}:
        raise RuntimeError("Editorial Engine V2 requires available whole-video context")
    if av_status != "received_and_parsed":
        raise RuntimeError("Editorial Engine V2 requires verified audiovisual Watch + Listen input")
    sources = tuple(whole.get("sources") or ())
    if not sources or any(not str(source.get("audiovisual_evidence") or "") for source in sources):
        raise RuntimeError("Editorial Engine V2 requires audiovisual evidence for every source")
    return whole


def _fold_alternates(draft):
    """V2 has a final KEEP/DISCARD decision, never a hidden third bucket."""
    discarded_by_id = {str(clip.clip_id): replace(clip, selected=False) for clip in draft.discarded}
    for clip in draft.alternates:
        discarded_by_id.setdefault(str(clip.clip_id), replace(clip, selected=False))
    discarded = tuple(sorted(
        discarded_by_id.values(),
        key=lambda clip: (clip.source_order, float(clip.start), float(clip.end), clip.clip_id),
    ))
    return replace(draft, alternates=(), discarded=discarded)


def run_editorial_engine_v2(
    result: ProcessingResult,
    *,
    selection_reasoner: UnifiedSelectionReasoner | None,
    recover_complete_boundaries: Callable[[ProcessingResult], ProcessingResult],
    execute_boundaries: Callable[[ProcessingResult], ProcessingResult],
) -> ProcessingResult:
    """Resolve once, freeze once, and reject every post-freeze semantic mutation."""
    if selection_reasoner is None:
        raise RuntimeError("Editorial Engine V2 requires a whole-video selection reasoner")

    whole = _require_whole_video_evidence(result.draft)
    candidate_ids = _candidate_ids(result.draft)
    if not candidate_ids:
        raise RuntimeError("Editorial Engine V2 received no editorial candidates")

    request_diagnostics = dict(result.draft.diagnostics or {})
    request_diagnostics["editorial_engine_v2_request"] = {
        "schema_version": SCHEMA_VERSION,
        "require_audiovisual_evidence": True,
        "allow_global_story_reordering": True,
    }
    request_draft = replace(result.draft, diagnostics=request_diagnostics)
    resolved = apply_unified_selection_reasoner(request_draft, selection_reasoner)
    reasoner_diag = dict((resolved.diagnostics or {}).get("unified_selection_reasoner") or {})
    if reasoner_diag.get("status") != "applied":
        detail = reasoner_diag.get("error") or reasoner_diag.get("status") or "unknown"
        raise RuntimeError(f"Editorial Engine V2 whole-video plan failed: {detail}")
    if _candidate_ids(resolved) != candidate_ids:
        raise RuntimeError("Editorial Engine V2 plan changed the candidate universe")

    resolved = _fold_alternates(resolved)
    diagnostics = dict(resolved.diagnostics or {})
    diagnostics["editorial_engine_v2"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "selection_resolved",
        "authority": "whole_video_reasoner_single_semantic_seam",
        "whole_video_source_count": len(tuple(whole.get("sources") or ())),
        "candidate_count": len(candidate_ids),
        "selected_count": len(tuple(resolved.selected)),
        "discarded_count": len(tuple(resolved.discarded)),
        "phases": [
            {"phase": phase, "status": "complete" if phase != "post_render_review" else "pending_render"}
            for phase in PHASES
        ],
        "post_freeze_semantic_mutation_allowed": False,
        "repair_policy": "localized_family_reopen_only_after_blocking_post_render_review",
    }
    result = replace(result, draft=replace(resolved, diagnostics=diagnostics))

    # Complete source-proven word edges before the semantic phase barrier.
    result = recover_complete_boundaries(result)
    result = replace(result, draft=_fold_alternates(result.draft))
    discard_signature = _discard_signature(result.draft)
    ordered_semantic_signature = _ordered_semantic_signature(result.draft)

    result = replace(result, draft=freeze_selection_contract(result.draft))
    result = execute_boundaries(result)
    result = replace(result, draft=enforce_selection_contract(result.draft))

    if tuple(result.draft.alternates):
        raise RuntimeError("Editorial Engine V2 post-freeze stage recreated alternates")
    if _discard_signature(result.draft) != discard_signature:
        raise RuntimeError("Editorial Engine V2 post-freeze stage changed DISCARD membership")
    if _ordered_semantic_signature(result.draft) != ordered_semantic_signature:
        raise RuntimeError("Editorial Engine V2 post-freeze stage changed story order")

    diagnostics = dict(result.draft.diagnostics or {})
    diagnostics["editorial_engine_v2"] = {
        **diagnostics["editorial_engine_v2"],
        "status": "frozen_boundary_verified_pending_post_render_review",
        "discard_membership_sha256": discard_signature,
        "ordered_semantic_sha256": ordered_semantic_signature,
        "selection_contract_status": (
            diagnostics.get("selection_boundary_contract") or {}
        ).get("status"),
    }
    return replace(
        result,
        draft=replace(result.draft, diagnostics=diagnostics),
        stage_status={
            **result.stage_status,
            "brain_mode": "editorial_engine_v2_whole_video",
            "selection_phase_authority": "v2_whole_video_reasoner_frozen",
            "unified_selection_reasoner": "required_applied",
            "selection_boundary_contract": "verified",
            "editorial_engine_v2": "pending_post_render_review",
        },
    )
