"""Active-path identity: proof of WHICH code produced a result (D-095 addendum).

"CODE EXISTS != VIDEO USED IT". Every benchmark result now carries, computed
INSIDE the worker that produced it:

- a content fingerprint of the mounted ``cutsell_worker`` package (sha256
  over every ``*.py`` file, sorted by relative path) so a workflow can
  compare it against the checkout it believes it dispatched;
- the build commit when the environment provides it
  (``CUTSELL_BUILD_GIT_SHA``), the Python version and the resolver mode;
- per-component ACTIVITY MARKERS derived from the result's own
  ``stage_status`` / ``diagnostics``: for each major canonical component,
  whether the evidence of it having run is present in this very result.

Observability only. Nothing here influences Selection, Boundary, Freeze,
rendering or QC; the identity block is additive and read only by QA
tooling and humans. Video00-specific values are never referenced.
"""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "cutsell.active_path_identity.v1"

# (component label, D-reference, how the result proves it ran).
# Each probe is a (section, key path) pair; a marker is "present" when the
# key exists and is non-empty / non-null in that section of the result.
_COMPONENT_PROBES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("AttemptReconstructor", "D-021", "stage_status", ("attempt_reconstruction",)),
    ("TakeSegmentation", "D-021", "stage_status", ("take_segmentation",)),
    ("HybridSemanticPass", "D-081", "stage_status", ("hybrid_editorial",)),
    ("IdeaClusterer.semantic_tier", "D-020/D-044", "diagnostics", ("semantic_idea_equivalence",)),
    ("IdeaClusterer.grouping_safety", "D-058/D-083/D-085/D-094.F3", "diagnostics", ("distinct_idea_grouping_safety",)),
    ("BestTakeResolver.family_window_labels", "D-094.3.F8", "diagnostics", ("take_judge_groups", "semantic_label_source")),
    ("BestTakeResolver.judge_groups", "D-021", "diagnostics", ("take_judge_groups",)),
    ("DeterministicBestTakeAuthority", "D-021", "diagnostics", ("deterministic_best_take_authority",)),
    ("RealizationResolver.authoritative", "D-050C2/D-087", "diagnostics", ("realization_resolver_authority",)),
    ("RealizationResolver.story_placement", "D-089/D-094.3.F9", "diagnostics", ("authoritative_story_placement",)),
    ("PostAuthorityValidation", "D-090/D-092", "diagnostics", ("post_authority_validation",)),
    ("StoryValidator", "D-020", "diagnostics", ("final_story_coherence_validation",)),
    ("CanonicalEditPlan", "D-023/D-087", "diagnostics", ("canonical_edit_plan",)),
    ("FinalEditReviewer", "D-023", "diagnostics", ("final_edit_reviewer",)),
    ("RepairLoop", "D-026", "diagnostics", ("repair_loop",)),
    ("SelectionFreeze", "D-023", "diagnostics", ("selection_boundary_contract",)),
    ("HybridBudgetLedger", "D-094.F2", "diagnostics", ("hybrid_editorial_budget_exhausted_chunk_count",)),
    ("EditorialSlotResolution", "D-042", "diagnostics", ("editorial_slot_resolution",)),
)


def package_fingerprint(package_dir: str | Path | None = None) -> dict[str, Any]:
    """sha256 over every ``*.py`` file of the package (sorted relative paths +
    contents). Deterministic across machines for the same source tree."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    byte_count = 0
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        byte_count += len(data)
        digest.update(rel)
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return {
        "package_root": root.name,
        "sha256": digest.hexdigest(),
        "module_count": len(files),
        "byte_count": byte_count,
    }


def _present(section: Mapping[str, Any] | None, path: tuple[str, ...]) -> bool:
    if not isinstance(section, Mapping):
        return False
    node: Any = section
    for index, key in enumerate(path):
        if isinstance(node, list):
            # any row carrying the key counts (e.g. take_judge_groups[*].semantic_label_source)
            return any(isinstance(row, Mapping) and _present(row, path[index:]) for row in node)
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    if node is None:
        return False
    if isinstance(node, (list, tuple, dict, str)) and len(node) == 0:
        return False
    return True


def component_markers(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = {
        "stage_status": result.get("stage_status") if isinstance(result, Mapping) else None,
        "diagnostics": result.get("diagnostics") if isinstance(result, Mapping) else None,
    }
    rows = []
    for label, reference, section_name, path in _COMPONENT_PROBES:
        rows.append({
            "component": label,
            "reference": reference,
            "evidence": f"{section_name}.{'.'.join(path)}",
            "present": _present(sections.get(section_name), path),
        })
    return rows


def build_active_path_identity(result: Mapping[str, Any], *, env: Mapping[str, str] | None = None,
                               package_dir: str | Path | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    fingerprint = package_fingerprint(package_dir)
    markers = component_markers(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "build_git_sha": str(env.get("CUTSELL_BUILD_GIT_SHA") or "") or None,
        "package": fingerprint,
        "python_version": platform.python_version(),
        "resolver_mode_env": str(env.get("CUTSELL_UNIFIED_REALIZATION_RESOLVER") or "") or None,
        "clean_cut_core_v1_env": str(env.get("CUTSELL_CLEAN_CUT_CORE_V1") or "") or None,
        "hybrid_max_edit_usd_env": str(env.get("CUTSELL_HYBRID_MAX_EDIT_USD") or "") or None,
        "component_markers": markers,
        "components_present": sum(1 for m in markers if m["present"]),
        "components_absent": [m["component"] for m in markers if not m["present"]],
    }
