"""Install minimum-sufficient-editorial-set semantics at retry-family formation.

Clean Cut Core V1 does not use the legacy whole-video Unified Selection reasoner as its
primary editor. Its active path forms retry families through the bounded semantic idea-
equivalence arbiter, then lets Best Take choose one winner inside each proven family.

The Human-Gold failure this policy addresses happens at that family-formation boundary:
two good complete deliveries can be left as separate ideas merely because one contains
extra supporting wording. Once separated, Best Take cannot make them compete and both
may survive. The active patch therefore teaches the semantic-equivalence request that
"same intended idea" is an editorial-function question, not a union-of-facts test.

The same active boundary also has a fixed per-request pair budget. Pure score ordering
can spend most of that budget comparing many variants inside one dense local region and
starve a later retry family entirely. This installer therefore keeps the existing pair
priority score but diversifies its ORDER: pairs that expose previously unseen groups are
asked before redundant extra comparisons among groups already represented. No extra
provider calls or tokens are introduced; the existing max-pairs budget remains intact.

The legacy Unified Selection contract is patched too for rollback parity, but it is not
relied upon for the active Clean Cut Core V1 behavior.
"""
from __future__ import annotations

from typing import Any, Mapping


_SLOT_RULES = (
    "Optimize for the MINIMUM SUFFICIENT EDITORIAL SET, not the union of every fact said across all usable takes.",
    "Infer audience-facing EDITORIAL FUNCTION before literal wording differences (for example hook, setup, diagnosis, symptom, example, reflection, conclusion, CTA).",
    "Two complete deliveries that perform the same editorial function and communicate the same core intended message are competing realizations of one retry family even when wording differs or one contains extra supporting detail.",
    "GOOD + GOOD does not imply keeping both. A complete realization may make another complete realization redundant.",
    "Distinguish REQUIRED PROPOSITIONS from SUPPORTING, RESTATED, or ELABORATIVE DETAIL. A genuinely new required story proposition may be a separate idea; unique supporting wording does not by itself create one.",
    "Use a composite only when no single realization is sufficient and complementary clean pieces are genuinely required to create one coherent complete realization.",
    "A later conclusion/restatement after an already complete conclusion is normally a competing realization of the CONCLUSION slot unless it advances the story with a genuinely different required proposition.",
    "Do not collapse genuinely complementary micro-deliveries: incomplete pieces that advance different necessary parts of one message may remain continuation/composite material.",
)

_SEMANTIC_EQUIVALENCE_POLICY = (
    "EDITORIAL-FUNCTION RULE: decide SAME intended idea/message the way a human editor "
    "would decide whether two deliveries should COMPETE for one rhetorical slot. Two "
    "deliveries can be the SAME idea even if wording, length, examples, or supporting "
    "facts differ. Extra detail that merely restates, supports, specifies, or elaborates "
    "the same core audience-facing point does NOT automatically make a new idea. Treat "
    "them as DIFFERENT only when the second delivery advances a genuinely distinct "
    "required story proposition or audience-facing job that should survive alongside "
    "the first. A second complete conclusion/restatement of the same takeaway is SAME; "
    "a complementary next story beat is DIFFERENT. Do not rank or choose a winner here. "
)


def _inject_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rollback-path parity for the legacy whole-video reasoner."""
    out = dict(payload)
    existing = [str(item) for item in (out.get("editorial_contract") or ())]
    marker = _SLOT_RULES[0]
    if marker not in existing:
        insertion = min(2, len(existing))
        existing[insertion:insertion] = list(_SLOT_RULES)
    out["editorial_contract"] = existing
    return out


def _inject_semantic_equivalence_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Inject the editorial-slot definition into the ACTIVE bounded arbiter request."""
    out = dict(payload)
    contents = [dict(item) for item in (out.get("contents") or ())]
    if not contents:
        return out
    first = contents[0]
    parts = [dict(item) for item in (first.get("parts") or ())]
    if not parts:
        return out
    text = str(parts[0].get("text") or "")
    if _SEMANTIC_EQUIVALENCE_POLICY not in text:
        parts[0]["text"] = _SEMANTIC_EQUIVALENCE_POLICY + text
    first["parts"] = parts
    contents[0] = first
    out["contents"] = contents
    return out


def _coverage_first_pair_order(ranked_pairs):
    """Diversify a score-ranked pair stream without changing eligibility or cost.

    Each pair is ``(left_group_index, right_group_index, left_clip_id, right_clip_id)``.
    The original rank already encodes proximity/overlap/restart evidence. We preserve
    that order as the tie-break, but first prefer the candidate that exposes the largest
    number of not-yet-represented group endpoints (2, then 1, then 0). This prevents a
    dense retry neighborhood from monopolizing a bounded request while still spending
    all remaining slots on the strongest original-score comparisons once coverage is
    exhausted.
    """
    remaining = list(ranked_pairs)
    ordered = []
    covered_groups: set[int] = set()
    while remaining:
        best_index = 0
        best_gain = -1
        for index, pair in enumerate(remaining):
            left_group_index, right_group_index = int(pair[0]), int(pair[1])
            gain = int(left_group_index not in covered_groups) + int(
                right_group_index not in covered_groups
            )
            if gain > best_gain:
                best_index = index
                best_gain = gain
                if gain == 2:
                    break
        pair = remaining.pop(best_index)
        ordered.append(pair)
        covered_groups.add(int(pair[0]))
        covered_groups.add(int(pair[1]))
    return tuple(ordered)


def _install_active_semantic_equivalence_policy() -> None:
    from . import semantic_idea_equivalence_google as module

    original = module.build_semantic_equivalence_request
    if getattr(original, "_cutsell_editorial_slot_resolution", False):
        return

    def build_request_with_editorial_slots(*args, **kwargs):
        return _inject_semantic_equivalence_policy(original(*args, **kwargs))

    build_request_with_editorial_slots._cutsell_editorial_slot_resolution = True
    module.build_semantic_equivalence_request = build_request_with_editorial_slots


def _install_semantic_pair_budget_coverage() -> None:
    from . import take_grouping_provider as module

    original = module._rank_candidate_pairs
    if getattr(original, "_cutsell_editorial_slot_coverage", False):
        return

    def rank_candidate_pairs_with_coverage(*args, **kwargs):
        return _coverage_first_pair_order(original(*args, **kwargs))

    rank_candidate_pairs_with_coverage._cutsell_editorial_slot_coverage = True
    module._rank_candidate_pairs = rank_candidate_pairs_with_coverage


def _install_legacy_unified_selection_policy() -> None:
    from . import unified_selection_google as module

    original = module.build_unified_selection_payload
    if getattr(original, "_cutsell_editorial_slot_resolution", False):
        return

    def build_payload_with_editorial_slots(*args, **kwargs):
        return _inject_contract(original(*args, **kwargs))

    build_payload_with_editorial_slots._cutsell_editorial_slot_resolution = True
    module.build_unified_selection_payload = build_payload_with_editorial_slots


def install_editorial_slot_resolution() -> None:
    _install_active_semantic_equivalence_policy()
    _install_semantic_pair_budget_coverage()
    _install_legacy_unified_selection_policy()
