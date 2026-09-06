"""Install the minimum-sufficient-editorial-set contract into Unified Selection.

This is intentionally a pre-Selection semantic-policy patch, not a post-hoc lexical
removal rule. The existing unified-selection safety layer already enforces at most one
SELECT inside a retry family. The failure fixed here occurs one step earlier: two good
complete deliveries can be misclassified as independent/continuation merely because
one contains extra wording or supporting detail. Once misclassified, the existing
single-family-winner authority cannot act.

The patch therefore strengthens the provider-neutral editorial contract presented to
the whole-video reasoner so that rhetorical/editorial function is resolved before
claim preservation, composite rescue, or co-keep decisions.
"""
from __future__ import annotations

from typing import Any, Mapping


_SLOT_RULES = (
    "Optimize for the MINIMUM SUFFICIENT EDITORIAL SET, not the union of every fact said across all usable takes.",
    "Before using wording overlap or unique facts, infer each candidate's audience-facing EDITORIAL FUNCTION (for example hook, setup, diagnosis, symptom, example, reflection, conclusion, CTA). Function is higher-level than literal wording.",
    "Two complete deliveries that perform the same editorial function are competing realizations of one slot even when wording differs and one contains extra supporting detail. Classify them as retry_winner/retry_alternate in the SAME family, not independent or continuation merely to preserve extra wording.",
    "GOOD + GOOD does not imply SELECT + SELECT. If two complete realizations do the same editorial job, choose the single realization that sufficiently communicates the required intent; the other is SWAP or DISCARD according to its usefulness.",
    "Distinguish REQUIRED PROPOSITIONS from SUPPORTING/RESTATED/ELABORATIVE DETAIL. Supporting detail is allowed to be sacrificed when a complete selected realization already communicates the required story intent cleanly.",
    "A fact is not automatically required merely because it appears only in one retry. Ask whether removing it changes the story's necessary audience-facing meaning; if not, treat it as supporting detail, not a reason to co-keep a second complete realization.",
    "Use a COMPOSITE only when no single realization is sufficient and complementary clean pieces are genuinely required to create one coherent complete realization. Never create or preserve a composite just to maximize semantic coverage.",
    "For competing complete realizations, rank in this order: required-idea coverage; contradiction/factual safety; completeness; redundancy with already-selected realization; delivery quality; narrative fit; rhythm/brevity.",
    "A later conclusion/restatement after an already complete conclusion is normally a competing retry of the CONCLUSION slot, not a necessary continuation, unless it introduces a genuinely required new story proposition that the first conclusion cannot communicate without it.",
    "Likewise, do not collapse genuinely complementary micro-deliveries: when A and B are incomplete pieces that together form one superior realization, retain them as composite_piece/continuation rather than forcing a whole-take winner.",
)


def _inject_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    existing = [str(item) for item in (out.get("editorial_contract") or ())]
    marker = _SLOT_RULES[0]
    if marker not in existing:
        # Insert immediately after the global understand-the-video directives so these
        # rules govern family formation before the legacy preservation clauses below.
        insertion = min(2, len(existing))
        existing[insertion:insertion] = list(_SLOT_RULES)
    out["editorial_contract"] = existing
    return out


def install_editorial_slot_resolution() -> None:
    from . import unified_selection_google as module

    original_payload = module.build_unified_selection_payload
    if getattr(original_payload, "_cutsell_editorial_slot_resolution", False):
        return

    def build_payload_with_editorial_slots(*args, **kwargs):
        return _inject_contract(original_payload(*args, **kwargs))

    build_payload_with_editorial_slots._cutsell_editorial_slot_resolution = True
    module.build_unified_selection_payload = build_payload_with_editorial_slots
