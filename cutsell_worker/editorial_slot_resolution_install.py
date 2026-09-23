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

import time
from contextvars import ContextVar
from typing import Any, Mapping

# D-288 (audit finding, `docs/CUTSELL_DECISIONS.md`): this module installs
# an ACTIVE monkeypatch at package import (`install_editorial_slot_
# resolution`, called once from `cutsell_worker/__init__.py`), but never
# wrote `diagnostics["editorial_slot_resolution"]` -- the exact key
# `active_path_identity.py`'s own `EditorialSlotResolution` presence probe
# reads. The probe therefore always read "absent", even though the policy
# genuinely fires on every real semantic-equivalence request. Four states
# a caller must never conflate:
#   1. POLICY INSTALLED -- this module's own install function ran at
#      import. Always true once imported; proves nothing about any ONE
#      run, so it is never reported as per-execution evidence.
#   2. REQUEST BUILT WITH THE POLICY PRESENT -- a real wire payload was
#      constructed with the policy text actually verified present in it.
#      THIS is what counts as real evidence below. A FAILED injection
#      attempt (malformed payload, no contents/parts to splice into) is
#      recorded too, for honest observability, but with `policy_injected:
#      False` -- it must never itself make the presence marker read
#      "present"; a failed attempt proves the opposite of an active
#      policy for that one call.
#   3. CALL EXECUTED -- the arbiter's `.check()` actually sent that
#      payload over the network. NOT observed here: this module only
#      wraps request CONSTRUCTION, never the network call, so it never
#      claims more than build-time evidence proves.
#   4. DECISION APPLIED -- the arbiter's returned verdict was actually
#      used to merge/reject a candidate pair. Recorded independently, by
#      `take_grouping_provider.py`'s own `distinct_idea_grouping_safety`
#      diagnostics (`arbiter_confirmed_pairs`/`edge_trace`/
#      `arbiter_rejected_pairs`) -- never fabricated or duplicated here.
#
# JOB-SCOPED, NOT READ-DESTRUCTIVE (hardened per the D-288 finding-5
# correction): a `ContextVar` alone does not by itself PROVE isolation
# between concurrent jobs sharing one worker process/thread -- CPython
# gives each OS THREAD its own default Context, which this module's own
# test suite now verifies empirically with real concurrent threads (see
# `tests/test_cutsell_d288_editorial_slot_resolution_observability.py`),
# but two SEQUENTIAL jobs in the SAME thread (the common case for an RQ/
# RunPod worker reusing a warm process) would otherwise see one another's
# evidence unless something explicitly draws a job boundary. That
# boundary is `reset_editorial_slot_resolution_evidence()` below --
# called ONCE at the START of each real per-job entry point (`universal_
# clean_cut_validation.run_single_universal_clean_cut_validation`,
# `export_job.run_export_job`) -- never implicitly, never tied to a read.
# Reading evidence (`read_editorial_slot_resolution_evidence` below) is
# deliberately NON-DESTRUCTIVE and idempotent: calling it any number of
# times during or after the same job returns the SAME answer, so a QA
# harness or a retried probe never sees "present" once and "absent" the
# next time for the exact same run. This replaces this module's earlier
# read-and-clear design (`collect_and_clear_...`, now removed), which
# would have silently returned "absent" on a second call -- a probe
# function must be safely repeatable.
_POLICY_INJECTION_EVIDENCE: "ContextVar[tuple[dict[str, Any], ...]]" = ContextVar(
    "_EDITORIAL_SLOT_RESOLUTION_POLICY_INJECTION_EVIDENCE", default=(),
)


def reset_editorial_slot_resolution_evidence() -> None:
    """D-288: the real per-job init/cleanup boundary. Call ONCE at the
    START of a per-job entry point, before any arbiter call could happen
    for that job, so a previous job's evidence (in the same warm worker
    thread) can never leak into this job's result. Idempotent to call
    more than once (always just clears to empty)."""
    _POLICY_INJECTION_EVIDENCE.set(())


def read_editorial_slot_resolution_evidence() -> tuple[dict[str, Any], ...]:
    """Non-destructive, idempotent read of THIS job's own evidence so far.
    Never clears -- only `reset_editorial_slot_resolution_evidence` does
    that, at the next job's own start."""
    return _POLICY_INJECTION_EVIDENCE.get()


def _record_policy_injection_evidence(*, policy_injected: bool, text_length: int) -> None:
    _POLICY_INJECTION_EVIDENCE.set(_POLICY_INJECTION_EVIDENCE.get() + ({
        "stage": "request_built",
        "policy_injected": policy_injected,
        "text_length": text_length,
        "recorded_at": time.time(),
    },))


_SLOT_RULES = (
    "Optimize for the MINIMUM SUFFICIENT EDITORIAL SET, not the union of every fact said across all usable takes.",
    "Infer audience-facing EDITORIAL FUNCTION before literal wording differences (for example hook, setup, diagnosis, symptom, example, reflection, conclusion, CTA).",
    "Two complete deliveries that perform the same editorial function and communicate the same core intended message are competing realizations of one retry family even when wording differs or one contains extra supporting detail.",
    "GOOD + GOOD does not imply keeping both. A complete realization may make another complete realization redundant.",
    "Distinguish REQUIRED PROPOSITIONS from SUPPORTING/RESTATED/ELABORATIVE DETAIL. A genuinely new required story proposition may be a separate idea; unique supporting wording does not by itself create one.",
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
        _record_policy_injection_evidence(policy_injected=False, text_length=0)
        return out
    first = contents[0]
    parts = [dict(item) for item in (first.get("parts") or ())]
    if not parts:
        _record_policy_injection_evidence(policy_injected=False, text_length=0)
        return out
    text = str(parts[0].get("text") or "")
    injected = _SEMANTIC_EQUIVALENCE_POLICY not in text
    if injected:
        parts[0]["text"] = _SEMANTIC_EQUIVALENCE_POLICY + text
    first["parts"] = parts
    contents[0] = first
    out["contents"] = contents
    # D-288: real, per-call evidence that THIS execution actually built a
    # request with the policy text present -- never a global counter, see
    # this module's own D-288 comment block above.
    _record_policy_injection_evidence(policy_injected=injected, text_length=len(str(parts[0].get("text") or "")))
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
    # D-097.9 (R12): `_install_semantic_pair_budget_coverage` is RETIRED from
    # the active path. It wrapped `take_grouping_provider._rank_candidate_pairs`
    # at import with a coverage-first re-order ("one pair per group before any
    # group gets a second"), a SECOND authority over the arbiter budget order
    # that silently defeated the D-097.8 R9 ranking: on RAW 34045158712 (68
    # candidate pairs, 14 asked) it promoted zero-evidence neighbours (hair
    # loss <-> stomach aside, score 0.02; gastritis <-> vaccine, 0.03) into the
    # budget because they "exposed new groups" and demoted a 0.95 same-opening
    # retry pair to 12th. One ranking authority now: content overlap +
    # restart/continuation evidence lead, proximity breaks ties (R9). The
    # function is kept, deactivated, for the record.
    _install_legacy_unified_selection_policy()
