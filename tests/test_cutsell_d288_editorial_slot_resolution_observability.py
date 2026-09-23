"""D-288 -- truthful observability for the EditorialSlotResolution marker.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): `editorial_
slot_resolution_install.py`'s active policy injection (D-042) fires on
every real semantic-equivalence request but never wrote `diagnostics
["editorial_slot_resolution"]`, so `active_path_identity.py`'s own presence
probe always read it as absent -- an observability gap, not a dead/
substituted authority (confirmed by direct code trace: `install_editorial_
slot_resolution()` IS called unconditionally at package import).

This closes the gap with REAL, per-execution evidence (a `ContextVar`, the
same side-channel pattern this codebase already uses elsewhere -- never a
global counter that would mix concurrent jobs together), and is explicit
about what it does and does NOT prove: request-built evidence, never a
claim that the network call executed or that a decision was applied. No
Video00 fact/id anywhere below.
"""
from __future__ import annotations

from cutsell_worker import active_path_identity as api
from cutsell_worker import editorial_slot_resolution_install as install


def setup_function(_):
    # Every test starts with a clean per-execution slate -- proves the
    # ContextVar is genuinely per-run, never carrying stale evidence.
    install.collect_and_clear_editorial_slot_resolution_evidence()


def test_no_request_built_this_run_marker_stays_honestly_absent():
    """If the arbiter was never actually invoked this run, the marker must
    stay absent -- never fabricated as present just because the policy is
    installed at import time."""
    result = {"stage_status": {}, "diagnostics": {}}
    markers = api.component_markers(result)
    row = next(m for m in markers if m["component"] == "EditorialSlotResolution")
    assert row["present"] is False


def test_a_real_request_built_this_run_makes_the_marker_present():
    payload = {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}
    install._inject_semantic_equivalence_policy(payload)  # the real per-call wrapper logic

    result = {"stage_status": {}, "diagnostics": {}}
    markers = api.component_markers(result)
    row = next(m for m in markers if m["component"] == "EditorialSlotResolution")
    assert row["present"] is True


def test_evidence_is_read_and_cleared_never_leaks_into_the_next_run():
    payload = {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}
    install._inject_semantic_equivalence_policy(payload)

    result = {"stage_status": {}, "diagnostics": {}}
    first = api.component_markers(result)
    assert next(m for m in first if m["component"] == "EditorialSlotResolution")["present"] is True

    # A second, independent probe (simulating the next job/run in the same
    # worker process) with NO new request built must read absent again --
    # this is the exact "no contadores globales que mezclen trabajos"
    # requirement: evidence never survives past the run that produced it.
    second = api.component_markers({"stage_status": {}, "diagnostics": {}})
    assert next(m for m in second if m["component"] == "EditorialSlotResolution")["present"] is False


def test_a_real_writer_for_the_key_is_never_overwritten():
    """If some future stage ever writes `diagnostics["editorial_slot_
    resolution"]` for real, this overlay must never clobber it."""
    payload = {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}
    install._inject_semantic_equivalence_policy(payload)

    real_value = {"a_real_future_writer": True}
    result = {"stage_status": {}, "diagnostics": {"editorial_slot_resolution": real_value}}
    diag = api._diagnostics_with_editorial_slot_resolution_evidence(result["diagnostics"])
    assert diag["editorial_slot_resolution"] == real_value


def test_the_overlay_never_mutates_the_callers_diagnostics_dict():
    payload = {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}
    install._inject_semantic_equivalence_policy(payload)

    original = {}
    api._diagnostics_with_editorial_slot_resolution_evidence(original)
    assert original == {}  # untouched -- the overlay returns a NEW dict


def test_evidence_distinguishes_request_built_from_call_executed():
    """Honest, bounded claim: this evidence proves a wire payload was
    constructed with the policy text -- it must never claim the network
    call itself executed, since this module has no way to observe that."""
    payload = {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}
    install._inject_semantic_equivalence_policy(payload)
    evidence = install.collect_and_clear_editorial_slot_resolution_evidence()
    assert len(evidence) == 1
    assert evidence[0]["stage"] == "request_built"
    assert "call_executed" not in evidence[0]
    assert "decision_applied" not in evidence[0]


def test_malformed_payload_records_no_injection_but_still_per_execution():
    """A payload this function cannot inject into (no contents/parts) is
    honestly recorded as `policy_injected: False` -- never silently
    dropped from observability, never claimed as a successful injection."""
    install._inject_semantic_equivalence_policy({})  # no "contents" key at all
    evidence = install.collect_and_clear_editorial_slot_resolution_evidence()
    assert len(evidence) == 1
    assert evidence[0]["policy_injected"] is False
