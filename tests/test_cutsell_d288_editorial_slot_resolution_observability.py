"""D-288 -- truthful, job-scoped observability for the EditorialSlotResolution marker.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): `editorial_
slot_resolution_install.py`'s active policy injection (D-042) fires on
every real semantic-equivalence request but never wrote `diagnostics
["editorial_slot_resolution"]`, so `active_path_identity.py`'s own presence
probe always read it as absent -- an observability gap, not a dead/
substituted authority.

Correction (this file, second pass): the first D-288 pass used a plain
read-and-clear `ContextVar`, which (a) is not itself proof of isolation
between two CONCURRENT jobs sharing one worker process/thread, (b) counted
a FAILED injection attempt as if it proved an active policy, and (c) made
the probe destructive -- a second read of the same job's evidence
incorrectly read "absent". This file proves all three are now fixed: a
real per-job `reset_editorial_slot_resolution_evidence()` boundary, only
`policy_injected: True` rows count, and `read_editorial_slot_resolution_
evidence`/`component_markers`/`build_active_path_identity` are all
non-destructive and idempotent. No Video00 fact/id anywhere below.
"""
from __future__ import annotations

import threading
import time

from cutsell_worker import active_path_identity as api
from cutsell_worker import editorial_slot_resolution_install as install


def setup_function(_):
    # Every test starts with a clean per-job slate via the REAL per-job
    # boundary function, not an implicit side effect of a prior read.
    install.reset_editorial_slot_resolution_evidence()


def _build_payload():
    return {"contents": [{"role": "user", "parts": [{"text": "some prompt"}]}]}


# =============================================================================
# Basic presence / absence
# =============================================================================

def test_no_request_built_this_run_marker_stays_honestly_absent():
    result = {"stage_status": {}, "diagnostics": {}}
    markers = api.component_markers(result)
    row = next(m for m in markers if m["component"] == "EditorialSlotResolution")
    assert row["present"] is False


def test_a_real_request_built_this_run_makes_the_marker_present():
    install._inject_semantic_equivalence_policy(_build_payload())
    result = {"stage_status": {}, "diagnostics": {}}
    markers = api.component_markers(result)
    row = next(m for m in markers if m["component"] == "EditorialSlotResolution")
    assert row["present"] is True


# =============================================================================
# Finding 5.a: a FAILED injection attempt must never count as "active"
# =============================================================================

def test_a_failed_injection_attempt_alone_never_makes_the_marker_present():
    """A malformed payload (no contents/parts) is recorded honestly with
    `policy_injected: False` -- it must NOT make the presence marker read
    "present". A failed attempt proves the opposite of an active policy
    for that call."""
    install._inject_semantic_equivalence_policy({})  # no "contents" key at all
    result = {"stage_status": {}, "diagnostics": {}}
    row = next(m for m in api.component_markers(result) if m["component"] == "EditorialSlotResolution")
    assert row["present"] is False


def test_one_failed_and_one_real_injection_still_reads_present():
    """A failed attempt does not poison a genuine one either -- only the
    real row is what makes it present, and it is counted correctly."""
    install._inject_semantic_equivalence_policy({})  # failed
    install._inject_semantic_equivalence_policy(_build_payload())  # real
    result = {"stage_status": {}, "diagnostics": {}}
    row = next(m for m in api.component_markers(result) if m["component"] == "EditorialSlotResolution")
    assert row["present"] is True
    real_evidence = api._real_editorial_slot_resolution_evidence()
    assert len(real_evidence) == 1
    assert real_evidence[0]["policy_injected"] is True


# =============================================================================
# Finding 5.b: idempotent, non-destructive reads
# =============================================================================

def test_reading_the_marker_twice_gives_the_same_answer():
    """A probe function must be safely repeatable -- calling component_
    markers (or build_active_path_identity) twice for the SAME job must
    never read "present" once and "absent" the next time."""
    install._inject_semantic_equivalence_policy(_build_payload())
    result = {"stage_status": {}, "diagnostics": {}}

    first = next(m for m in api.component_markers(result) if m["component"] == "EditorialSlotResolution")
    second = next(m for m in api.component_markers(result) if m["component"] == "EditorialSlotResolution")
    third = next(m for m in api.component_markers(result) if m["component"] == "EditorialSlotResolution")
    assert first["present"] is True
    assert second["present"] is True
    assert third["present"] is True


def test_build_active_path_identity_is_idempotent_across_calls():
    install._inject_semantic_equivalence_policy(_build_payload())
    result = {"stage_status": {}, "diagnostics": {}}
    identity_a = api.build_active_path_identity(result, env={})
    identity_b = api.build_active_path_identity(result, env={})
    assert identity_a["components_present"] == identity_b["components_present"]
    assert len(identity_a["editorial_slot_resolution_evidence"]) == len(identity_b["editorial_slot_resolution_evidence"]) == 1


# =============================================================================
# Finding 5.c: real per-job reset boundary (not implicit clearing on read)
# =============================================================================

def test_evidence_survives_multiple_reads_but_reset_clears_it():
    install._inject_semantic_equivalence_policy(_build_payload())
    assert api._real_editorial_slot_resolution_evidence()  # non-empty, still there
    assert api._real_editorial_slot_resolution_evidence()  # still there on a 2nd read

    install.reset_editorial_slot_resolution_evidence()  # the real job boundary
    assert api._real_editorial_slot_resolution_evidence() == ()


def test_a_second_sequential_job_in_the_same_thread_does_not_see_the_first_jobs_evidence():
    """The realistic case for a warm RQ/RunPod worker: job 1 and job 2 run
    SEQUENTIALLY in the SAME thread. Without an explicit reset at job
    start, job 2 would incorrectly inherit job 1's evidence -- this is
    exactly why `reset_editorial_slot_resolution_evidence()` must be
    called at the start of every real per-job entry point, not left to
    ContextVar's own thread-default behavior."""
    # Job 1
    install.reset_editorial_slot_resolution_evidence()
    install._inject_semantic_equivalence_policy(_build_payload())
    assert api._real_editorial_slot_resolution_evidence()

    # Job 2 starts -- the real per-job entry points call this first.
    install.reset_editorial_slot_resolution_evidence()
    assert api._real_editorial_slot_resolution_evidence() == ()


# =============================================================================
# Finding 5.b (continued): "ContextVar alone does not prove isolation" --
# prove it empirically with real concurrent OS threads, not just assert it.
# =============================================================================

def test_two_concurrent_jobs_in_different_threads_never_mix_evidence():
    results: dict[str, tuple] = {}
    barrier = threading.Barrier(2)

    def _job(name: str, inject: bool):
        install.reset_editorial_slot_resolution_evidence()
        barrier.wait(timeout=5)  # maximize the chance of genuine interleaving
        if inject:
            install._inject_semantic_equivalence_policy(_build_payload())
        time.sleep(0.05)  # let the other thread run its own body too
        results[name] = api._real_editorial_slot_resolution_evidence()

    t1 = threading.Thread(target=_job, args=("job_with_injection", True))
    t2 = threading.Thread(target=_job, args=("job_without_injection", False))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(results["job_with_injection"]) == 1
    assert results["job_with_injection"][0]["policy_injected"] is True
    # The thread that never injected anything must see ZERO evidence --
    # proof the other thread's real request-built row never leaked across.
    assert results["job_without_injection"] == ()


# =============================================================================
# Other existing guarantees, re-verified against the corrected API
# =============================================================================

def test_a_real_writer_for_the_key_is_never_overwritten():
    install._inject_semantic_equivalence_policy(_build_payload())
    real_value = {"a_real_future_writer": True}
    result = {"stage_status": {}, "diagnostics": {"editorial_slot_resolution": real_value}}
    diag = api._diagnostics_with_editorial_slot_resolution_evidence(
        result["diagnostics"], api._real_editorial_slot_resolution_evidence(),
    )
    assert diag["editorial_slot_resolution"] == real_value


def test_the_overlay_never_mutates_the_callers_diagnostics_dict():
    install._inject_semantic_equivalence_policy(_build_payload())
    original = {}
    api._diagnostics_with_editorial_slot_resolution_evidence(original, api._real_editorial_slot_resolution_evidence())
    assert original == {}


def test_evidence_distinguishes_request_built_from_call_executed():
    install._inject_semantic_equivalence_policy(_build_payload())
    evidence = install.read_editorial_slot_resolution_evidence()
    assert len(evidence) == 1
    assert evidence[0]["stage"] == "request_built"
    assert "call_executed" not in evidence[0]
    assert "decision_applied" not in evidence[0]


def test_evidence_is_persisted_into_build_active_path_identitys_own_result():
    """Finding 5: 'evidencia persistida en su resultado' -- the raw
    evidence rows must be present in the ACTUAL returned/serialized
    result, not only visible as a side effect of internal probing."""
    install._inject_semantic_equivalence_policy(_build_payload())
    identity = api.build_active_path_identity({"stage_status": {}, "diagnostics": {}}, env={})
    assert identity["editorial_slot_resolution_evidence"]
    assert identity["editorial_slot_resolution_evidence"][0]["policy_injected"] is True


def test_run_single_universal_clean_cut_validations_own_entry_resets_first():
    """The real per-job entry point resets evidence before doing anything
    else -- proven here by calling it directly and checking a stale
    evidence row from a PRIOR (simulated) job never reaches the
    unsupported-key ValueError path with old evidence still attached."""
    install._inject_semantic_equivalence_policy(_build_payload())
    assert api._real_editorial_slot_resolution_evidence()

    from cutsell_worker.universal_clean_cut_validation import run_single_universal_clean_cut_validation
    try:
        run_single_universal_clean_cut_validation("not-a-real-video-key")
    except ValueError:
        pass  # expected -- we only care that reset ran before this raised
    assert api._real_editorial_slot_resolution_evidence() == ()
