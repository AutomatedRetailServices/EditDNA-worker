"""D-288 -- typed perceptual delivery authority.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288, real RAW #122
evidence): `perceptual_watch_listen.PerceptualReview.watch_listen_status`
(D-154/D-155's own typed 4-state authority -- BLOCKED / HUMAN_REVIEW_
REQUIRED / SYSTEM_PASS / HUMAN_APPROVED) was already computed correctly on
every run, but nothing in the delivery chain ever read it:
`universal_clean_cut_validation._live_render_qc_diagnostics` built its
`delivery_status` string from the coarse `PerceptualReview.status`
(PASS/FAIL/UNCERTAIN) instead, `human_watch_listen_required` was hardcoded
`True` regardless of the real verdict, and the real production export path
(`export_job.run_export_job`) never called the perceptual reviewer at all --
a technical-QC PASS alone was sufficient to mark a project "finished" and
hand back a real `download_url`.

This file proves, at the `render_delivery.py` contract level (the shared
authority both `export_job.py` and any future caller must go through): a
BLOCKED verdict can never reach a ready delivery status, no matter what;
a HUMAN_REVIEW_REQUIRED verdict holds the file at PENDING (available for
inspection, not auto-deliverable); SYSTEM_PASS/HUMAN_APPROVED proceed
normally; an approval is honored only when it is bound to the EXACT
current artifact (render_identity AND output_sha256); and a caller that
never opts into the gate (`watch_listen_status=None`) is byte-identical to
pre-D-288 behavior. No Video00 fact/id anywhere below.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker import render_delivery as rd
from cutsell_worker.perceptual_watch_listen import (
    WATCH_LISTEN_BLOCKED,
    WATCH_LISTEN_HUMAN_APPROVED,
    WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS,
)


def _rendered_file(tmp_path, content=b"rendered-bytes") -> str:
    path = tmp_path / "out.mp4"
    path.write_bytes(content)
    return str(path)


def _build(tmp_path, *, watch_listen_status, require_upload=False):
    return rd.build_render_delivery_record(
        render_identity="render_" + "a" * 24,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=_rendered_file(tmp_path),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=require_upload,
        watch_listen_status=watch_listen_status,
    )


# =============================================================================
# The gate itself
# =============================================================================

def test_watch_listen_blocked_never_reaches_ready(tmp_path):
    record = _build(tmp_path, watch_listen_status=WATCH_LISTEN_BLOCKED)
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_BLOCKED
    assert record.ready_for_delivery is False
    assert "watch_listen_blocked" in record.errors


def test_watch_listen_human_review_required_holds_file_for_inspection_not_ready(tmp_path):
    record = _build(tmp_path, watch_listen_status=WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    assert record.delivery_status == rd.DELIVERY_STATUS_WATCH_LISTEN_PENDING
    assert record.ready_for_delivery is False
    # "archivo disponible para inspección": the file itself is still real,
    # hashed, and technically clean -- only the DELIVERY gate is held.
    assert record.output_sha256 is not None
    assert record.technical_qc_status == rd.TECHNICAL_QC_STATUS_PASS


def test_watch_listen_system_pass_reaches_ready(tmp_path):
    record = _build(tmp_path, watch_listen_status=WATCH_LISTEN_SYSTEM_PASS)
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD
    assert record.ready_for_delivery is False  # local-only, require_upload=False stops here (D-267 Stage 7)


def test_watch_listen_human_approved_reaches_ready(tmp_path):
    record = _build(tmp_path, watch_listen_status=WATCH_LISTEN_HUMAN_APPROVED)
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD


def test_watch_listen_system_pass_with_upload_reaches_delivery_ready(tmp_path):
    record = rd.build_render_delivery_record(
        render_identity="render_" + "a" * 24,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=_rendered_file(tmp_path),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
        watch_listen_status=WATCH_LISTEN_SYSTEM_PASS,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY
    assert record.ready_for_delivery is True


def test_watch_listen_gate_runs_before_upload_branch(tmp_path):
    """A negative control: even a fully-succeeded upload never overrides a
    BLOCKED perceptual verdict -- the gate runs before the upload branch."""
    record = rd.build_render_delivery_record(
        render_identity="render_" + "a" * 24,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=_rendered_file(tmp_path),
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
        watch_listen_status=WATCH_LISTEN_BLOCKED,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_DELIVERY_BLOCKED
    assert record.ready_for_delivery is False


def test_unrecognized_watch_listen_status_fails_closed_to_pending(tmp_path):
    """WHEN UNCERTAIN, KEEP: a malformed/unknown status string is never
    silently treated as SYSTEM_PASS."""
    record = _build(tmp_path, watch_listen_status="NOT_A_REAL_STATUS")
    assert record.delivery_status == rd.DELIVERY_STATUS_WATCH_LISTEN_PENDING


def test_opting_out_with_none_is_byte_identical_to_pre_d288_behavior(tmp_path):
    """`watch_listen_status=None` (every pre-D-288 caller, and this
    module's own D-267 test suite) leaves the gate unapplied entirely."""
    record = _build(tmp_path, watch_listen_status=None)
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD


# =============================================================================
# "archivo disponible para inspección" vs "aprobado para entrega" (diagnostics)
# =============================================================================

def test_diagnostics_separate_file_available_from_approved_for_delivery(tmp_path):
    blocked = _build(tmp_path, watch_listen_status=WATCH_LISTEN_BLOCKED)
    diag = rd.render_delivery_diagnostics(blocked)
    assert diag["file_available_for_inspection"] is True
    assert diag["approved_for_delivery"] is False
    assert diag["watch_listen_status"] == WATCH_LISTEN_BLOCKED

    passed = _build(tmp_path, watch_listen_status=WATCH_LISTEN_SYSTEM_PASS)
    diag_pass = rd.render_delivery_diagnostics(passed)
    assert diag_pass["file_available_for_inspection"] is True
    assert diag_pass["approved_for_delivery"] is True


# =============================================================================
# Traceable, artifact-bound HUMAN_APPROVED (negative tests required by the gate)
# =============================================================================

def _approval(*, render_identity="render_" + "a" * 24, output_sha256="sha-1", approved=True):
    return rd.WatchListenApproval(
        render_identity=render_identity, output_sha256=output_sha256,
        approved=approved, approver="reviewer@example.com", approved_at=1_700_000_000.0,
    )


def test_approval_bound_to_exact_artifact_promotes_human_review_required():
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=_approval(),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_APPROVED
    assert diag["human_approval_applied"] is True


def test_approval_bound_to_exact_artifact_promotes_system_pass():
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_SYSTEM_PASS,
        approval=_approval(),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_APPROVED


def test_blocked_is_never_promotable_even_when_approved():
    """Negative control (explicitly required): BLOCKED can never be
    approved away -- a confirmed perceptual defect is a root-authority fix
    or a re-render, never something human sign-off launders through this
    gate (mirrors `perceptual_watch_listen.apply_human_watch_listen_
    approval`'s own contract)."""
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_BLOCKED,
        approval=_approval(),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_BLOCKED
    assert diag["human_approval_applied"] is False
    assert diag["reason"] == "blocked_never_promotable"


def test_stale_approval_from_a_different_render_identity_never_promotes():
    """Negative control (explicitly required): "aprobación obsoleta no
    sirve para otro render" -- an approval bound to a DIFFERENT render
    never carries over to a new candidate, even one from a re-run of what
    looks like the same edit."""
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=_approval(render_identity="render_" + "b" * 24),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert diag["human_approval_applied"] is False
    assert diag["reason"] == "approval_bound_to_different_artifact"


def test_stale_approval_from_a_different_output_hash_never_promotes():
    """Same render_identity (identical plan) but a DIFFERENT output_sha256
    -- D-267's own documented non-determinism (two encodes of the same
    plan can legitimately hash differently). The approval must still not
    carry over to bytes a human never actually watched."""
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=_approval(output_sha256="sha-DIFFERENT"),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert diag["human_approval_applied"] is False


def test_rejected_approval_never_promotes():
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=_approval(approved=False),
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert diag["reason"] == "approval_recorded_as_rejected"


def test_no_approval_at_all_never_promotes():
    status, diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=None,
        current_render_identity="render_" + "a" * 24,
        current_output_sha256="sha-1",
    )
    assert status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
    assert diag["reason"] == "no_approval_recorded"


def test_end_to_end_approval_then_build_reaches_ready(tmp_path):
    """The full contract: an artifact-bound approval promotes the status,
    and THAT promoted status is what a caller then passes into
    `build_render_delivery_record` to actually reach a ready state."""
    render_identity = "render_" + "c" * 24
    final_path = _rendered_file(tmp_path)
    output_sha256 = rd.compute_output_sha256(final_path)
    promoted, _diag = rd.resolve_watch_listen_status_for_delivery(
        WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
        approval=_approval(render_identity=render_identity, output_sha256=output_sha256),
        current_render_identity=render_identity,
        current_output_sha256=output_sha256,
    )
    record = rd.build_render_delivery_record(
        render_identity=render_identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=final_path,
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        watch_listen_status=promoted,
    )
    assert record.delivery_status == rd.DELIVERY_STATUS_READY_FOR_UPLOAD
    assert record.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED
