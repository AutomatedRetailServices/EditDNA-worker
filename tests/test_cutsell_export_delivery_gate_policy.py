"""CUTSELL_EXPORT_DELIVERY_GATE: the Product Owner's 2026-10-06 decision that a technically clean
export is delivered directly instead of waiting for a human approval the mobile app cannot give.

Reuses the D-288 gate test's own fixtures so the same real `_tenant_safe_deliver` seam is exercised.
"""
from __future__ import annotations

import pytest

from cutsell_worker import config as cutsell_config
from cutsell_worker import export_job
from cutsell_worker import render_delivery as rd
from cutsell_worker.perceptual_watch_listen import WATCH_LISTEN_BLOCKED, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED

from tests.test_cutsell_d288_export_job_perceptual_gate import (  # noqa: F401  (fixtures)
    _fake_draft, _fake_qc_result, _plan, _rendered_file, _stub_review, fake_redis, fake_s3, wire_real_store_export,
)


def _deliver(tmp_path, **extra):
    return export_job._tenant_safe_deliver(
        output_path=_rendered_file(tmp_path), plan=_plan(), draft=_fake_draft(), local_paths={},
        qc_result=_fake_qc_result(), project_id="proj-1", user_id="user-1", job_id="job-1", **extra,
    )


def test_gate_defaults_to_human_review_and_rejects_unknown_values():
    assert cutsell_config.export_delivery_gate({}) == "human_review"
    assert cutsell_config.export_delivery_gate({"CUTSELL_EXPORT_DELIVERY_GATE": " Technical_QC "}) == "technical_qc"
    with pytest.raises(ValueError):
        cutsell_config.export_delivery_gate({"CUTSELL_EXPORT_DELIVERY_GATE": "off"})


def test_default_still_holds_a_human_review_required_export(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    monkeypatch.delenv("CUTSELL_EXPORT_DELIVERY_GATE", raising=False)
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    with pytest.raises(export_job.PendingHumanWatchListenReview):
        _deliver(tmp_path, pending_review_redis_client=fake_redis)


def test_technical_qc_policy_delivers_a_human_review_required_export(tmp_path, wire_real_store_export, fake_redis, monkeypatch):
    monkeypatch.setenv("CUTSELL_EXPORT_DELIVERY_GATE", "technical_qc")
    _stub_review(monkeypatch, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED)
    result = _deliver(tmp_path)
    assert result["delivery_status"] == rd.DELIVERY_STATUS_DELIVERY_READY
    assert result["download_url"] is not None
    assert result["watch_listen_status"] is None            # never reported as a pass it did not earn


def test_technical_qc_policy_still_never_delivers_a_blocked_export(tmp_path, wire_real_store_export, monkeypatch):
    monkeypatch.setenv("CUTSELL_EXPORT_DELIVERY_GATE", "technical_qc")
    _stub_review(monkeypatch, WATCH_LISTEN_BLOCKED)
    with pytest.raises(export_job.TenantSafeDeliveryBlocked):
        _deliver(tmp_path)
    assert wire_real_store_export.objects == {}
