from pathlib import Path

from cutsell_worker.commercial_store import (
    durable_upsert_project,
    initialize_schema,
    record_usage,
    upsert_user,
    usage_total,
)
from cutsell_worker.config import load_runtime_config
from cutsell_worker.usage_limits import check_processing_allowance


def _db_url(tmp_path: Path) -> str:
    return "sqlite:///" + str(tmp_path / "cutsell-commercial.db")


def test_commercial_store_persists_users_projects_and_usage(tmp_path):
    url = _db_url(tmp_path)
    initialize_schema(url)
    user = upsert_user(url, user_id="usr_1", apple_subject="apple-subject-1", email="creator@example.com")
    assert user["user_id"] == "usr_1"
    assert user["apple_subject"] == "apple-subject-1"

    durable_upsert_project(url, {
        "project_id": "prj_1",
        "user_id": "usr_1",
        "title": "Demo cut",
        "state": "draft_ready",
        "created_at": "2026-08-08T00:00:00+00:00",
        "updated_at": "2026-08-08T00:00:00+00:00",
        "sources": [],
    })
    record_usage(
        url,
        event_id="usage_1",
        user_id="usr_1",
        project_id="prj_1",
        event_type="processing_minutes",
        quantity=4.5,
        unit="minutes",
    )
    assert usage_total(url, user_id="usr_1", event_type="processing_minutes") == 4.5


def test_runtime_config_exposes_commercial_guardrails():
    config = load_runtime_config({
        "DATABASE_URL": "sqlite:///tmp.db",
        "SENTRY_DSN": "https://example.invalid/1",
        "CUTSELL_MAX_SOURCE_MINUTES": "20",
        "CUTSELL_MAX_CONCURRENT_JOBS_PER_USER": "3",
        "CUTSELL_MONTHLY_PROCESSING_MINUTES": "250",
    })
    assert config.commercial_db_ready is True
    assert config.sentry_dsn_present is True
    assert config.max_source_minutes == 20
    assert config.max_concurrent_jobs_per_user == 3
    assert config.monthly_processing_minutes == 250


def test_usage_guard_rejects_oversized_source(monkeypatch):
    monkeypatch.setenv("CUTSELL_MAX_SOURCE_MINUTES", "5")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = check_processing_allowance(user_id="usr_1", durations_sec=[301])
    assert result.allowed is False
    assert result.reason == "source_duration_limit"
