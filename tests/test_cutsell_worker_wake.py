"""worker_wake: the API's best-effort "a job was enqueued" call to the on-demand Modal worker."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cutsell_worker import queueing, worker_wake


class _FakeRequests:
    def __init__(self, status=200, raises=None):
        self.calls = []
        self._status, self._raises = status, raises

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if self._raises:
            raise self._raises
        return SimpleNamespace(status_code=self._status)


@pytest.fixture
def fake_requests(monkeypatch):
    import sys

    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def test_does_nothing_when_not_configured(monkeypatch, fake_requests):
    monkeypatch.delenv("CUTSELL_WORKER_WAKE_URL", raising=False)
    assert worker_wake.wake_worker() == {"status": "not_configured"}
    assert fake_requests.calls == []


def test_refuses_plain_http_or_a_missing_token(monkeypatch, fake_requests):
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_URL", "http://example.test/wake")
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_TOKEN", "t")
    assert worker_wake.wake_worker() == {"status": "misconfigured"}
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_URL", "https://example.test/wake")
    monkeypatch.delenv("CUTSELL_WORKER_WAKE_TOKEN")
    assert worker_wake.wake_worker() == {"status": "misconfigured"}
    assert fake_requests.calls == []


def test_posts_the_token_and_reports_success(monkeypatch, fake_requests):
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_URL", "https://example.test/wake")
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_TOKEN", "secret-token")
    assert worker_wake.wake_worker() == {"status": "woken"}
    assert fake_requests.calls == [("https://example.test/wake", {"token": "secret-token"}, worker_wake.WAKE_TIMEOUT_SEC)]


def test_a_failed_wake_never_raises(monkeypatch, fake_requests):
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_URL", "https://example.test/wake")
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_TOKEN", "t")
    fake_requests._status = 500
    assert worker_wake.wake_worker() == {"status": "failed", "http_status": 500}
    fake_requests._raises = TimeoutError("slow")
    assert worker_wake.wake_worker() == {"status": "failed", "reason": "TimeoutError"}


def test_every_enqueue_wakes_the_worker_after_the_job_is_on_the_queue(monkeypatch):
    order = []

    class FakeQueue:
        name = "cutsell"

        def enqueue(self, *args, **kwargs):
            order.append("enqueue")
            return SimpleNamespace(id="job-1")

    monkeypatch.setattr(queueing, "wake_worker", lambda: order.append("wake"))
    monkeypatch.setattr(queueing, "reserve_processing_slot", lambda **kw: {"allowed": True})
    queue = FakeQueue()
    queueing.enqueue_flow_b({"user_id": "u", "project_id": "p"}, queue=queue)
    queueing.enqueue_export({"user_id": "u", "project_id": "p"}, queue=queue)
    queueing.enqueue_batch_item(batch_id="b", user_id="u", index=0, queue=queue)
    assert order == ["enqueue", "wake"] * 3
