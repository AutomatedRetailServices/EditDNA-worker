"""PO 8 Nov 2026: when a subscription really ends (period paid or trial over), everything is deleted, with a dated
warning while it runs out; paying again cancels the deletion. No payments yet -> never deletes anything."""
import fakeredis
import pytest

from cutsell_worker import subscription_lifecycle as sub

DAY = 86400.0
NOW = 1_800_000_000.0


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def test_no_payments_yet_means_nothing_is_ever_deleted(r):
    calls = []
    assert sub.status("u", client=r) == {"state": "none"}
    assert sub.purge_ended(client=r, now=NOW, delete=calls.append)["due"] == 0 and calls == []


def test_canceled_warns_with_the_date_and_is_deleted_only_after_the_paid_period(r):
    calls = []
    sub.record("u", state="canceled", ends_at=NOW + 10 * DAY, client=r, now=NOW)
    s = sub.status("u", client=r)
    assert s["state"] == "canceled" and s["deletes_on"] in s["warning"] and "can't be recovered" in s["warning"]
    assert "No se pueden recuperar" in s["warning_es"]
    assert sub.purge_ended(client=r, now=NOW + 9 * DAY, delete=calls.append)["deleted"] == 0 and calls == []
    out = sub.purge_ended(client=r, now=NOW + 10 * DAY + 2 * 3600, delete=lambda u: calls.append(u) or {"status": "deleted"})
    assert out["deleted"] == 1 and calls == ["u"] and sub.status("u", client=r) == {"state": "none"}


def test_paying_again_before_the_date_cancels_the_deletion(r):
    calls = []
    sub.record("u", state="canceled", ends_at=NOW + DAY, client=r, now=NOW)
    sub.record("u", state="active", client=r, now=NOW + 0.5 * DAY)
    assert sub.purge_ended(client=r, now=NOW + 5 * DAY, delete=calls.append)["due"] == 0 and calls == []


def test_a_partial_deletion_is_finished_on_the_next_run(r):
    sub.record("u", state="expired", ends_at=NOW, client=r, now=NOW)
    assert sub.purge_ended(client=r, now=NOW + DAY, delete=lambda u: {"status": "partial"})["failed"] == 1
    assert sub.purge_ended(client=r, now=NOW + 2 * DAY, delete=lambda u: {"status": "deleted"})["deleted"] == 1


def test_bad_states_are_refused(r):
    with pytest.raises(ValueError):
        sub.record("u", state="paused", client=r)
    with pytest.raises(ValueError):
        sub.record("u", state="canceled", client=r)
