"""PO 2026-10-23: Bulk and Mix calls go through the Batch API (50 %), never more than 30 minutes of waiting."""
import json
from types import SimpleNamespace

from cutsell_worker.simple_engine import llm


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _resp(code, data=None, text=""):
    return SimpleNamespace(status_code=code, json=lambda: data, text=text)


def test_a_batch_that_ends_gives_the_same_message():
    clock = Clock()
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["body"] = json
        return _resp(200, {"id": "b1", "processing_status": "in_progress"})

    polls = iter([{"id": "b1", "processing_status": "in_progress"},
                  {"id": "b1", "processing_status": "ended", "results_url": "https://r/results-1"}])

    def get(url, headers=None, timeout=None):
        if url.endswith("/b1"):
            return _resp(200, next(polls))
        cid = seen["body"]["requests"][0]["custom_id"]
        line = json.dumps({"custom_id": cid, "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": "OK"}], "usage": {"input_tokens": 10, "output_tokens": 5}}}})
        return _resp(200, text=line)

    body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]}
    with llm.batch_scope(True):
        llm._BATCH_DEADLINE.set(clock.now() + 1800)
        message, why = llm.via_batch(body, {}, post=post, get=get, sleep=clock.sleep, now=clock.now)
    assert why == "batch" and message["content"][0]["text"] == "OK"
    assert seen["body"]["requests"][0]["params"] == body          # the very same request (the Gold cannot change)


def test_past_30_minutes_the_batch_is_cancelled_and_the_normal_api_answers():
    clock = Clock()
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return _resp(200, {"id": "b2", "processing_status": "in_progress"})

    def get(url, headers=None, timeout=None):
        return _resp(200, {"id": "b2", "processing_status": "in_progress"})

    token = llm._BATCH_DEADLINE.set(clock.now() + 30 * 60)
    try:
        message, why = llm.via_batch({"model": "m"}, {}, post=post, get=get, sleep=clock.sleep, now=clock.now)
    finally:
        llm._BATCH_DEADLINE.reset(token)
    assert message is None and why == "timeout" and calls[-1].endswith("/b2/cancel")
    assert clock.t - 1000.0 <= 30 * 60 + 15                      # never more than 30 minutes (+ one poll)


def test_an_errored_request_falls_back():
    clock = Clock()
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["cid"] = json["requests"][0]["custom_id"]
        return _resp(200, {"id": "b3", "processing_status": "ended", "results_url": "https://r/b3"})

    def get(url, headers=None, timeout=None):
        return _resp(200, text=json.dumps({"custom_id": seen["cid"], "result": {"type": "errored"}}))

    token = llm._BATCH_DEADLINE.set(clock.now() + 60)
    try:
        assert llm.via_batch({}, {}, post=post, get=get, sleep=clock.sleep, now=clock.now) == (None, "result_errored")
    finally:
        llm._BATCH_DEADLINE.reset(token)


def test_single_and_combine_never_batch_and_the_switch_turns_it_off(monkeypatch):
    assert llm.batch_deadline() is None                              # outside a Bulk / Mix scope
    with llm.batch_scope(False):
        assert llm.batch_deadline() is None
    monkeypatch.setenv("CUTSELL_ENGINE_BATCH", "0")
    with llm.batch_scope(True):
        assert llm.batch_deadline() is None


def test_batch_halves_every_price_and_the_report_says_so():
    u = {"input_tokens": 1000, "output_tokens": 2000, "cache_read_input_tokens": 9000, "batch": True, "batch_wait_s": 120}
    r = llm.cache_report([u])
    assert abs(r["usd"] - r["usd_normal_api"] / 2) < 1e-9 and r["batch_calls"] == 1 and r["saved_by_batch_usd"] > 0
    f = llm.cache_report([{**u, "batch": False, "batch_fallback": "timeout"}])
    assert f["batch_fallbacks"] == 1 and f["saved_by_batch_usd"] == 0
