"""Remembered AI answers (PO 6 Nov 2026): the same video gives the same cut, and the repeat costs nothing."""
import fakeredis
import pytest

from cutsell_worker.simple_engine import engine
from cutsell_worker.simple_engine import llm as _llm


@pytest.fixture
def r(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(_llm, "_memory_client", lambda: client)
    monkeypatch.delenv("CUTSELL_SIMPLE_ENGINE_REMEMBER", raising=False)
    return client


@pytest.fixture
def model(monkeypatch):
    calls = []

    def fake(prompt, **kwargs):
        calls.append(str(prompt))
        return '{"hide": [[1, 1, "x%d"]]}' % len(calls), {"input_tokens": 100, "output_tokens": 10}
    monkeypatch.setattr(_llm, "_call_anthropic", fake)
    return calls


def _ask(text):
    return engine._ask(_llm.call_anthropic, "v3dec-0", ("RULES\n", text))


def test_the_same_words_get_the_same_answer_without_asking_again(r, model):
    with _llm.memory_owner("usr_me"):
        first, _ = _ask("L1| hola")
        again, usage = _ask("L1| hola")
    assert first == again and len(model) == 1
    assert usage[0]["remembered"] and usage[0]["input_tokens"] == 0
    assert _llm.cache_report(usage)["usd"] == 0


def test_different_words_or_another_creator_ask_the_model(r, model):
    with _llm.memory_owner("usr_me"):
        _ask("L1| hola")
        _ask("L1| adios")
    with _llm.memory_owner("usr_other"):
        _ask("L1| hola")
    assert len(model) == 3


def test_switch_off_and_no_owner_never_remember(r, model, monkeypatch):
    _ask("L1| hola")
    _ask("L1| hola")
    monkeypatch.setenv("CUTSELL_SIMPLE_ENGINE_REMEMBER", "0")
    with _llm.memory_owner("usr_me"):
        _ask("L1| hola")
        _ask("L1| hola")
    assert len(model) == 4 and not r.keys()


def test_a_broken_memory_never_breaks_the_cut(monkeypatch, model):
    class Down:
        def get(self, *a):
            raise ConnectionError

        def set(self, *a, **k):
            raise ConnectionError
    monkeypatch.setattr(_llm, "_memory_client", lambda: Down())
    with _llm.memory_owner("usr_me"):
        out, _ = _ask("L1| hola")
    assert out["hide"] and len(model) == 1


def test_account_deletion_forgets_her_answers(r, model):
    with _llm.memory_owner("usr_me"):
        _ask("L1| hola")
    with _llm.memory_owner("usr_other"):
        _ask("L1| hola")
    assert len(list(r.scan_iter(match=_llm.memory_pattern("usr_me")))) == 1 and len(r.keys()) == 2
