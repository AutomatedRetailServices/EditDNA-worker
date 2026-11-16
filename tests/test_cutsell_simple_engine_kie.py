"""The separate kie.ai copy of the engine's model call (PO 16 Nov 2026). The original engine files are untouched."""
from __future__ import annotations

import json

import pytest

from cutsell_worker.simple_engine import llm as base
from cutsell_worker.simple_engine_kie import llm_kie


class Resp:
    def __init__(self, status, data):
        self.status_code, self.text = status, json.dumps(data)


class Fake:
    def __init__(self, answers):
        self.answers, self.bodies = list(answers), []

    def post(self, url, json=None, headers=None, timeout=None):
        assert url == llm_kie.KIE_URL and headers["Authorization"] == "Bearer k"
        self.bodies.append(json)
        return self.answers.pop(0)


def ok(text, credits=0.9):
    return Resp(200, {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "credits_consumed": credits,
                      "usage": {"input_tokens": 8, "cache_creation_input_tokens": 3000, "output_tokens": 300}})


SCHEMA = {"type": "object", "properties": {"hide": {"type": "array"}}, "required": ["hide"]}


def test_schema_goes_into_the_question_and_fences_are_removed(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "k")
    fake = Fake([ok('Sure:\n```json\n{"hide": [3]}\n```')])
    text, usage = llm_kie.call_kie(base.Prompt("fixed rules", "video words", schema=SCHEMA), session=fake)
    assert json.loads(text) == {"hide": [3]}
    body = fake.bodies[0]
    assert "format" not in body.get("output_config", {})                 # kie.ai ignores it: not sent
    assert "JSON schema" in body["messages"][0]["content"][-1]["text"]    # it is in the question instead
    assert usage["provider"] == "kie" and usage["cost_usd"] == pytest.approx(0.0045)
    assert usage["cache_read_input_tokens"] == 0


def test_a_non_json_answer_is_asked_again_once(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "k")
    fake = Fake([ok("I think lines 3 and 4"), ok('{"hide": [3, 4]}')])
    text, usage = llm_kie.call_kie(base.Prompt("f", "v", schema=SCHEMA), session=fake, sleep=lambda s: None)
    assert json.loads(text)["hide"] == [3, 4]
    assert usage["reasks"] == 1 and usage["calls"] == 2
    assert "not valid JSON" in fake.bodies[1]["messages"][0]["content"][-1]["text"]


def test_errors_raise_so_the_comparison_sees_them(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "k")
    with pytest.raises(RuntimeError):
        llm_kie.call_kie(base.Prompt("f", "v"), session=Fake([Resp(500, {})] * 3), sleep=lambda s: None)
    monkeypatch.delenv("KIE_API_KEY")
    with pytest.raises(RuntimeError, match="KIE_API_KEY"):
        llm_kie.call_kie(base.Prompt("f", "v"), session=Fake([]))


def test_the_step_effort_is_the_same_as_the_original(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "k")
    monkeypatch.delenv("CUTSELL_SIMPLE_ENGINE_EFFORT", raising=False)
    fake = Fake([ok('{"hide": []}')])
    with base.step("pass"):
        llm_kie.call_kie(base.Prompt("f", "v", schema=SCHEMA), session=fake)
    assert fake.bodies[0]["output_config"] == {"effort": "medium"}
