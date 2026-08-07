"""Align Take Judge tests with editable Flow B: losers stay as alternates."""
from pathlib import Path

path = Path("tests/test_take_judge_v2.py")
text = path.read_text(encoding="utf-8")
old = '''@pytest.mark.parametrize("outcome,removed,statuses", [\n    (result(), ["b"], ["candidate_winner", "candidate_loser"]),\n    (result(None, abstain=True), [], ["abstained", "abstained"]),\n    (result(confidence=.69), [], ["low_confidence", "low_confidence"])])\ndef test_selection_winner_abstain_and_low_confidence(monkeypatch, outcome, removed, statuses):\n    group, outside, calls = configure_pipeline(monkeypatch, outcome)\n    pipeline.run_take_judge(group + [outside], "/private/session", "/private/input.mp4")\n    assert [item["id"] for item in group if not item["meta"]["keep"]] == removed\n    assert group[0]["meta"]["keep"] and outside["meta"]["keep"] and len(calls) == 1\n    assert [item["meta"]["take_judge_execution_status"] for item in group] == statuses\n    assert outside["meta"]["take_judge_execution_status"] == "not_candidate"\n'''
new = '''@pytest.mark.parametrize("outcome,statuses", [\n    (result(), ["candidate_winner", "candidate_loser"]),\n    (result(None, abstain=True), ["abstained", "abstained"]),\n    (result(confidence=.69), ["low_confidence", "low_confidence"])])\ndef test_selection_winner_abstain_and_low_confidence_preserves_alternates(monkeypatch, outcome, statuses):\n    group, outside, calls = configure_pipeline(monkeypatch, outcome)\n    pipeline.run_take_judge(group + [outside], "/private/session", "/private/input.mp4")\n    assert all(item["meta"]["keep"] for item in group + [outside])\n    assert len(calls) == 1\n    assert [item["meta"]["take_judge_execution_status"] for item in group] == statuses\n    if outcome.winner_id and not outcome.abstain and outcome.confidence >= pipeline.TAKE_JUDGE_V2_MIN_CONFIDENCE:\n        assert group[0]["meta"]["take_judge_selected"] is True\n        assert group[1]["meta"]["take_judge_selected"] is False\n    assert outside["meta"]["take_judge_execution_status"] == "not_candidate"\n'''
if old in text:
    text = text.replace(old, new, 1)
elif "test_selection_winner_abstain_and_low_confidence_preserves_alternates" not in text:
    raise SystemExit("Unexpected Take Judge test baseline")
path.write_text(text, encoding="utf-8")
print("Take Judge tests aligned with editable alternates")
