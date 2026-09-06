from cutsell_worker.editorial_slot_resolution_install import (
    _SEMANTIC_EQUIVALENCE_POLICY,
    _inject_semantic_equivalence_policy,
)
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalencePair, IdeaEquivalenceRequest
from cutsell_worker.semantic_idea_equivalence_google import build_semantic_equivalence_request


def _prompt(payload):
    return payload["contents"][0]["parts"][0]["text"]


def test_editorial_slot_policy_is_injected_into_active_semantic_equivalence_request():
    request = IdeaEquivalenceRequest(pairs=(
        IdeaEquivalencePair(
            left_text="This is my experience; only a small percentage is hereditary.",
            right_text="I am the first in my family; science says only a small percentage is hereditary.",
        ),
    ))

    base = build_semantic_equivalence_request(request, max_output_tokens=320)
    injected = _inject_semantic_equivalence_policy(base)
    prompt = _prompt(injected)

    assert _SEMANTIC_EQUIVALENCE_POLICY in prompt
    assert "supporting facts differ" in prompt
    assert "second complete conclusion/restatement" in prompt
    assert "Do not rank or choose a winner here" in prompt


def test_editorial_slot_policy_injection_is_idempotent():
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "base prompt"}]}],
        "generationConfig": {"temperature": 0.0},
    }

    once = _inject_semantic_equivalence_policy(payload)
    twice = _inject_semantic_equivalence_policy(once)

    assert _prompt(twice).count(_SEMANTIC_EQUIVALENCE_POLICY) == 1
    assert payload["contents"][0]["parts"][0]["text"] == "base prompt"


def test_policy_preserves_complementary_story_beats_as_distinct():
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "base prompt"}]}],
    }

    prompt = _prompt(_inject_semantic_equivalence_policy(payload))

    assert "genuinely distinct required story proposition" in prompt
    assert "complementary next story beat is DIFFERENT" in prompt
