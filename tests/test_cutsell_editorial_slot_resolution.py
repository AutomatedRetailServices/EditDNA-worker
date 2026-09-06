from cutsell_worker.editorial_slot_resolution_install import (
    _SLOT_RULES,
    _inject_contract,
    install_editorial_slot_resolution,
)


def test_editorial_slot_contract_is_inserted_before_legacy_selection_rules():
    payload = {
        "task": "cutsell_unified_whole_video_selection",
        "editorial_contract": [
            "Understand the full creator message before deciding any individual take.",
            "First infer idea families and retry relationships across the entire timeline.",
            "Preserve numbers, negations, names, causal claims, and genuinely new story facts.",
        ],
    }

    updated = _inject_contract(payload)
    contract = updated["editorial_contract"]

    assert contract[0] == payload["editorial_contract"][0]
    assert contract[1] == payload["editorial_contract"][1]
    assert contract[2 : 2 + len(_SLOT_RULES)] == list(_SLOT_RULES)
    assert contract.index(_SLOT_RULES[0]) < contract.index(
        "Preserve numbers, negations, names, causal claims, and genuinely new story facts."
    )


def test_editorial_slot_contract_injection_is_idempotent():
    payload = {"editorial_contract": ["Understand the full creator message before deciding any individual take."]}
    once = _inject_contract(payload)
    twice = _inject_contract(once)
    assert twice["editorial_contract"].count(_SLOT_RULES[0]) == 1


def test_contract_explicitly_separates_complete_retry_competition_from_composite():
    joined = "\n".join(_SLOT_RULES).lower()
    assert "minimum sufficient editorial set" in joined
    assert "same editorial function" in joined
    assert "required propositions" in joined
    assert "supporting/restated/elaborative detail" in joined
    assert "composite only when no single realization is sufficient" in joined


def test_installer_patches_unified_selection_payload_builder_once():
    from cutsell_worker import unified_selection_google as module

    install_editorial_slot_resolution()
    first = module.build_unified_selection_payload
    install_editorial_slot_resolution()
    second = module.build_unified_selection_payload

    assert first is second
    assert getattr(second, "_cutsell_editorial_slot_resolution", False) is True
