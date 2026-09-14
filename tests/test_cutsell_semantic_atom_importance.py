"""General semantic-atom importance classification (D-031).

RAW 33402023395's own failure shape, generalized: a discarded clip's
incidental year blocked Freeze even though the Human Gold oracle itself
does not preserve it. No Video00 fact/phrase/literal value is hardcoded
in any fixture below -- every case uses generic, made-up subject matter to
prove the mechanism is general.
"""
from cutsell_worker.semantic_atom_importance import (
    CONTEXTUAL,
    CRITICAL,
    UNCERTAIN,
    blocks_freeze,
    classify_negation_atom,
    classify_number_atom,
    resolve_uncertain_with_arbiter,
)


def test_negation_is_always_critical():
    result = classify_negation_atom("no")
    assert result.importance == CRITICAL
    assert result.atom_type == "negation"


def test_incidental_year_in_ordinary_temporal_aside_is_contextual():
    text = "During one period in 2023 I had some minor issues with my equipment."
    result = classify_number_atom("2023", text)
    assert result.importance == CONTEXTUAL
    assert result.resolved_by == "deterministic"


def test_year_with_correction_language_is_critical():
    # The canonical directive's own example: "diagnosed in 2023 instead of 2022".
    text = "I was actually diagnosed in 2023, instead of 2022 like I first said."
    result = classify_number_atom("2023", text)
    assert result.importance == CRITICAL
    assert result.evidence == "correction_language_present"


def test_year_with_chronology_relation_language_is_critical():
    text = "I started feeling worse in 2023, and before that I had no symptoms at all."
    result = classify_number_atom("2023", text)
    assert result.importance == CRITICAL
    assert result.evidence == "chronology_relation_language_present"


def test_percentage_is_critical():
    result = classify_number_atom("15", "Only 15% of users ever finish the full course.")
    assert result.importance == CRITICAL
    assert result.evidence == "percentage"


def test_price_is_critical():
    result = classify_number_atom("49", "It costs $49 for the whole bundle.")
    assert result.importance == CRITICAL
    assert result.evidence == "price"


def test_measurement_is_critical():
    result = classify_number_atom("3", "The nodule was 3 centimeters across.")
    assert result.importance == CRITICAL
    assert result.evidence == "measurement"


def test_dose_is_critical():
    result = classify_number_atom("2", "Take 2 pills every morning for a month.")
    assert result.importance == CRITICAL
    assert result.evidence == "dose_or_quantity"


def test_bare_ambiguous_quantity_is_uncertain_and_blocks():
    # No unit, no currency, no percent, no correction, not a plausible year.
    result = classify_number_atom("7", "I tried it 7 different ways and it finally worked well.")
    assert result.importance == UNCERTAIN
    assert blocks_freeze(result.importance) is True


def test_contextual_year_does_not_block_but_uncertain_and_critical_do():
    assert blocks_freeze(CONTEXTUAL) is False
    assert blocks_freeze(CRITICAL) is True
    assert blocks_freeze(UNCERTAIN) is True


class _FakeArbiter:
    def __init__(self, verdict):
        self._verdict = verdict

    def classify_atom(self, atom_text, source_sentence, kept_text):
        return self._verdict


def test_uncertain_atom_resolved_contextual_by_confirming_arbiter():
    pending = [classify_number_atom("7", "I tried it 7 different ways and it finally worked well.")]
    arbiter = _FakeArbiter((CONTEXTUAL, 0.8, "removing this does not change the claim"))

    resolved = resolve_uncertain_with_arbiter(
        pending, source_text="I tried it 7 different ways and it finally worked well.",
        kept_text="", arbiter=arbiter,
    )

    assert resolved[0].importance == CONTEXTUAL
    assert resolved[0].resolved_by == "semantic_arbiter"


def test_uncertain_atom_stays_uncertain_without_an_arbiter():
    pending = [classify_number_atom("7", "I tried it 7 different ways and it finally worked well.")]
    resolved = resolve_uncertain_with_arbiter(
        pending, source_text="...", kept_text="", arbiter=None,
    )
    assert resolved[0].importance == UNCERTAIN
    assert blocks_freeze(resolved[0].importance) is True


def test_uncertain_atom_stays_uncertain_on_arbiter_exception():
    class _BrokenArbiter:
        def classify_atom(self, *_args):
            raise RuntimeError("provider down")

    pending = [classify_number_atom("7", "I tried it 7 different ways and it finally worked well.")]
    resolved = resolve_uncertain_with_arbiter(
        pending, source_text="...", kept_text="", arbiter=_BrokenArbiter(),
    )
    assert resolved[0].importance == UNCERTAIN


def test_a_critical_deterministic_verdict_is_never_second_guessed_by_an_arbiter():
    pending = [classify_number_atom("15", "Only 15% of users ever finish the full course.")]
    arbiter = _FakeArbiter((CONTEXTUAL, 0.99, "irrelevant -- should not be consulted"))
    resolved = resolve_uncertain_with_arbiter(
        pending, source_text="...", kept_text="", arbiter=arbiter,
    )
    assert resolved[0].importance == CRITICAL
    assert resolved[0].resolved_by == "deterministic"


def test_malformed_arbiter_verdict_leaves_atom_uncertain():
    pending = [classify_number_atom("7", "I tried it 7 different ways and it finally worked well.")]
    arbiter = _FakeArbiter(("MAYBE", 0.5, "not a real class"))
    resolved = resolve_uncertain_with_arbiter(
        pending, source_text="...", kept_text="", arbiter=arbiter,
    )
    assert resolved[0].importance == UNCERTAIN
