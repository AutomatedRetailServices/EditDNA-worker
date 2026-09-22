"""D-147 (Gate 6 correction, real RAW #118 audit): `causal_order_validator.py`
connector-strength correction.

D-146 added direct diagnosis-reveal phrasing ("I was diagnosed with...",
"me diagnosticaron con...") to `_STRONG_DEPENDENCY_CONNECTORS` -- classified
as deterministic evidence needing no arbiter, exactly like "therefore" or
"eso confirmó". A real audit against the actual RAW #118 artifact found this
was unsafe in general, not just for that one run: unlike the original STRONG
connectors, which are grammatically anaphoric (they only parse with SOME
earlier referent), a direct diagnosis statement is a complete, self-
contained sentence that reads fine with no antecedent. Scored STRONG, it
would deterministically manufacture a false causal dependency on WHATEVER
same-source clip happens to sit within `_MAX_SOURCE_GAP_SEC` beforehand --
related or not -- and block Freeze over a non-existent dependency.

This file proves: (1) the false-positive this correction closes, with a
diagnosis phrase sitting near a genuinely unrelated clip, must never create
a deterministic dependency; (2) real detection is still possible for a
genuinely dependent pair, but only through the bounded `CausalOrderArbiter`
escalation path every other WEAK connector already uses -- never
deterministically. Generic, non-medical-disease fixtures throughout.
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.causal_order_validator import find_causal_order_breaks
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import CAUSAL_ORDER_BREAK, review


def clip(clip_id, start, end, text, *, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=True,
    )


def draft(*, selected):
    groups = [{"group_id": f"g_{c.clip_id}", "ranked": [{"clip_id": c.clip_id, "score": 0.9, "reason": "x"}]} for c in selected]
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=(),
        diagnostics={
            "take_judge_groups": groups,
            "final_story_coherence_validation": {"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
            "hybrid_editorial_chunks": [],
        },
    )


class _FakeArbiter:
    def __init__(self, verdict):
        self._verdict = verdict

    def check_dependency(self, required_text, dependent_text):
        return self._verdict


def test_diagnosis_phrase_near_an_unrelated_clip_never_creates_a_strong_dependency():
    """The core regression: a diagnosis-reveal phrase sitting right after a
    COMPLETELY UNRELATED clip in the same source must never be treated as
    dependent on it -- no arbiter supplied, no dependency, no block."""
    unrelated = clip("unrelated", 0.0, 5.0, "we also spent some time talking about the weather that day")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition not long after")
    d = draft(selected=(diagnosis, unrelated))  # order is irrelevant -- must never block either way

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
    assert review(plan).status == "PASS"


def test_diagnosis_phrase_near_an_unrelated_clip_stays_unblocked_even_with_a_denying_arbiter():
    """Same shape, this time WITH an arbiter available that correctly denies
    the dependency -- proves the escalation path itself is sound, not just
    "no arbiter means no evidence"."""
    unrelated = clip("unrelated", 0.0, 5.0, "we also spent some time talking about the weather that day")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition not long after")
    d = draft(selected=(diagnosis, unrelated))

    plan = build_canonical_edit_plan(d)
    arbiter = _FakeArbiter((False, 0.9, "the two clips are unrelated"))
    assert find_causal_order_breaks(plan, arbiter=arbiter) == ()
    assert review(plan, causal_order_arbiter=arbiter).status == "PASS"


def test_diagnosis_phrase_dependency_is_still_detectable_via_a_confirming_arbiter():
    """Real detection remains possible -- through the bounded arbiter path,
    exactly like every other weak connector -- when the pair really is
    dependent and the arbiter confirms it."""
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis, finding))  # diagnosis rendered before its own finding

    plan = build_canonical_edit_plan(d)
    arbiter = _FakeArbiter((True, 0.88, "the diagnosis depends on the finding that precedes it"))
    breaks = find_causal_order_breaks(plan, arbiter=arbiter)

    assert len(breaks) == 1
    assert breaks[0].resolved_by == "semantic_arbiter"
    assert breaks[0].required_clip_id == "finding"
    assert breaks[0].dependent_clip_id == "diagnosis"

    result = review(plan, causal_order_arbiter=arbiter)
    assert result.status == "FAIL"
    assert any(f.kind == CAUSAL_ORDER_BREAK for f in result.findings)


def test_diagnosis_phrase_arbiter_exception_fails_open_same_as_every_other_weak_connector():
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis, finding))

    class _BrokenArbiter:
        def check_dependency(self, required_text, dependent_text):
            raise RuntimeError("arbiter unavailable")

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan, arbiter=_BrokenArbiter()) == ()


def test_all_migrated_direct_diagnosis_phrases_are_weak_not_strong():
    """Structural guard: every phrase D-146 added must live in the WEAK
    lexicon, never the STRONG one -- protects against a future edit
    accidentally re-promoting them without re-deriving the anaphoric
    justification the STRONG tier requires."""
    from cutsell_worker.causal_order_validator import (
        _STRONG_DEPENDENCY_CONNECTORS, _WEAK_DEPENDENCY_CONNECTORS,
    )
    migrated = (
        "i was diagnosed with", "they diagnosed me with", "the diagnosis was",
        "confirmed i had", "confirmed that i had", "turned out to be",
        "the results showed", "the test results showed",
        "me diagnosticaron con", "me dijeron que tenía", "me dijeron que tenia",
        "el diagnóstico fue", "el diagnostico fue", "resultó ser", "resulto ser",
        "los resultados mostraron", "el resultado fue",
    )
    for phrase in migrated:
        assert phrase in _WEAK_DEPENDENCY_CONNECTORS, phrase
        assert phrase not in _STRONG_DEPENDENCY_CONNECTORS, phrase
