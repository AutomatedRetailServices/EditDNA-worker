"""General causal/story order validation (D-027).

Covers the required CleanCutBench-style shapes from the canonical directive:
valid chronology, inverted cause/effect, continuation before parent,
diagnosis before discovery, independent ideas that may safely reorder, CTA
ending preservation, ambiguous order requiring the arbiter, and
false-positive prevention (a weak/generic connector alone must never block).

No Video00 fact, disease, phrase, or timestamp appears anywhere below --
every fixture uses generic, made-up subject matter to prove the mechanism is
general (source chronology + connector language), not content-specific.
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


def draft(*, selected, discarded=(), take_judge_groups=None, coherence=None):
    groups = take_judge_groups
    if groups is None:
        groups = [{"group_id": f"g_{c.clip_id}", "ranked": [{"clip_id": c.clip_id, "score": 0.9, "reason": "x"}]} for c in selected]
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={
            "take_judge_groups": groups,
            "final_story_coherence_validation": coherence or {"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
            "hybrid_editorial_chunks": [],
        },
    )


def test_valid_chronology_produces_no_break():
    setup = clip("setup", 0.0, 5.0, "we ran the test on the sample")
    result_clip = clip("result", 5.0, 10.0, "and that confirmed the reading was accurate")
    d = draft(selected=(setup, result_clip))

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)

    assert breaks == ()
    assert review(plan).status == "PASS"


def test_inverted_cause_and_effect_is_a_blocking_break():
    setup = clip("setup", 0.0, 5.0, "we ran the test on the sample")
    result_clip = clip("result", 5.0, 10.0, "and that confirmed the reading was accurate")
    # Final composed order puts the consequence BEFORE its cause.
    d = draft(selected=(result_clip, setup))

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)

    assert len(breaks) == 1
    assert breaks[0].required_clip_id == "setup"
    assert breaks[0].dependent_clip_id == "result"
    assert breaks[0].resolved_by == "deterministic_connector_language"

    result = review(plan)
    assert result.status == "FAIL"
    causal = [f for f in result.findings if f.kind == CAUSAL_ORDER_BREAK]
    assert len(causal) == 1
    assert causal[0].clip_ids == ("setup", "result")
    assert causal[0].blocking is True


def test_continuation_placed_before_its_parent_is_a_blocking_break():
    parent = clip("parent", 0.0, 4.0, "here is how we set up the whole approach")
    continuation = clip("continuation", 4.0, 8.0, "and that's how we finished the rest of the process")
    d = draft(selected=(continuation, parent))  # continuation rendered first -- wrong

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "FAIL"
    assert any(f.kind == CAUSAL_ORDER_BREAK for f in result.findings)


def test_diagnosis_before_discovery_is_a_blocking_break():
    # Generic stand-in for "diagnosis before the test that produced it" --
    # deliberately not medical vocabulary, to prove the mechanism is general.
    discovery = clip("discovery", 0.0, 5.0, "the inspection turned up an anomaly in the part")
    diagnosis = clip("diagnosis", 5.0, 10.0, "therefore the part was flagged as defective")
    d = draft(selected=(diagnosis, discovery))  # diagnosis rendered before its discovery

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "FAIL"
    causal = [f for f in result.findings if f.kind == CAUSAL_ORDER_BREAK]
    assert causal[0].detail["required_clip_id"] == "discovery"
    assert causal[0].detail["dependent_clip_id"] == "diagnosis"


def test_independent_ideas_with_no_connector_language_may_safely_reorder():
    # Two genuinely independent ideas, deliberately reordered relative to
    # source chronology (legitimate Composer pacing) -- neither opens with a
    # dependency connector, so no relationship is inferred at all.
    idea_one = clip("idea_one", 0.0, 5.0, "our product ships in three colors")
    idea_two = clip("idea_two", 20.0, 25.0, "our product also comes with a two year warranty")
    d = draft(selected=(idea_two, idea_one))  # reordered for pacing -- must NOT be flagged

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "PASS"
    assert not any(f.kind == CAUSAL_ORDER_BREAK for f in result.findings)


def test_cta_ending_is_preserved_when_correctly_placed_after_its_required_body():
    body = clip("body", 0.0, 5.0, "the plan covers everything you need for setup")
    cta = clip("cta", 5.0, 8.0, "so go ahead and get started today")
    d = draft(selected=(body, cta))  # CTA correctly follows its required body

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "PASS"


def test_weak_connector_alone_is_not_flagged_without_an_arbiter_false_positive_prevention():
    # "so " is a WEAK/generic connector -- on its own, without a confirming
    # arbiter, it must never block Freeze. False-positive prevention is the
    # whole point of the strong/weak confidence split.
    body = clip("body", 0.0, 5.0, "the plan covers everything you need for setup")
    cta = clip("cta", 5.0, 8.0, "so go ahead and get started today")
    d = draft(selected=(cta, body))  # order actually inverted, but connector is weak

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)  # no arbiter supplied

    assert breaks == ()
    assert review(plan).status == "PASS"


class _FakeArbiter:
    def __init__(self, verdict):
        self._verdict = verdict

    def check_dependency(self, required_text, dependent_text):
        return self._verdict


def test_ambiguous_weak_connector_is_resolved_by_a_confirming_arbiter():
    body = clip("body", 0.0, 5.0, "the plan covers everything you need for setup")
    cta = clip("cta", 5.0, 8.0, "so go ahead and get started today")
    d = draft(selected=(cta, body))  # inverted, weak connector

    plan = build_canonical_edit_plan(d)
    arbiter = _FakeArbiter((True, 0.85, "cta depends on the body it follows"))
    breaks = find_causal_order_breaks(plan, arbiter=arbiter)

    assert len(breaks) == 1
    assert breaks[0].resolved_by == "semantic_arbiter"

    result = review(plan, causal_order_arbiter=arbiter)
    assert result.status == "FAIL"
    assert any(f.kind == CAUSAL_ORDER_BREAK for f in result.findings)


def test_ambiguous_weak_connector_dropped_when_arbiter_denies_dependency():
    body = clip("body", 0.0, 5.0, "the plan covers everything you need for setup")
    cta = clip("cta", 5.0, 8.0, "so go ahead and get started today")
    d = draft(selected=(cta, body))

    plan = build_canonical_edit_plan(d)
    arbiter = _FakeArbiter((False, 0.9, "not actually dependent"))
    breaks = find_causal_order_breaks(plan, arbiter=arbiter)

    assert breaks == ()
    assert review(plan, causal_order_arbiter=arbiter).status == "PASS"


def test_a_strong_connector_hit_is_never_second_guessed_by_a_denying_arbiter():
    setup = clip("setup", 0.0, 5.0, "we ran the test on the sample")
    result_clip = clip("result", 5.0, 10.0, "and that confirmed the reading was accurate")
    d = draft(selected=(result_clip, setup))

    plan = build_canonical_edit_plan(d)
    arbiter = _FakeArbiter((False, 0.9, "irrelevant -- should not be consulted"))
    breaks = find_causal_order_breaks(plan, arbiter=arbiter)

    assert len(breaks) == 1
    assert breaks[0].resolved_by == "deterministic_connector_language"


def test_dependent_clip_with_required_context_entirely_discarded_is_a_detached_explanation_break():
    setup = clip("setup", 0.0, 5.0, "we ran the test on the sample")
    result_clip = clip("result", 5.0, 10.0, "and that confirmed the reading was accurate")
    # setup was discarded entirely (never even in selected) -- the
    # explanation is left detached from the fact it explains.
    d = draft(selected=(result_clip,), discarded=(setup,))

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)

    assert len(breaks) == 1
    assert breaks[0].dependent_clip_id == "result"
    assert breaks[0].required_clip_id == "setup"


def test_arbiter_exception_is_treated_as_not_available_and_drops_the_ambiguous_hit():
    body = clip("body", 0.0, 5.0, "the plan covers everything you need for setup")
    cta = clip("cta", 5.0, 8.0, "so go ahead and get started today")
    d = draft(selected=(cta, body))

    class _BrokenArbiter:
        def check_dependency(self, *_args):
            raise RuntimeError("provider unavailable")

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan, arbiter=_BrokenArbiter())

    assert breaks == ()


def test_far_apart_clips_in_the_same_source_are_not_treated_as_dependent():
    # Same source, but far beyond the continuous-take gap tolerance --
    # a shared connector word this far apart is not reliable adjacency
    # evidence, so no dependency is inferred at all.
    setup = clip("setup", 0.0, 5.0, "we ran the test on the sample")
    result_clip = clip("result", 500.0, 505.0, "and that confirmed the reading was accurate")
    d = draft(selected=(result_clip, setup))

    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)

    assert breaks == ()
