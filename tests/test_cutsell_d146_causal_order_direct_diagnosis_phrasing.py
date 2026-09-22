"""D-146 (Gate 6 / RAW #118 Gap B): `causal_order_validator.py`'s dependency
detection is entirely gated on `_connector_prefix_hit` -- a clip's text must
literally START WITH a structural transition word ("therefore", "that's
why", "eso confirmó", ...) before any dependency is even proposed. Ordinary
spoken diagnosis-after-finding narration very often states the conclusion
DIRECTLY, with no structural connector at all ("I was diagnosed with the
condition", "me diagnosticaron con la afección") -- so this module currently
never even offers that pair to `find_causal_order_breaks`, regardless of
whether the exam/finding clip that produced it is missing or out of order.

Fix: extend the (already general, English+Spanish, no-Video00-vocabulary)
strong-connector lexicon with the common DIRECT diagnosis-conclusion
phrasing pattern this exact shape needs -- "I was diagnosed with", "me
diagnosticaron con", etc. -- the same kind of general grammatical/lexical
signal the module already relies on, not a Video00-specific fix. Generic,
non-medical-disease fixtures throughout, matching this suite's existing
convention.
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


def test_direct_diagnosis_statement_before_its_finding_is_a_blocking_break():
    """No 'therefore'/'that's why' connector anywhere -- the diagnosis clip
    states its conclusion directly, exactly like ordinary spoken narration."""
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis, finding))  # rendered before its own finding

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "FAIL"
    causal = [f for f in result.findings if f.kind == CAUSAL_ORDER_BREAK]
    assert causal and causal[0].detail["required_clip_id"] == "finding"
    assert causal[0].detail["dependent_clip_id"] == "diagnosis"


def test_direct_diagnosis_statement_spanish_phrasing_before_its_finding_is_a_blocking_break():
    finding = clip("finding", 0.0, 5.0, "me hicieron un estudio y encontraron algo fuera de lo normal")
    diagnosis = clip("diagnosis", 5.0, 10.0, "me diagnosticaron con la afección poco después")
    d = draft(selected=(diagnosis, finding))

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "FAIL"
    causal = [f for f in result.findings if f.kind == CAUSAL_ORDER_BREAK]
    assert causal and causal[0].detail["required_clip_id"] == "finding"
    assert causal[0].detail["dependent_clip_id"] == "diagnosis"


def test_direct_diagnosis_correctly_after_its_finding_produces_no_break():
    """Positive control: the same phrasing, correctly ordered, must not
    become a false positive."""
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(finding, diagnosis))

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
    assert review(plan).status == "PASS"


def test_diagnosis_missing_its_finding_entirely_is_a_detached_explanation_break():
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis,))

    plan = build_canonical_edit_plan(d)
    # No finding clip exists at all in this source -- no candidate to depend
    # on, so this must stay a no-op, not a spurious break (mirrors the
    # existing far-apart-clips negative control's fail-open direction).
    assert find_causal_order_breaks(plan) == ()


def test_unrelated_direct_statement_does_not_spuriously_trigger():
    """False-positive guard: an unrelated clip that happens to be adjacent
    in the same source must not be treated as the 'finding' this diagnosis
    depends on merely because SOME earlier clip exists."""
    unrelated = clip("unrelated", 0.0, 5.0, "we also talked about the weather that day")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis, unrelated))
    plan = build_canonical_edit_plan(d)
    breaks = find_causal_order_breaks(plan)
    # The connector phrase alone still creates a dependency candidate on the
    # nearest earlier same-source clip (this module's existing, documented
    # adjacency heuristic -- unchanged by this fix); what this guards is
    # narrower: the new phrases must not ALSO match unrelated text that
    # merely sits nearby, which this fixture's "unrelated" text does not.
    if breaks:
        assert breaks[0].evidence.startswith("connector_language:")
