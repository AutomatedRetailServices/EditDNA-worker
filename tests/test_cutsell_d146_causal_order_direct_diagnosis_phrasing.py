"""D-146 (Gate 6 / RAW #118 Gap B), corrected by D-147.

`causal_order_validator.py`'s dependency detection is gated on
`_connector_prefix_hit` -- a clip's text must literally START WITH a
connector phrase before any dependency is even proposed. Ordinary spoken
diagnosis-after-finding narration very often states the conclusion
DIRECTLY, with no structural transition word at all ("I was diagnosed with
the condition", "me diagnosticaron con la afección"). D-146 recognized this
real coverage gap and added the direct diagnosis-reveal phrasing pattern to
the connector lexicon.

D-146's ORIGINAL mistake (found by a real RAW #118 audit, corrected in
D-147): it classified these phrases as STRONG (deterministic, no arbiter
needed). Unlike "therefore"/"eso confirmó" -- which are grammatically
anaphoric and only parse with SOME preceding referent, making the nearest
earlier same-source clip a safe deterministic target -- "I was diagnosed
with X" is a complete, self-contained statement that reads fine with no
antecedent at all. Scored STRONG, it deterministically manufactured a false
causal dependency on whatever same-source clip happened to sit nearby,
related or not. D-147 moved these phrases to WEAK (arbiter-gated, exactly
like every other ambiguous connector) -- see
`tests/test_cutsell_d147_causal_connector_strength_correction.py` for the
false-positive regression this corrects and the confirming-arbiter path
that makes real detection still possible once an arbiter exists.

This file now documents the CORRECTED, honest behavior: without a live
arbiter (none is wired into production today, an existing, already-
documented gap), a bare direct-diagnosis phrase is insufficient
deterministic evidence on its own and must never block Freeze -- exactly
like every other weak connector. Generic, non-medical-disease fixtures
throughout, matching this suite's existing convention.
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.causal_order_validator import find_causal_order_breaks
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import review


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


def test_direct_diagnosis_statement_alone_never_blocks_without_an_arbiter():
    """No 'therefore'/'that's why' connector anywhere -- the diagnosis clip
    states its conclusion directly. Without a confirming arbiter, this is
    insufficient deterministic evidence (WEAK, not STRONG) and must never
    block Freeze on its own, even when the finding it might depend on is
    genuinely misordered."""
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis, finding))  # rendered before its own finding

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
    assert review(plan).status == "PASS"


def test_direct_diagnosis_statement_spanish_phrasing_alone_never_blocks_without_an_arbiter():
    finding = clip("finding", 0.0, 5.0, "me hicieron un estudio y encontraron algo fuera de lo normal")
    diagnosis = clip("diagnosis", 5.0, 10.0, "me diagnosticaron con la afección poco después")
    d = draft(selected=(diagnosis, finding))

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
    assert review(plan).status == "PASS"


def test_direct_diagnosis_correctly_after_its_finding_produces_no_break():
    """Positive control: the same phrasing, correctly ordered, must not
    become a false positive either."""
    finding = clip("finding", 0.0, 5.0, "we ran a scan that found something unusual in the sample")
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(finding, diagnosis))

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
    assert review(plan).status == "PASS"


def test_diagnosis_missing_its_finding_entirely_is_still_a_no_op_without_an_arbiter():
    diagnosis = clip("diagnosis", 5.0, 10.0, "I was diagnosed with the condition after that")
    d = draft(selected=(diagnosis,))

    plan = build_canonical_edit_plan(d)
    assert find_causal_order_breaks(plan) == ()
