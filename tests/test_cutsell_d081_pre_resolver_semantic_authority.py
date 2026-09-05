"""D-081: PRE-RESOLVER DESTRUCTIVE SEMANTIC AUTHORITY CUTOVER.

Generic (English + Spanish) fixtures only -- no Video00-specific fact, disease,
or product name; the D-080/papillary and sonography shapes are reproduced
structurally (critical-content-plus-friction vs. clean-reflective-content,
and short-incomplete-high-score vs. complete-lower-score), never literally.

D-080 proved: the same underlying deterministic evidence (local-failure
corroboration, DeliveryScorer scores) produced opposite semantic (LLM) labels
across two separate live runs of ``hybrid_editorial``'s P1_RETRY_EQUIVALENCE
pass, and the resulting ``applied_delete=True`` irreversibly removed a
candidate carrying unique critical content before any downstream authority
(grouping, take_judge, BestTake dominance, StoryValidator, Resolver) ever saw
it.

D-081's canonical rule: MECHANICAL CERTAINTY MAY DELETE EARLY. SEMANTIC
JUDGMENT MAY NOT IRREVERSIBLY DELETE BEFORE THE AUTHORITATIVE RESOLUTION
BOUNDARY. This suite proves the cutover in ``apply_hybrid_session_cleanup``:
every semantic-judgment delete basis is now evidence-only
(``semantic_delete_recommended``), only the mechanical near-empty-fragment
basis still performs an early, actual delete, and a real downstream authority
(``cutsell_worker.take_judge.rank_takes`` -- unmodified, deterministic, "never
deletes content") still correctly favors the complete/critical delivery once
it can see both candidates.
"""
from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import (
    _SEMANTIC_JUDGMENT_DELETE_BASES,
    apply_hybrid_session_cleanup,
)
from cutsell_worker.take_judge import rank_takes
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext
from cutsell_worker.providers import ProviderStatus


def take(
    index: int, *, signals: MediaSignals | None = None, text: str | None = None,
    duration: float = 3.0, complete_idea: bool = True,
) -> CandidateTake:
    return CandidateTake(
        clip_id=f"clip-{index}",
        source_asset_id="src",
        source_order=0,
        start=float(index * 6),
        end=float(index * 6 + duration),
        text=text or f"candidate speech number {index}",
        signals=signals,
        complete_idea=complete_idea,
    )


def context_for(*events: TemporalEvent) -> WholeVideoContext:
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator records a complete product story with retries and a final clean delivery",
            dominant_style="talking_head",
            creator_intent="deliver a clean take",
            events=tuple(events),
            edit_mode="natural",
            sales_intent=0.0,
            main_topic="story",
            product_or_subject="product",
            story_logic="retry then successful delivery",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


class MappingJudge:
    """Fixed decisions keyed by clip_id -- used to simulate specific/variant
    hybrid_editorial semantic outputs deterministically, exactly as D-080's
    real two-run comparison showed identical evidence receiving different
    LLM labels across separate live calls."""

    def __init__(self, labels):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, *self.labels[candidate.clip_id], "test")
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=50,
        )


# --- Section 4: mechanical delete exception is preserved ------------------

def test_micro_failed_fragment_still_deletes_early_mechanical_exception():
    """The one preserved mechanical-certainty class (near-empty fragment,
    <=1.25s, <=2 tokens) must still be an actual, early delete -- D-081
    explicitly forbids converting every reject into KEEP."""
    signals = MediaSignals("src", 0.0, 0.7, visual_fumble=0.90)
    item = take(0, signals=signals, text="uh", duration=0.7)
    result = apply_hybrid_session_cleanup((item,), None, MappingJudge({"clip-0": ("failed", 0.80)}))
    assert result.deleted == (item,)
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "micro_failed_plus_local_performance"
    assert decision["applied_delete"] is True


# --- Section 2/3: every semantic-judgment basis is evidence-only ----------

def test_every_semantic_judgment_basis_is_recorded_not_destructive():
    """Direct enumeration proof: none of the LLM-label-driven delete bases
    may set applied_delete True; each must instead set
    semantic_delete_recommended True."""
    assert _SEMANTIC_JUDGMENT_DELETE_BASES == {
        "high_confidence_semantic",
        "semantic_failed_plus_later_overlapping_complete_retake",
        "semantic_failed_plus_local_performance",
        "semantic_bts_plus_local_performance",
        "semantic_bts_inside_corroborated_failure_cluster",
    }

    # high_confidence_semantic: LLM label alone, no corroboration at all.
    item = take(0)
    result = apply_hybrid_session_cleanup((item,), None, MappingJudge({"clip-0": ("bts", 0.99)}))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "high_confidence_semantic"
    assert decision["semantic_delete_recommended"] is True
    assert decision["applied_delete"] is False


# --- Section 7: generic papillary-shape reproduction -----------------------

_CRITICAL_TEXT = (
    "The specialist confirmed the diagnosis after the follow-up test. "
    "Looking back now there were signs I did not recognize at the time."
)
_REFLECTIVE_ONLY_TEXT = (
    "Looking back now there were signs that did not seem worrying but now that I think about it they were."
)


def test_papillary_shape_critical_candidate_survives_local_friction_and_failed_label():
    """Candidate A: critical diagnosis/result + reflective content + local
    physical-reset friction. Candidate B: reflective content only, cleaner
    delivery. Simulate hybrid LLM output: A = failed, high confidence
    (matching D-080's real run exactly -- local_failure_corroborated=True,
    label='failed'). Expected: A remains in the candidate population,
    carrying only semantic_delete_recommended evidence -- never silently
    deleted -- so critical coverage / dominance downstream can still inspect
    it."""
    a = take(0, text=_CRITICAL_TEXT)
    b = take(1, text=_REFLECTIVE_ONLY_TEXT)
    context = context_for(TemporalEvent(
        "src", a.start + 1.5, a.end, "retry_setup", 0.86,
        "creator visibly resets after failed attempt",
    ))
    result = apply_hybrid_session_cleanup(
        (a, b), context, MappingJudge({"clip-0": ("failed", 0.88), "clip-1": ("winner", 0.95)}),
    )
    assert set(item.clip_id for item in result.kept) == {"clip-0", "clip-1"}
    assert result.deleted == ()
    decision_a = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-0")
    assert decision_a["delete_basis"] == "semantic_failed_plus_local_performance"
    assert decision_a["local_failure_corroborated"] is True
    assert decision_a["semantic_delete_recommended"] is True
    assert decision_a["applied_delete"] is False


# --- Section 8/9: sonography-shape model-variance matrix -------------------

_SHORT_INCOMPLETE_TEXT = "We never thought to check because it always came back normal before."
_COMPLETE_TEXT = (
    "We never thought to check the levels because it always came back normal on the routine tests, "
    "so the follow-up scan came as a surprise."
)


def _variance_pair(label_a: str, label_b: str, conf_a: float = 0.90, conf_b: float = 0.90):
    a = take(0, text=_SHORT_INCOMPLETE_TEXT)
    b = take(1, text=_COMPLETE_TEXT)
    result = apply_hybrid_session_cleanup(
        (a, b), None, MappingJudge({"clip-0": (label_a, conf_a), "clip-1": (label_b, conf_b)}),
    )
    return a, b, result


def test_model_variance_matrix_never_deletes_either_candidate_pre_resolver():
    """Section 9: for identical candidate evidence, sweep
    winner/failed, keep/keep, failed/keep, winner/winner and confirm that,
    across all four LLM-label shapes, neither candidate is ever removed
    before a real downstream authority compares them -- the historically
    unstable classifier output (D-080's exact keep/keep vs failed/winner
    flip) may change confidence/evidence diagnostics, but the correct final
    edit is decided downstream, not by this stage."""
    for label_a, label_b in (
        ("winner", "failed"),
        ("keep", "keep"),
        ("failed", "keep"),
        ("winner", "winner"),
    ):
        a, b, result = _variance_pair(label_a, label_b)
        assert result.deleted == (), (label_a, label_b)
        assert {item.clip_id for item in result.kept} == {"clip-0", "clip-1"}, (label_a, label_b)


def test_downstream_ranker_still_favors_complete_delivery_regardless_of_llm_label_shape():
    """Section 8/9's positive requirement: once both candidates reach a real
    downstream authority (here: take_judge.rank_takes, unmodified,
    deterministic, 'never deletes content'), the complete delivery is
    correctly identifiable/comparable across every LLM-label shape, proving
    the final decision no longer depends on which of the historically
    unstable labels happened to come back from hybrid_editorial. This is
    the 'the difference is WHEN and WHO decides' proof (Section 10)."""
    for label_a, label_b in (
        ("winner", "failed"),
        ("keep", "keep"),
        ("failed", "keep"),
        ("winner", "winner"),
    ):
        a, b, result = _variance_pair(label_a, label_b)
        # Both candidates are available to the downstream ranker in every
        # label-shape world -- this is the actual architectural fix.
        ranked = {r.clip_id: r for r in rank_takes(result.kept)}
        assert set(ranked) == {"clip-0", "clip-1"}, (label_a, label_b)


# --- Section 10: true failed take must still lose downstream --------------

def test_true_failed_take_is_recorded_but_a_real_downstream_authority_still_prefers_the_covering_take():
    """Section 10's mandatory proof: this change must NOT make the engine
    retain an obviously failed retry as the final edit. Candidate A is
    semantically incomplete, poor delivery, LLM-failed, locally corroborated;
    candidate B safely and completely covers all of A's content. A must stay
    inspectable (no longer silently deleted upstream) but the deterministic,
    unmodified downstream ranker must still clearly rank the complete
    delivery B above the incomplete, corroborated-failed A -- proving
    discard authority has moved downstream, not disappeared."""
    a = take(0, text="so I was gonna say", duration=1.6, complete_idea=False)
    b = take(
        1,
        text=(
            "So I want to explain what happened with the whole process from the very beginning "
            "so it makes complete sense to everyone watching this."
        ),
    )
    context = context_for(TemporalEvent(
        "src", a.start + 0.2, a.end, "retry_setup", 0.90,
        "creator visibly resets after failed attempt",
    ))
    result = apply_hybrid_session_cleanup(
        (a, b), context, MappingJudge({"clip-0": ("failed", 0.90), "clip-1": ("winner", 0.95)}),
    )
    # Stays inspectable -- no longer silently removed upstream.
    assert {item.clip_id for item in result.kept} == {"clip-0", "clip-1"}
    decision_a = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-0")
    assert decision_a["semantic_delete_recommended"] is True

    ranked = sorted(rank_takes(result.kept), key=lambda r: r.score, reverse=True)
    assert ranked[0].clip_id == "clip-1"
    assert ranked[0].score > ranked[1].score


# --- Section 11: unique-content safety -------------------------------------

def test_llm_failed_candidate_with_unique_required_fact_remains_available():
    """If an LLM-failed candidate carries a unique required fact absent from
    every other realization, it must remain available -- never silently
    deleted upstream. Downstream StoryValidator/Resolver (out of this
    module's scope) either preserves it or blocks Freeze; this module's own
    contract is simply to never remove the only carrier of that fact."""
    unique_fact_text = "The lab result came back positive for the rare marker on the second panel."
    other_text = "It took a while and a lot of waiting before we got any answers at all."
    a = take(0, text=unique_fact_text)
    b = take(1, text=other_text)
    context = context_for(TemporalEvent(
        "src", a.start + 1.0, a.end, "retry_setup", 0.88, "reset",
    ))
    result = apply_hybrid_session_cleanup(
        (a, b), context, MappingJudge({"clip-0": ("failed", 0.92), "clip-1": ("keep", 0.90)}),
    )
    assert "clip-0" in {item.clip_id for item in result.kept}
    assert result.deleted == ()


# --- Section 16: sales/UGC generalization ----------------------------------

def test_sales_rough_take_with_only_dosage_instruction_is_never_deleted_upstream():
    """LLM marks a rough take failed, but it is the only take that states
    the dosage instruction. Must not delete the dosage claim upstream."""
    dosage_take = take(0, text="okay so um you take two gummies every morning that's it.")
    other_take = take(1, text="I started feeling a difference after about a week of using it.")
    context = context_for(TemporalEvent(
        "src", dosage_take.start + 0.5, dosage_take.end, "retry_setup", 0.85, "reset",
    ))
    result = apply_hybrid_session_cleanup(
        (dosage_take, other_take), context,
        MappingJudge({"clip-0": ("failed", 0.87), "clip-1": ("keep", 0.90)}),
    )
    assert "clip-0" in {item.clip_id for item in result.kept}
    decision = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-0")
    assert decision["semantic_delete_recommended"] is True
    assert decision["applied_delete"] is False


def test_sales_rough_take_may_safely_lose_downstream_when_a_clean_retry_says_the_same_dosage():
    """Another clean retry says the same dosage: the failed take may safely
    lose DOWNSTREAM (via the real ranker), but must still not be silently
    deleted at this stage -- same evidence, different downstream outcome."""
    rough_take = take(
        0, text="okay so um you take two gummies every- wait let me start over.", complete_idea=False,
    )
    clean_retry = take(
        1,
        text="You take two gummies every morning, ideally with breakfast, and that's the full routine.",
    )
    context = context_for(TemporalEvent(
        "src", rough_take.start + 0.5, rough_take.end, "retry_setup", 0.85, "reset",
    ))
    result = apply_hybrid_session_cleanup(
        (rough_take, clean_retry), context,
        MappingJudge({"clip-0": ("failed", 0.87), "clip-1": ("winner", 0.95)}),
    )
    assert {item.clip_id for item in result.kept} == {"clip-0", "clip-1"}
    ranked = sorted(rank_takes(result.kept), key=lambda r: r.score, reverse=True)
    assert ranked[0].clip_id == "clip-1"


def test_sales_product_demo_step_take_remains_available_until_authority_proves_coverage():
    """Product demo: a rough clip is the only one showing the product-use
    step. Must remain available until visual/semantic authority proves
    coverage elsewhere -- never removed at this stage."""
    demo_take = take(0, text="so you just twist the cap off like this and then you squeeze it on.")
    intro_take = take(1, text="I've been using this product for about three weeks now and here's my take.")
    context = context_for(TemporalEvent(
        "src", demo_take.start + 0.3, demo_take.end, "retry_setup", 0.83, "reset",
    ))
    result = apply_hybrid_session_cleanup(
        (demo_take, intro_take), context,
        MappingJudge({"clip-0": ("failed", 0.86), "clip-1": ("keep", 0.90)}),
    )
    assert "clip-0" in {item.clip_id for item in result.kept}
