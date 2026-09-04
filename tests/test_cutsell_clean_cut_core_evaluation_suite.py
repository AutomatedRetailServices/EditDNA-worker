"""Clean Cut Core V1 general evaluation suite (a.k.a. CleanCutBench).

Video-agnostic fixtures growing toward the canonical directive's target scale
(100-300+ cases; this file is an honest subset, not that scale yet). Covers:
exact retry, paraphrased retry, two good takes expressing the same idea,
false start -> clean retry, incomplete -> complete retry, long bad take vs
concise complete take, composite required, continuation that must not
collapse, similar vocabulary but distinct ideas, visual fumble with
semantically valid transcript, unique-fact preservation, CTA/story-ending
preservation, multiple retries with progressively better delivery, same idea
expressed with substantially different wording, contradictory factual
retries (hard freeze-blocking invariant), self-correction within one
continuous take, retry disguised as continuation, continuation disguised as
retry, composite-forbidden-because-complete-exists, orphan fragment,
duplicate semantic beat restated far apart in time (documents a known
current limitation, not yet a passing capability -- see its own docstring),
uncertain semantic case (arbiter unavailable fails open to preserve, never
guesses), multilingual paraphrased retry, causal/story order preserved
across independent (non-retry) beats, numeric correction across two takes
(conservative freeze-block by design, not an auto-pick), and two general
content-loss cases added after RAW 33345946000 (a retry winner missing the
loser's unique numeric fact / missing the loser's unrelated symptom beat) --
see the ledger comment before fixture 26 for which requested content-loss
categories are instead covered at the final_story_coherence_validation unit
level, and why. Fixtures 28-29 (D-026/D-027) extend the exercised chain one
stage further -- CanonicalEditPlan -> [repair_loop|FinalEditReviewer] -- to
prove the automatic targeted repair loop and the general causal/story order
validator integrate correctly with the REAL take-grouping/idea-equivalence/
take-judge/coherence chain above, not just synthetic drafts built by hand
(their own exhaustive category coverage lives in tests/test_cutsell_repair_
loop.py and tests/test_cutsell_causal_order_validator.py respectively -- these
two fixtures are integration proof, not a duplicate of that coverage).
PostRenderWatchListenQC's real media checks (D-028) are deliberately NOT
exercised here: they operate on decoded/probed MEDIA FILES, orthogonal to
this suite's take-grouping/text-level chain -- their CleanCutBench-style
required-category coverage is tests/test_cutsell_post_render_media_qc.py's
own synthetic-fixture suite, in full.

Explicitly out of this file's scope, because they belong to other
already-existing pipeline stages this suite does not exercise (clean_cut.py's
pre-grouping garbage removal, Boundary's physical timing): BTS/meta-recording
material and intentional-pause/dead-air pacing. Both already have their own
existing test coverage elsewhere in this repo.

Each fixture has an explicit oracle (kept clip_ids, discarded clip_ids, and
for a couple of cases an explicit note on composite/continuation
expectations) and is exercised through the REAL production decision chain:

    take_grouping_provider.safe_group_takes (lexical tier)
    -> reconcile_semantic_idea_equivalence (bounded semantic-equivalence tier)
    -> take_judge.rank_takes (completeness + performance-quality competition)
    -> deterministic_best_take_authority (KEEP/DISCARD Best-Take authority)
    -> final_story_coherence_validation (residual-ambiguity + missing-ending)

Deliberately out of this suite's exercised scope, and documented as such
rather than silently skipped: hybrid_composite_best_take.py's dedicated
composite-reconciliation machinery (installed as a production monkeypatch
wrapper deep in flow_b's session-cleanup chain, impractical to invoke in
isolation without real media/ASR) and the full media->ASR->attempt-
reconstruction front end (needs real audio/video, not available here). The
"composite required" and "continuation must not collapse" fixtures instead
validate the invariant this suite CAN check without that machinery: two
genuinely different, non-competing beats must never be forced into one
retry contest and must both survive to KEEP.

No Video00 timestamps, phrases, or clip IDs anywhere in this file.
"""
from dataclasses import replace

from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.claim_coverage_best_take import apply_claim_coverage_best_take
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, MediaSignals, SCHEMA_VERSION
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.final_edit_reviewer import CAUSAL_ORDER_BREAK, CRITICAL_CLAIM_LOST, review
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.repair_loop import run_repair_loop
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence, safe_group_takes
from cutsell_worker.take_judge import rank_takes


class OracleArbiter:
    """A fake semantic-equivalence arbiter driven by an explicit oracle: a
    set of clip_id pairs a *correct* arbiter would call the same idea. This
    isolates the pipeline mechanics under test from real Gemini classification
    quality (covered separately by the isolation probe and its own tests)."""

    def __init__(self, same_idea_clip_id_pairs=frozenset()):
        self.same_idea_pairs = {frozenset(pair) for pair in same_idea_clip_id_pairs}
        self.calls = 0

    def check(self, request):
        self.calls += 1
        # The arbiter only ever sees text, never clip_ids -- this fake looks
        # up clip_id via the caller-supplied text->clip_id map it's handed
        # out-of-band by the test harness below (see _run_core).
        decisions = tuple(
            IdeaEquivalenceDecision(
                pair_index=i,
                same_idea=frozenset(self._resolve(pair.left_text, pair.right_text)) in self.same_idea_pairs,
                confidence=0.9,
                reason="oracle",
            )
            for i, pair in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(decisions=decisions, provider="fake", model="fake", requested=True, available=True)

    def _resolve(self, left_text, right_text):
        return (self._text_to_id[left_text], self._text_to_id[right_text])


def _take(clip_id, start, end, text, *, complete=True, signals=None, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text, complete_idea=complete, signals=signals)


def _draft_clip(take: CandidateTake, *, selected: bool) -> DraftClip:
    return DraftClip(
        clip_id=take.clip_id, source_asset_id=take.source_asset_id, source_order=take.source_order,
        start=take.start, end=take.end, text=take.text, caption_text=take.text, selected=selected,
    )


def _run_core(takes: tuple[CandidateTake, ...], *, oracle_pairs=frozenset(), claim_equivalence_arbiter=None):
    """Run the real Clean Cut Core V1 decision chain over synthetic takes."""
    arbiter = OracleArbiter(oracle_pairs)
    arbiter._text_to_id = {take.text: take.clip_id for take in takes}

    baseline = safe_group_takes(None, takes)
    merged_groups, equivalence_diag = reconcile_semantic_idea_equivalence(baseline.groups, takes, arbiter)

    take_by_id = {take.clip_id: take for take in takes}
    take_judge_groups = []
    selected_ids: list[str] = []
    for index, ids in enumerate(merged_groups):
        members = [take_by_id[cid] for cid in ids]
        ranked = rank_takes(members)
        if len(members) >= 2:
            take_judge_groups.append({
                "group_id": f"g{index}",
                "ranked": [{"clip_id": r.clip_id, "score": r.score, "reason": r.reason} for r in ranked],
            })
        selected_ids.extend(m.clip_id for m in members)

    selected = tuple(_draft_clip(take_by_id[cid], selected=True) for cid in selected_ids)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="eval-suite", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=(),
        diagnostics={"take_judge_groups": take_judge_groups},
    )
    draft = apply_deterministic_best_take_authority(draft, swap_enabled=False)
    # D-038: real production order (universal_clean_cut.py) runs the
    # claim-coverage-aware Best-Take override strictly between the
    # deterministic ranker and Final Story Coherence Validation -- inserted
    # here too so every fixture in this file (not just the D-038-specific
    # ones below) exercises the exact real chain.
    draft = apply_claim_coverage_best_take(draft, claim_equivalence_arbiter=claim_equivalence_arbiter)
    draft = apply_final_story_coherence_validation(
        draft, semantic_equivalence_arbiter=arbiter, claim_equivalence_arbiter=claim_equivalence_arbiter,
    )
    return draft, equivalence_diag, arbiter


def _kept(draft) -> set[str]:
    return {clip.clip_id for clip in draft.selected}


def _discarded(draft) -> set[str]:
    return {clip.clip_id for clip in draft.discarded}


# 1. Exact retry -----------------------------------------------------------

def test_exact_retry_keeps_complete_discards_stumbled_duplicate():
    text = "We just launched our brand new product line today."
    winner = _take("w", 0.0, 3.0, text, complete=True)
    loser = _take("l", 3.5, 6.5, text, complete=False)  # same words, but a stumbled retry

    draft, _, _ = _run_core((winner, loser))

    assert _kept(draft) == {"w"}
    assert _discarded(draft) == {"l"}


# 2. Paraphrased retry -------------------------------------------------------

def test_paraphrased_retry_merges_via_arbiter_and_keeps_one_winner():
    winner_text = "We just launched our brand new skincare line today."
    loser_text = "Today we're excited to finally roll out our new skincare line."
    winner = _take("w", 0.0, 3.0, winner_text, complete=True)
    loser = _take("l", 5.0, 8.0, loser_text, complete=False)

    draft, equivalence_diag, arbiter = _run_core((winner, loser), oracle_pairs={("w", "l")})

    assert equivalence_diag["status"] == "applied"
    assert arbiter.calls == 1
    assert _kept(draft) == {"w"}
    assert _discarded(draft) == {"l"}


# 3. Two good takes expressing the same idea (GOOD TAKE != UNIQUE IDEA) -----

def test_two_good_takes_same_idea_keeps_exactly_one():
    text_a = "This product completely changed my daily routine for the better."
    text_b = "Honestly, this product has totally transformed how I do things every day."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 5.0, 8.0, text_b, complete=True)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    # Both are individually "good" (complete, well-timed) -- score is tied,
    # so deterministic_best_take_authority alone cannot resolve it (gap < 0.30
    # minimum); final_story_coherence_validation's arbiter fallback must be
    # the one that actually collapses this to one winner.
    assert len(_kept(draft)) == 1
    assert len(_discarded(draft)) == 1
    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["resolved_family_count"] == 1


# 4. False start -> clean retry ---------------------------------------------

def test_false_start_prefix_is_discarded_for_clean_complete_retry():
    full_text = "I want to tell you about the three reasons this product works so well."
    prefix_text = "I want to tell you about"
    false_start = _take("start", 0.0, 1.5, prefix_text, complete=False)
    clean = _take("clean", 2.0, 6.0, full_text, complete=True)

    draft, _, _ = _run_core((false_start, clean))

    assert _kept(draft) == {"clean"}
    assert _discarded(draft) == {"start"}


# 5. Incomplete take -> complete retry ---------------------------------------

def test_incomplete_take_discarded_for_complete_retry_different_wording():
    incomplete_text = "I had some stomach issues for a while and they did some tests and found—"
    complete_text = "I had digestive problems for a while, they ran some tests, and found out it was mild gastritis."
    incomplete = _take("incomplete", 0.0, 4.0, incomplete_text, complete=False)
    complete = _take("complete", 5.0, 10.0, complete_text, complete=True)

    draft, _, _ = _run_core((incomplete, complete), oracle_pairs={("incomplete", "complete")})

    assert _kept(draft) == {"complete"}
    assert _discarded(draft) == {"incomplete"}


# 6. Long bad take vs concise complete take ----------------------------------

def test_long_incomplete_take_discarded_for_concise_complete_take():
    long_bad_text = "So um, basically, what I was trying to say, I mean, the thing is—"
    concise_text = "This product is genuinely worth the price."
    long_bad = _take("long_bad", 0.0, 25.0, long_bad_text, complete=False)  # outside optimal duration AND incomplete
    concise = _take("concise", 30.0, 33.0, concise_text, complete=True)

    draft, _, _ = _run_core((long_bad, concise), oracle_pairs={("long_bad", "concise")})

    assert _kept(draft) == {"concise"}
    assert _discarded(draft) == {"long_bad"}


# 7. Composite required (two genuinely different, non-competing beats) ------

def test_composite_required_both_compatible_sub_deliveries_kept_together():
    part_one = _take("part1", 0.0, 3.0, "There are three reasons this product works so well.", complete=False)
    part_two = _take("part2", 3.2, 7.0, "The first reason is durability, the second is comfort, the third is price.", complete=False)

    # A correct arbiter says these are NOT the same idea -- part2 continues
    # part1, it does not compete with it -- so nothing should ever merge them
    # into one retry contest or force a single winner between them.
    draft, equivalence_diag, arbiter = _run_core((part_one, part_two), oracle_pairs=frozenset())

    assert _kept(draft) == {"part1", "part2"}
    assert _discarded(draft) == set()


# 8. Continuation that must NOT collapse -------------------------------------

def test_continuation_with_shared_vocabulary_does_not_collapse():
    setup_text = "Let me show you how this product performed in my first week."
    continuation_text = "In the second week, the results were even more noticeable than the first."
    setup = _take("setup", 0.0, 3.0, setup_text, complete=True)
    continuation = _take("continuation", 4.0, 7.0, continuation_text, complete=True)

    draft, equivalence_diag, arbiter = _run_core((setup, continuation), oracle_pairs=frozenset())

    assert _kept(draft) == {"setup", "continuation"}
    assert _discarded(draft) == set()


# 9. Similar vocabulary but distinct ideas -----------------------------------

def test_similar_vocabulary_distinct_ideas_both_survive():
    text_a = "The product comes in three different colors to choose from."
    text_b = "The product ships within three business days of ordering."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=True)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs=frozenset())

    assert _kept(draft) == {"a", "b"}
    assert equivalence_diag["status"] in {"checked_no_merge", "no_eligible_pairs"}


# 10. Visual fumble with semantically valid transcript -----------------------

def _signals(**overrides) -> MediaSignals:
    base = dict(
        source_asset_id="src", start=0.0, end=1.0,
        audio_quality=0.9, face_visibility=0.9, eye_contact=0.9, framing_quality=0.9,
        product_visibility=0.9, motion_stability=0.9, continuity=0.9,
        visual_fumble=0.05, expression_naturalness=0.9, gesture_naturalness=0.9,
        delivery_energy=0.9, distraction_risk=0.05,
    )
    base.update(overrides)
    return MediaSignals(**base)


def test_visual_fumble_discarded_despite_semantically_valid_transcript():
    text = "This product genuinely solved a problem I've had for years."
    clean = _take("clean", 0.0, 3.0, text, complete=True, signals=_signals())
    fumbled_text = "This product genuinely solved a problem I have had for a long time."
    fumbled = _take(
        "fumbled", 4.0, 7.0, fumbled_text, complete=True,
        signals=_signals(visual_fumble=0.9, expression_naturalness=0.3, gesture_naturalness=0.3, distraction_risk=0.5),
    )

    draft, _, _ = _run_core((clean, fumbled), oracle_pairs={("clean", "fumbled")})

    # Transcript alone would tie these -- the visual/performance signal is
    # what must break the tie here, not text completeness.
    assert _kept(draft) == {"clean"}
    assert _discarded(draft) == {"fumbled"}


# 11. Unique-fact preservation ------------------------------------------------

def test_unique_fact_preserved_alongside_general_topic_statement():
    general_text = "Let's talk about how this product is made."
    fact_text = "This product uses exactly 12 grams of a patented compound per unit."
    general = _take("general", 0.0, 3.0, general_text, complete=True)
    fact = _take("fact", 4.0, 7.0, fact_text, complete=True)

    # A correct arbiter recognizes the unique numeric fact makes this a
    # different idea, not a competing retry of the general statement.
    draft, _, _ = _run_core((general, fact), oracle_pairs=frozenset())

    assert _kept(draft) == {"general", "fact"}
    assert _discarded(draft) == set()


# 12. CTA / story-ending preservation -----------------------------------------

def test_cta_ending_preserved_flag_stays_false_when_not_dropped():
    opener = _take("opener", 0.0, 3.0, "Here's my honest review of this product.", complete=True)
    cta = _take("cta", 4.0, 6.0, "Thanks for watching, link is in the description.", complete=True)

    draft, _, _ = _run_core((opener, cta))

    assert _kept(draft) == {"opener", "cta"}
    assert draft.diagnostics["final_story_coherence_validation"]["possible_missing_story_ending"] is False


def test_cta_ending_flagged_when_last_take_is_discarded():
    opener = replace(
        DraftClip(clip_id="opener", source_asset_id="src", source_order=0, start=0.0, end=3.0,
                  text="opener", caption_text="opener", selected=True),
    )
    cta = DraftClip(clip_id="cta", source_asset_id="src", source_order=0, start=4.0, end=6.0,
                     text="cta", caption_text="cta", selected=False)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="eval-suite", strategy=EditStrategy.STORYTELLING,
        selected=(opener,), alternates=(), discarded=(cta,), diagnostics={},
    )

    out = apply_final_story_coherence_validation(draft)

    assert out.diagnostics["final_story_coherence_validation"]["possible_missing_story_ending"] is True


# 13. Multiple retries with progressively better delivery ---------------------

def test_three_progressive_retries_keeps_only_the_best():
    worst = _take("worst", 0.0, 3.0, "So, um, I guess this product is okay I think—", complete=False)
    middle = _take("middle", 4.0, 7.0, "This product is pretty good, I'd say it works well.", complete=True)
    best = _take("best", 8.0, 11.0, "This product is excellent and I'd recommend it to anyone.", complete=True)

    draft, equivalence_diag, arbiter = _run_core(
        (worst, middle, best),
        oracle_pairs={("worst", "middle"), ("middle", "best"), ("worst", "best")},
    )

    assert len(_kept(draft)) == 1
    assert _discarded(draft) == {"worst", "middle", "best"} - _kept(draft)


# 14. Same idea, substantially different wording (the core Phase 2 case) -----

def test_same_idea_substantially_different_wording_merges_and_resolves():
    # Mirrors the real-run audit's clearest miss: near-zero lexical overlap,
    # unambiguously the same intended statement to a human reader.
    text_a = "At the end of my contract I talked to my doctor and asked for every test she could think of."
    text_b = "When my contract ended I switched doctors and had her run every test imaginable."
    a = _take("a", 0.0, 4.0, text_a, complete=True)
    b = _take("b", 6.0, 10.0, text_b, complete=False)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    assert equivalence_diag["status"] == "applied"
    assert _kept(draft) == {"a"}
    assert _discarded(draft) == {"b"}


# 15. Contradictory factual retries -- hard freeze-blocking invariant --------

def test_contradictory_factual_retries_block_freeze_not_silently_resolved():
    text_a = "No soy la unica en mi familia con este problema."
    text_b = "Soy la unica en mi familia con este problema."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=True)

    # Score-tied (both complete, same duration bucket) -> ambiguous gap ->
    # deterministic_best_take_authority cannot resolve it locally, AND the
    # arbiter confirming "same idea" must NOT be treated as license to
    # silently keep one and discard the other when the two texts are
    # factually incompatible -- freeze must block instead.
    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    assert len(diag["contradiction_findings"]) == 1
    # Neither was silently discarded on the arbiter's same-idea verdict alone.
    assert _kept(draft) == {"a", "b"}


# 16. Self-correction within one continuous take -----------------------------

def test_self_correction_within_one_take_is_not_treated_as_a_retry_family():
    # A single delivery that corrects itself mid-sentence is ONE take, not
    # two competing attempts -- there is nothing to group it with.
    text = "It costs fifty dollars—actually, sorry, forty dollars."
    only_take = _take("only", 0.0, 4.0, text, complete=True)

    draft, equivalence_diag, arbiter = _run_core((only_take,))

    assert _kept(draft) == {"only"}
    assert _discarded(draft) == set()
    assert "take_judge_groups" not in draft.diagnostics or not draft.diagnostics["take_judge_groups"]


# 17. Retry disguised as continuation -----------------------------------------

def test_retry_disguised_as_continuation_still_merges_as_same_idea():
    # Superficially reads like it's adding detail, but it's actually just
    # restating the same point in a slightly expanded way -- a correct
    # arbiter still calls this the same idea, not a real continuation.
    text_a = "This routine helped me sleep better."
    text_b = "Something that really helped me sleep better was this exact routine."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=False)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    # High lexical overlap here means the lexical tier may already catch
    # this one (status "applied" via semantic equivalence is not guaranteed
    # -- what matters editorially is the outcome, not which tier caught it).
    assert _kept(draft) == {"a"}
    assert _discarded(draft) == {"b"}


# 18. Continuation disguised as retry -----------------------------------------

def test_continuation_disguised_as_retry_does_not_merge():
    # Shares an opening phrase/vocabulary with a plausible "retry" shape, but
    # actually adds genuinely new information -- a correct arbiter says
    # different idea, so both must survive.
    text_a = "This routine helped me sleep better."
    text_b = "This routine also helped clear up my skin within a month."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=True)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs=frozenset())

    assert _kept(draft) == {"a", "b"}
    assert _discarded(draft) == set()


# 19. Composite forbidden because a complete take already exists -------------

def test_composite_not_forced_when_complete_take_exists_for_its_own_idea():
    # "complete" fully covers idea 1 on its own. "part1"/"part2" are a
    # genuine two-piece composite of a DIFFERENT idea (idea 2). Nothing here
    # should wrongly merge idea 2's pieces into idea 1's already-complete
    # winner, and idea 1 must not be turned into a composite it doesn't need.
    complete = _take("complete", 0.0, 3.0, "This product is excellent value for the price.", complete=True)
    part1 = _take("part1", 4.0, 6.0, "There are two things I'd change about it.", complete=False)
    part2 = _take("part2", 6.2, 9.0, "The charging cable is short and the box is flimsy.", complete=False)

    draft, equivalence_diag, arbiter = _run_core((complete, part1, part2), oracle_pairs=frozenset())

    assert _kept(draft) == {"complete", "part1", "part2"}
    assert _discarded(draft) == set()


# 20. Orphan fragment (no competing retry, nothing outranks it) --------------

def test_orphan_incomplete_fragment_kept_by_default_when_uncertain():
    # A short, incomplete fragment with no retry-family partner has nothing
    # to lose a competition to -- WHEN UNCERTAIN, KEEP (do not fabricate a
    # competition/discard where no competing evidence exists).
    fragment = _take("fragment", 0.0, 2.0, "And another thing about the setup process—", complete=False)
    unrelated = _take("unrelated", 3.0, 6.0, "The packaging was really nice, by the way.", complete=True)

    draft, equivalence_diag, arbiter = _run_core((fragment, unrelated), oracle_pairs=frozenset())

    assert _kept(draft) == {"fragment", "unrelated"}
    assert _discarded(draft) == set()


# 21. Duplicate semantic beat restated far apart in time ----------------------

def test_duplicate_beat_far_apart_in_time_not_yet_merged_known_limitation():
    # Documents a REAL, currently-known limitation rather than a passing
    # capability: the semantic-equivalence eligibility gate caps at 30s
    # (take_grouping_provider.reconcile_semantic_idea_equivalence's
    # maximum_gap_sec default), so a fact restated well beyond that window
    # is not even proposed to the arbiter as a candidate pair today. This
    # test pins CURRENT behavior (both survive, duplicated) so a future
    # widening of that window is a deliberate, visible decision -- not a
    # silent behavior change this suite fails to notice.
    text_a = "We just launched our brand new skincare line today."
    text_b = "Today we're excited to finally roll out our new skincare line."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 45.0, 48.0, text_b, complete=True)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    assert equivalence_diag["status"] in {"no_eligible_pairs", "not_requested"}
    assert arbiter.calls == 0
    assert _kept(draft) == {"a", "b"}


# 22. Uncertain semantic case -- arbiter unavailable fails open --------------

def test_uncertain_case_with_unavailable_arbiter_fails_open_preserves_both():
    # No arbiter provided at all (e.g. provider down/disabled) -- a
    # genuinely low-overlap pair that WOULD need the arbiter to resolve must
    # never be guessed at; both survive rather than one being discarded on
    # no evidence.
    text_a = "We just launched our brand new skincare line today."
    text_b = "Today we're excited to finally roll out our new skincare line."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=False)

    draft, equivalence_diag, arbiter = _run_core((a, b))  # no oracle_pairs, arbiter never confirms

    assert equivalence_diag["status"] in {"checked_no_merge", "arbiter_unavailable", "no_eligible_pairs", "not_requested"}
    assert _kept(draft) == {"a", "b"}
    assert _discarded(draft) == set()


# 23. Multilingual paraphrased retry ------------------------------------------

def test_multilingual_paraphrased_retry_merges_and_keeps_one_winner():
    text_a = "Nunca se nos ocurrió hacer un chequeo de la tiroides porque los examenes siempre salian normales."
    text_b = "No se nos ocurrio revisar la tiroides porque en mis examenes previos todo salia bien."
    a = _take("a", 0.0, 4.0, text_a, complete=True)
    b = _take("b", 5.0, 9.0, text_b, complete=False)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    assert equivalence_diag["status"] == "applied"
    assert _kept(draft) == {"a"}
    assert _discarded(draft) == {"b"}


# 24. Causal/story order preserved across independent beats ------------------

def test_causal_order_preserved_across_independent_non_retry_beats():
    # Three genuinely independent story beats in causal/chronological order
    # must all survive, in their original relative order -- nothing here is
    # a retry of anything else.
    beat1 = _take("beat1", 0.0, 3.0, "First I noticed the symptoms getting worse.", complete=True)
    beat2 = _take("beat2", 4.0, 7.0, "So I went to see a specialist about it.", complete=True)
    beat3 = _take("beat3", 8.0, 11.0, "They ran some tests and found the cause.", complete=True)

    draft, equivalence_diag, arbiter = _run_core((beat1, beat2, beat3), oracle_pairs=frozenset())

    assert _kept(draft) == {"beat1", "beat2", "beat3"}
    kept_in_order = [clip.clip_id for clip in sorted(draft.selected, key=lambda c: c.start)]
    assert kept_in_order == ["beat1", "beat2", "beat3"]


# 25. Factual/numeric correction across two takes -- conservative by design --

def test_numeric_correction_across_two_takes_conservatively_blocks_freeze():
    # A creator visibly self-corrects a number ACROSS two separate takes
    # (not one continuous utterance -- see fixture 16 for that case). A
    # human editor would obviously keep the corrected, more confident take.
    # This suite intentionally does NOT auto-resolve that: distinguishing
    # "genuine correction" from "genuinely contradictory competing claims"
    # would need new correction-marker heuristics this deterministic pass
    # does not have, and guessing wrong here ships a wrong fact. The
    # designed behavior is the conservative one -- block freeze, let a human
    # resolve it in seconds -- not a silent auto-pick. If correction-marker
    # detection is added later, this fixture's oracle should change with it.
    text_a = "I think the event happened back in 2019."
    text_b = "Actually, I checked, and it was 2020."
    a = _take("a", 0.0, 3.0, text_a, complete=True)
    b = _take("b", 4.0, 7.0, text_b, complete=True)

    draft, equivalence_diag, arbiter = _run_core((a, b), oracle_pairs={("a", "b")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    assert diag["contradiction_findings"][0]["number_conflict"] is True
    assert _kept(draft) == {"a", "b"}


# 26. Retry winner discards a loser that carried a unique numeric fact -------
#
# RAW 33345946000 (Clean Cut Core V1's first controlled run) found this
# exact failure shape live: a hybrid deletion silently dropped a delivery's
# unique medical fact even though a "better" delivery of the same idea
# survived, and final_story_coherence_validation reported freeze_blocked=
# false because its idea-coverage check was scoped to take_judge_groups,
# which never saw the deleted candidate. This suite still cannot invoke
# hybrid_session_cleanup itself in isolation (see module docstring), but
# the identical failure shape is reachable through the real Best-Take/
# coherence chain whenever the winning realization of a genuine retry
# family simply does not carry every fact the losing member did -- which is
# what the lost-semantic-atoms coverage ledger (final_story_coherence_
# validation._lost_semantic_atoms) now catches regardless of which
# authority did the discarding.

def test_retry_winner_missing_losers_unique_numeric_fact_blocks_freeze():
    keeper_text = "I had thyroid surgery last year and it went well."
    lost_fact_text = "I had thyroid surgery last year, my 3rd surgery, and it went well."
    keeper = _take("keeper", 0.0, 3.0, keeper_text, complete=True)
    loser = _take("loser_with_fact", 4.0, 7.0, lost_fact_text, complete=True)

    draft, _, _ = _run_core((keeper, loser), oracle_pairs={("keeper", "loser_with_fact")})

    assert _kept(draft) == {"keeper"}
    assert _discarded(draft) == {"loser_with_fact"}
    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    lost_ids = [row["clip_id"] for row in diag["lost_semantic_atoms"]]
    assert "loser_with_fact" in lost_ids


# 27. Retry winner discards a loser carrying an entire unrelated story beat --
#
# The same shape as #26, but for a non-numeric story/symptom beat rather
# than a single critical atom -- the general content-vocabulary side of the
# coverage ledger, not the number/negation side. Directly maps to this RAW's
# other confirmed loss (the pimples/rash symptom beat, discarded alongside a
# retry of an unrelated statement and never independently covered again).

def test_retry_winner_missing_losers_unrelated_symptom_beat_blocks_freeze():
    keeper_text = "The dermatologist looked at my skin and said it was fine."
    loser_text = (
        "The dermatologist looked at my skin, but I also had pimples "
        "breaking out behind my ear that nobody addressed."
    )
    keeper = _take("keeper", 0.0, 3.0, keeper_text, complete=True)
    loser = _take("loser_beat", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((keeper, loser), oracle_pairs={("keeper", "loser_beat")})

    assert _kept(draft) == {"keeper"}
    assert _discarded(draft) == {"loser_beat"}
    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    lost_ids = [row["clip_id"] for row in diag["lost_semantic_atoms"]]
    assert "loser_beat" in lost_ids


# Categories requested for this expansion that this file's harness cannot
# reach honestly (same scoping limit already documented above for BTS/
# composite: they need either hybrid_session_cleanup's pre-grouping deletion
# or hybrid_composite_best_take.py's monkeypatch machinery, neither
# invocable here in isolation) are instead covered directly at the
# final_story_coherence_validation unit level, in
# tests/test_cutsell_final_story_coherence_validation.py:
#   - a whole symptom/story beat discarded before any grouping ever ran
#     (test_lost_semantic_atoms_blocks_freeze_for_content_discarded_before_any_grouping)
#   - destructive deletion followed by zero idea coverage / final KEEP
#     timeline coverage checked directly rather than inferred from
#     take_judge_groups (same test -- the fixture has no take_judge_groups
#     entry at all for the lost clip)
#   - a correctly-discarded redundant retry must not false-positive
#     (test_lost_semantic_atoms_does_not_flag_correctly_discarded_redundant_retry)
# "Composite replacement loses one complementary fact" and "causal
# transition accidentally removed" both reduce to the identical coverage-
# ledger question (is the fact anywhere in the final KEEP text?) and are not
# duplicated again here.


# 28. End-to-end (D-026): repair loop fixes a disordered composite reached
# through the REAL take-grouping/idea-equivalence/take-judge/coherence chain.
#
# part1/part2 are the same non-competing, complementary-beats shape as
# fixture 7 ("composite required") -- the real chain above correctly keeps
# both, never forcing them into one retry contest. This suite still cannot
# invoke hybrid_composite_best_take.py's own composite-reconciliation
# machinery in isolation (see module docstring), so the composite mark
# itself is injected exactly the way that machinery's real output already
# looks (a hybrid_editorial_chunks row naming the split_group_clip_ids) --
# the same simulation fixture 7 already relies on to prove its own
# invariant without that machinery. What is new and real here: the physical
# render order is deliberately reversed after the real chain runs, and
# build_canonical_edit_plan -> run_repair_loop -> review are then run for
# real against that output -- proving the repair loop actually integrates
# with this chain's real diagnostics shape, not just a hand-built draft.

def test_repair_loop_fixes_disordered_composite_reached_through_the_real_chain():
    part_one = _take("part1", 0.0, 3.0, "There are three reasons this product works so well.", complete=False)
    part_two = _take("part2", 3.2, 7.0, "The first reason is durability, the second is comfort, the third is price.", complete=False)

    draft, _, _ = _run_core((part_one, part_two), oracle_pairs=frozenset())
    assert _kept(draft) == {"part1", "part2"}

    # Simulate CompositeResolver's real acceptance of this pair (see
    # docstring above), and physically render them in the wrong order --
    # the exact failure shape STORY_ORDER_BREAK/the repair loop exist for.
    # part1/part2 never contested each other (no shared take_judge_groups
    # entry -- the real chain correctly never treats non-competing beats as
    # one retry family), so CanonicalEditPlan needs that same union CompositeResolver's
    # real output would have produced, to see them as one Idea at all.
    reversed_selected = tuple(sorted(draft.selected, key=lambda c: c.clip_id, reverse=True))
    assert [c.clip_id for c in reversed_selected] == ["part2", "part1"]
    composite_diagnostics = dict(draft.diagnostics)
    composite_diagnostics["take_judge_groups"] = [
        {"group_id": "g_composite", "ranked": [
            {"clip_id": "part1", "score": 0.7, "reason": "composite"},
            {"clip_id": "part2", "score": 0.7, "reason": "composite"},
        ]},
    ]
    composite_diagnostics["hybrid_editorial_chunks"] = [
        {"hybrid_composite_best_take": {"split_group_clip_ids": ["part1", "part2"]}}
    ]
    disordered_draft = replace(draft, selected=reversed_selected, diagnostics=composite_diagnostics)

    result = run_repair_loop(disordered_draft)

    assert result.status == "PASS"
    assert len(result.attempts) == 1
    assert result.attempts[0].finding_kind == "STORY_ORDER_BREAK"
    assert [c.clip_id for c in result.final_draft.selected] == ["part1", "part2"]


# 29. End-to-end (D-027): a general causal order break, reached through the
# same real chain, is detected by FinalEditReviewer.
#
# beat1/beat2 are two independent, non-retry story beats -- the real chain
# above keeps both (same invariant as fixture 24), never merging or
# reordering them itself. beat2 opens with a strong, general dependency
# connector ("therefore") naming it a consequence of beat1. The physical
# render order is then deliberately inverted -- the same kind of authoring
# mistake causal_order_validator.py exists to catch -- and CanonicalEditPlan
# + review() are run for real against that output.

def test_general_causal_order_break_reached_through_the_real_chain_blocks_review():
    beat1 = _take("beat1", 0.0, 3.0, "We ran the test on the sample.", complete=True)
    beat2 = _take("beat2", 4.0, 7.0, "Therefore the results were confirmed as accurate.", complete=True)

    draft, _, _ = _run_core((beat1, beat2), oracle_pairs=frozenset())
    assert _kept(draft) == {"beat1", "beat2"}

    inverted_draft = replace(draft, selected=tuple(reversed(draft.selected)))
    assert [c.clip_id for c in inverted_draft.selected] == ["beat2", "beat1"]

    plan = build_canonical_edit_plan(inverted_draft)
    result = review(plan)

    assert result.status == "FAIL"
    causal_findings = [f for f in result.findings if f.kind == CAUSAL_ORDER_BREAK]
    assert len(causal_findings) == 1
    assert causal_findings[0].detail["required_clip_id"] == "beat1"
    assert causal_findings[0].detail["dependent_clip_id"] == "beat2"


# 30-36. Semantic-atom importance classification (D-031), reached through
# the REAL take-grouping/idea-equivalence/take-judge/coherence chain.
#
# RAW 33402023395's own failure shape, generalized: a losing retry's
# incidental year blocked Freeze even though the audience-facing idea was
# fully intact in the winning delivery. These fixtures map the canonical
# directive's ten named categories onto seven real-chain cases (several
# categories collapse onto the same underlying CRITICAL/CONTEXTUAL split):
#   - incidental year safely omitted            -> fixture 30
#   - year required for chronology               -> fixture 31
#   - numeric measurement that must survive      -> fixture 32
#   - percentage that must survive               -> fixture 33
#   - dose/price/quantity that must survive      -> fixture 34
#   - redundant date repeated in two attempts    -> fixture 35
#   - unique factual date referenced by later idea,
#     ambiguous atom routed to arbiter           -> fixture 36 (no arbiter
#     configured -- stays UNCERTAIN, which still blocks; the dedicated
#     confirm/deny arbiter-routing mechanics are unit-tested exhaustively
#     in tests/test_cutsell_semantic_atom_importance.py)
#   - removal changes meaning -> critical / removal does not change the
#     proposition -> contextual: the CRITICAL fixtures (31-34) and the
#     CONTEXTUAL fixture (30) together ARE this contrast.
# No Video00 fact/phrase/literal value appears in any fixture below.

def test_incidental_year_safely_omitted_does_not_block_freeze():
    winner_text = "I had digestion problems and it turned out to be gastritis, nothing severe."
    loser_text = "During one period in 2023 I had digestion problems and it turned out to be gastritis, nothing severe."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    assert _kept(draft) == {"keeper"}
    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    finding = diag["lost_semantic_atoms"][0]
    assert finding["blocking"] is False


def test_year_required_for_chronology_still_blocks_freeze():
    winner_text = "I started feeling worse and went to get checked out right away."
    loser_text = "I started feeling worse in 2023, and before that everything felt completely normal."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    assert finding["blocking"] is True
    classes = {c["importance"] for c in finding["atom_classifications"]}
    assert "CRITICAL" in classes


def test_numeric_measurement_must_survive_blocks_freeze():
    winner_text = "They found something unusual during the scan and sent it for testing."
    loser_text = "They found something unusual during the scan measuring 3 centimeters and sent it for testing."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    classes = {c["importance"]: c["evidence"] for c in finding["atom_classifications"]}
    assert classes.get("CRITICAL") == "measurement"


def test_percentage_must_survive_via_critical_coverage_dominance():
    # D-063: this fixture's two candidates tie exactly on the deterministic
    # ranker's own score (both "text_timing_baseline" 1.0), so
    # `apply_deterministic_best_take_authority` cannot break the tie and
    # leaves BOTH selected -- the genuinely ambiguous, 2-selected shape
    # CRITICAL_COVERAGE_DOMINANCE now resolves in `claim_coverage_best_
    # take.py`, BEFORE Final Story Coherence Validation ever runs. Before
    # D-063, this ambiguity reached StoryValidator's own residual-family
    # fallback, which (with no claim-coverage awareness) could pick either
    # side -- the percentage then only "survived" in the sense that losing
    # it was CAUGHT and blocked. Now the fact survives directly: the
    # candidate that actually carries it (`loser_atom`) is the one
    # CRITICAL_COVERAGE_DOMINANCE prefers, so nothing is lost and Freeze is
    # correctly not blocked at all -- a stronger guarantee than "catch the
    # loss after the fact", not a weaker one.
    winner_text = "Most people who try the program end up quitting before they finish it."
    loser_text = "Only 15% of people who try the program actually end up finishing it."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    assert [c.clip_id for c in draft.selected] == ["loser_atom"]
    assert [c.clip_id for c in draft.discarded] == ["keeper"]
    dominance = draft.diagnostics["claim_coverage_best_take"]["dominance_resolutions"]
    assert len(dominance) == 1
    assert dominance[0]["winner_clip_id"] == "loser_atom"
    assert dominance[0]["reason"] == "critical_coverage_dominance"

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["lost_semantic_atoms"] == []


def test_dose_quantity_must_survive_blocks_freeze():
    winner_text = "They put me on medication for a while and it cleared up."
    loser_text = "They put me on medication, 2 pills every morning for a while, and it cleared up."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    classes = {c["importance"]: c["evidence"] for c in finding["atom_classifications"]}
    assert classes.get("CRITICAL") == "dose_or_quantity"


def test_redundant_date_repeated_in_two_attempts_is_not_flagged_at_all():
    # The year appears in BOTH the winner and the loser -- never actually
    # "missing" from the final KEEP text, so no atom finding is raised for
    # it at all (the pre-check that decides something is "missing" already
    # handles this; the importance classifier is never even reached).
    winner_text = "During one period in 2023 I had digestion problems and it turned out to be gastritis."
    loser_text = "In 2023 I had some digestion problems that turned out to be gastritis, nothing major."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["lost_semantic_atoms"] == []


def test_ambiguous_atom_with_no_deterministic_signal_stays_blocking_without_an_arbiter():
    # No unit/currency/percent/correction/chronology marker, not a
    # plausible year -- an ordinary bare quantity this deterministic layer
    # cannot confidently place. UNCERTAIN, and with no arbiter configured
    # (the default everywhere in this pipeline today), it stays blocking --
    # WHEN UNCERTAIN, KEEP, never silently downgraded to a warning.
    winner_text = "I tried a bunch of different approaches before something finally worked."
    loser_text = "I tried 7 different approaches before something finally worked for me."
    winner = _take("keeper", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser_atom", 4.0, 7.0, loser_text, complete=True)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("keeper", "loser_atom")})

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    assert finding["blocking"] is True
    classes = {c["importance"] for c in finding["atom_classifications"]}
    assert "UNCERTAIN" in classes


# 38-49. Per-Idea semantic claim coverage (D-038), reached through the REAL
# take-grouping/idea-equivalence/take-judge/coherence chain, with the
# claim-coverage-aware Best-Take override (claim_coverage_best_take.py) now
# wired into that chain (see _run_core above).
#
# RAW 33423953391's own failure shape, generalized: BestTake picked a
# cleaner-but-incomplete candidate over one carrying a critical diagnosis-
# confirmation claim, and the OLD whole-KEEP-timeline vocabulary check
# (fixtures 26-27's `_lost_semantic_atoms`) missed it because the same
# words happened to recur in an unrelated clip elsewhere in the video. These
# fixtures map the canonical directive's 12 named categories onto the real
# chain. No Video00 fact/phrase/literal value appears in any fixture below.

def test_cleaner_take_losing_a_diagnosis_claim_must_not_win():
    cleaner = _take("cleaner", 0.0, 3.0, "Anyway that's been my honest take on the whole thing so far.", complete=True)
    diagnosis = _take("diagnosis", 4.0, 7.0, "The biopsy confirmed it was a benign tumor, which was a huge relief.", complete=False)

    draft, _, _ = _run_core((cleaner, diagnosis), oracle_pairs={("cleaner", "diagnosis")})

    assert _kept(draft) == {"diagnosis"}
    assert _discarded(draft) == {"cleaner"}
    override = draft.diagnostics["claim_coverage_best_take"]["overrides"][0]
    assert override["previous_winner_clip_id"] == "cleaner"
    assert override["new_winner_clip_id"] == "diagnosis"
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_shorter_take_losing_a_supporting_cause_effect_claim_must_not_force_an_override():
    # D-038 item 6: only CRITICAL claim loss must always block/override --
    # a cause/effect explanatory clause classifies SUPPORTING (see
    # semantic_claims.classify_claim), never CRITICAL, so its loss alone
    # must NOT force the current (decisive, complete) winner to lose. This
    # is the deliberate boundary, not an oversight: overcorrecting into
    # preserving every explanatory sentence would defeat Best-Take entirely.
    winner = _take("winner", 0.0, 3.0, "I want to share my quick update after using the product for a while.", complete=True)
    loser = _take("loser", 4.0, 7.0, "I switched brands because the smell was too strong for me.", complete=False)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})

    assert _kept(draft) == {"winner"}
    assert _discarded(draft) == {"loser"}
    assert draft.diagnostics.get("claim_coverage_best_take") is None  # never touched -- no critical claim in the group


def test_take_with_all_critical_claims_wins_despite_worse_performance():
    # "worse performance" here is the take_judge signal this synthetic
    # harness actually has to work with (complete_idea=False -- see
    # take_judge.score_take): a rougher-scoring delivery that nonetheless
    # carries EVERY critical claim in the family must still win over a
    # clean-scoring delivery carrying none of them.
    cleaner = _take("cleaner", 0.0, 3.0, "Anyway that's been my honest take on the whole thing so far.", complete=True)
    all_critical = _take(
        "all_critical", 4.0, 7.0,
        "The test came back positive. The biopsy also confirmed it was a serious condition needing treatment.",
        complete=False,
    )

    draft, _, _ = _run_core((cleaner, all_critical), oracle_pairs={("cleaner", "all_critical")})

    assert _kept(draft) == {"all_critical"}
    assert _discarded(draft) == {"cleaner"}
    override = draft.diagnostics["claim_coverage_best_take"]["overrides"][0]
    assert len(override["missing_claim_ids"]) == 2  # both claims were missing from the wrong winner
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_complementary_critical_claims_require_a_composite():
    winner = _take("winner", 0.0, 3.0, "So overall that's been my journey with this so far.", complete=True)
    piece_a = _take("piece_a", 4.0, 7.0, "The test came back positive for the condition.", complete=False)
    piece_b = _take(
        "piece_b", 8.0, 11.0,
        "The biopsy confirmed it was something more serious that needed treatment.",
        complete=False,
    )

    draft, _, _ = _run_core(
        (winner, piece_a, piece_b),
        oracle_pairs={("winner", "piece_a"), ("winner", "piece_b"), ("piece_a", "piece_b")},
    )

    assert _kept(draft) == {"piece_a", "piece_b"}
    assert _discarded(draft) == {"winner"}
    composite = draft.diagnostics["claim_coverage_best_take"]["composites"][0]
    assert set(composite["clip_ids"]) == {"piece_a", "piece_b"}
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_claim_missing_from_its_own_idea_still_fails_despite_similar_words_elsewhere_in_the_video():
    # The exact invariant RAW 33423953391 exposed: whole-video vocabulary
    # presence must never satisfy a DIFFERENT idea's missing claim. Three
    # critical claims are split three ways here so claim_coverage_best_
    # take.py's own bounded resolution (single candidate / 2-piece
    # composite) cannot safely fix it either -- proving the real backstop,
    # final_story_coherence_validation._lost_critical_claims, still catches
    # it on its own, per-idea, even with an unrelated but lexically similar
    # clip sitting elsewhere in the same final KEEP timeline.
    winner = _take("winner", 0.0, 3.0, "Anyway that's been my honest take on the whole thing so far.", complete=True)
    a = _take("a", 4.0, 7.0, "The test came back positive for the condition.", complete=False)
    b = _take("b", 8.0, 11.0, "The biopsy confirmed it was something concerning that needed follow up.", complete=False)
    c = _take("c", 12.0, 15.0, "Only 5 percent of people ever experience this particular reaction.", complete=False)
    elsewhere = _take(
        "elsewhere", 100.0, 103.0,
        "We also asked about biopsy procedures and testing percentages in a totally different consultation.",
        complete=True,
    )

    draft, _, _ = _run_core(
        (winner, a, b, c, elsewhere),
        oracle_pairs={("winner", "a"), ("winner", "b"), ("winner", "c"), ("a", "b"), ("a", "c"), ("b", "c")},
    )

    assert _kept(draft) == {"winner", "elsewhere"}
    assert _discarded(draft) == {"a", "b", "c"}
    ccbt = draft.diagnostics["claim_coverage_best_take"]
    assert ccbt["overrides"] == []
    assert ccbt["composites"] == []
    assert len(ccbt["unresolved_gaps"]) == 1

    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    lost_ids = {row["source_clip_id"] for row in diag["lost_critical_claims"]}
    assert lost_ids == {"a", "b", "c"}
    for row in diag["lost_critical_claims"]:
        assert row["winning_clip_ids"] == ["winner"]  # never satisfied by "elsewhere"


def test_same_nouns_present_in_wrong_winner_does_not_produce_false_coverage():
    # winner shares "biopsy"/"tumor" vocabulary with the actual critical
    # claim but never asserts it -- claim_coverage's token-overlap check
    # alone would land in the ambiguous band (not confidently covered), and
    # with no arbiter configured it fails open to NOT covered, so the
    # override still correctly fires despite the shared nouns.
    winner = _take("winner", 0.0, 3.0, "The biopsy and the tumor came up briefly in conversation.", complete=True)
    loser = _take("loser", 4.0, 7.0, "The biopsy confirmed it was a benign tumor.", complete=False)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})

    assert _kept(draft) == {"loser"}
    assert _discarded(draft) == {"winner"}
    override = draft.diagnostics["claim_coverage_best_take"]["overrides"][0]
    assert override["new_winner_clip_id"] == "loser"


def test_contextual_claim_safely_omitted_never_touches_claim_coverage_best_take():
    winner = _take("winner", 0.0, 3.0, "I want to share my quick update on this whole thing.", complete=True)
    loser = _take("loser", 4.0, 7.0, "After trying it for a while, I want to share my update on this.", complete=False)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})

    assert _kept(draft) == {"winner"}
    assert draft.diagnostics.get("claim_coverage_best_take") is None
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_supporting_claim_omitted_but_core_idea_intact_never_blocks():
    winner = _take("winner", 0.0, 3.0, "Overall I am really happy with this whole thing.", complete=True)
    loser = _take("loser", 4.0, 7.0, "I picked it up at the store and gave it a try right away.", complete=False)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})

    assert _kept(draft) == {"winner"}
    assert draft.diagnostics.get("claim_coverage_best_take") is None
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_critical_correction_is_preserved_over_a_cleaner_but_wrong_winner():
    cleaner = _take("cleaner", 0.0, 3.0, "So that's the update on where things stand right now.", complete=True)
    correction = _take(
        "correction", 4.0, 7.0,
        "Actually, I was wrong earlier, it turned out to be something different entirely.",
        complete=False,
    )

    draft, _, _ = _run_core((cleaner, correction), oracle_pairs={("cleaner", "correction")})

    assert _kept(draft) == {"correction"}
    assert _discarded(draft) == {"cleaner"}
    override = draft.diagnostics["claim_coverage_best_take"]["overrides"][0]
    assert override["new_winner_clip_id"] == "correction"


def test_critical_claim_split_across_a_genuinely_independent_continuation_is_left_alone():
    # beat2 is a natural continuation of beat1, not a retry of it -- no
    # oracle pair, so the real chain never groups them into one retry-
    # family contest (same invariant as fixture 8, "continuation that must
    # NOT collapse"). Neither claim_coverage_best_take nor
    # _lost_critical_claims ever evaluates clips that were never placed in
    # a shared take_judge_groups entry -- both correctly stay untouched.
    beat1 = _take("beat1", 0.0, 3.0, "The test came back positive for the infection.", complete=True)
    beat2 = _take("beat2", 4.0, 7.0, "Because of that, we started the treatment right away.", complete=True)

    draft, _, _ = _run_core((beat1, beat2), oracle_pairs=frozenset())

    assert _kept(draft) == {"beat1", "beat2"}
    assert draft.diagnostics.get("take_judge_groups") == []
    assert draft.diagnostics.get("claim_coverage_best_take") is None
    assert draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"] == []


def test_duplicate_retry_carrying_no_unique_claim_is_a_safe_plain_discard():
    # Both sides state the IDENTICAL critical claim -- after dedup there is
    # only one, and the surviving winner already covers it fully, so
    # claim_coverage_best_take is correctly a complete no-op: this is an
    # ordinary exact-retry discard, not a claim-coverage case at all.
    text = "The biopsy confirmed it was a benign tumor."
    winner = _take("w", 0.0, 3.0, text, complete=True)
    loser = _take("l", 3.5, 6.5, text, complete=False)

    draft, _, _ = _run_core((winner, loser))

    assert _kept(draft) == {"w"}
    assert _discarded(draft) == {"l"}
    assert draft.diagnostics.get("claim_coverage_best_take") is None


class _AlwaysConfirmClaimArbiter:
    """Bounded ClaimEquivalenceArbiter fake: always confirms the proposed
    winning realization preserves the claim. Used only to prove the arbiter
    is consulted (and actually changes the outcome) in the ambiguous
    coverage band -- never for a confidently-covered or confidently-lost
    case (see resolve_ambiguous_coverage's own floor/threshold gate)."""

    def __init__(self):
        self.calls = 0

    def claim_covered(self, claim_text, winning_realization_text):
        self.calls += 1
        return True, 0.85, "paraphrase confirmed by arbiter"


def test_arbiter_is_consulted_only_for_ambiguous_coverage_and_can_change_the_outcome():
    # winner's phrasing shares enough of the claim's own words to land in
    # the ambiguous coverage band (0.3-0.6, see semantic_claims.py) but not
    # enough to be confidently covered -- exactly the one case
    # resolve_ambiguous_coverage escalates to a bounded arbiter.
    winner = _take("winner", 0.0, 3.0, "After the biopsy, they said the tumor situation was fine.", complete=True)
    loser = _take("loser", 4.0, 7.0, "The biopsy confirmed it was a benign tumor.", complete=False)

    # Without an arbiter: ambiguous coverage fails open to NOT covered, so
    # the fully-covering loser correctly becomes the new winner.
    no_arbiter_draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})
    assert _kept(no_arbiter_draft) == {"loser"}

    # With an arbiter that confirms the paraphrase preserves the claim: no
    # override is needed at all, and the arbiter was actually consulted.
    arbiter = _AlwaysConfirmClaimArbiter()
    with_arbiter_draft, _, _ = _run_core(
        (winner, loser), oracle_pairs={("winner", "loser")}, claim_equivalence_arbiter=arbiter,
    )
    assert _kept(with_arbiter_draft) == {"winner"}
    assert with_arbiter_draft.diagnostics.get("claim_coverage_best_take") is None
    assert arbiter.calls > 0


# 50. Distinct-addition marker overrides an arbiter's same-idea merge --
# reached through the REAL take-grouping/idea-equivalence chain.
#
# Offline audit of RAW 33432104336 (not a D-038 code regression -- see
# docs/CUTSELL_DECISIONS.md's D-038 entry): the live semantic-equivalence
# arbiter confirmed a "otro sintoma..." ("ANOTHER symptom...") mention as
# the same idea as an earlier, unrelated pimples mention, collapsing both
# into one retry contest and discarding one entirely. The speaker's own
# "this is a different/additional point" discourse marker is stronger,
# more general evidence than a topical-similarity arbiter verdict --
# take_grouping_provider.reconcile_semantic_idea_equivalence's own
# distinct-addition guard (general, no Video00 phrase hardcoded) now
# overrides exactly this shape, proven here through the real chain rather
# than only at the unit level (test_cutsell_semantic_idea_equivalence_
# grouping.py's own dedicated tests).

def test_distinct_addition_marker_prevents_a_real_chain_false_merge():
    first_mention = _take("first", 0.0, 3.0, "También tenía manchas rojas en la piel del brazo.", complete=True)
    another_mention = _take(
        "another", 5.0, 8.0,
        "Otro síntoma que noté fueron manchas rojas en la piel de la pierna.",
        complete=True,
    )

    draft, equivalence_diag, arbiter = _run_core(
        (first_mention, another_mention), oracle_pairs={("first", "another")},
    )

    assert arbiter.calls == 1  # the arbiter WAS asked and DID confirm same-idea
    assert equivalence_diag["status"] == "checked_no_merge"
    assert len(equivalence_diag["distinct_addition_blocked"]) == 1
    # Both survive as two independent, uncontested deliveries -- never
    # forced into one retry contest, so neither is silently discarded.
    assert _kept(draft) == {"first", "another"}
    assert _discarded(draft) == set()


# 51. Claim granularity (D-040): a winner that preserves the CORE claim but
# drops a discarded sibling's merely-supporting reason clause must not
# block Freeze -- reached through the REAL take-grouping/idea-equivalence/
# take-judge/coherence chain, reproducing RAW 33448261223's own false
# positive (the winning realization's phrasing matched the Human Gold
# reference verbatim; the CRITICAL_CLAIM_LOST finding was wrong).

def test_core_claim_preserved_supporting_reason_dropped_does_not_block_freeze():
    winner_text = (
        "Nunca se nos ocurrio hacer un chequeo de la tiroides, "
        "porque cada ano me hacia dos examenes normales."
    )
    loser_text = (
        "Nunca se nos ocurrio hacer un chequeo de la tiroides por sonografia "
        "porque siempre en mis examenes salia funcionando perfectamente."
    )
    winner = _take("winner", 0.0, 3.0, winner_text, complete=True)
    loser = _take("loser", 4.0, 7.0, loser_text, complete=False)

    draft, _, _ = _run_core((winner, loser), oracle_pairs={("winner", "loser")})

    assert _kept(draft) == {"winner"}
    assert _discarded(draft) == {"loser"}
    diag = draft.diagnostics["final_story_coherence_validation"]
    assert diag["lost_critical_claims"] == []
    assert diag["freeze_blocked"] is False


# 52. Fragment-provenance-aware coverage (D-046 FIX A): a retry-family
# winner that survives Selection through the REAL take-grouping/idea-
# equivalence/take-judge/coherence chain, then gets physically split
# afterward (simulating post_selection_interior_gap_trim's real production
# behavior -- a legitimate Boundary-flavored trim that divides an already-
# selected clip into two fragments, each carrying D-036's existing
# `parent_semantic_clip_id` provenance back to the original winner), must
# not be misreported as having vanished. Reproduces D-045 Case A's exact
# false-positive shape with generic vocabulary (no Video00 phrase/clip_id
# hardcoded): before the fix, canonical_edit_plan.py/final_story_
# coherence_validation.py's exact clip_id equality against draft.selected
# found neither the original id nor either fragment and wrongly reported
# `coverage_status: "missing"` + a false `missing_idea_coverage` freeze
# block for an idea whose winning content was, in fact, fully intact.

def test_split_winner_fragments_still_satisfy_idea_coverage_through_the_real_chain():
    take_a = _take("take_a", 0.0, 6.0, "Empece a notar cambios despues de mi rutina diaria.", complete=True)
    take_b = _take("take_b", 7.0, 12.0, "Empece a notar cambios despues de mi rutina diaria completa.", complete=True)

    draft, _, _ = _run_core((take_a, take_b), oracle_pairs={("take_a", "take_b")})
    kept_ids = _kept(draft)
    assert len(kept_ids) == 1  # Best Take resolved a single winner as usual
    winner_id = next(iter(kept_ids))
    loser_id = "take_a" if winner_id == "take_b" else "take_b"
    assert _discarded(draft) == {loser_id}
    assert draft.diagnostics["final_story_coherence_validation"]["missing_idea_coverage"] == []

    # Simulate the real post_selection_interior_gap_trim hook: the winner
    # is removed from draft.selected and replaced by two fragments whose
    # `parent_semantic_clip_id` names it -- exactly what that hook stamps
    # in production (see its own D-046 FIX A changes).
    winner_clip = next(clip for clip in draft.selected if clip.clip_id == winner_id)
    midpoint = (winner_clip.start + winner_clip.end) / 2.0
    fragment_left = replace(
        winner_clip, clip_id=f"{winner_id}__psiglabc", end=midpoint,
        text="Empece a notar cambios", caption_text="Empece a notar cambios",
        parent_semantic_clip_id=winner_id,
    )
    fragment_right = replace(
        winner_clip, clip_id=f"{winner_id}__psigrdef", start=midpoint,
        text="despues de mi rutina.", caption_text="despues de mi rutina.",
        parent_semantic_clip_id=winner_id,
    )
    split_draft = replace(
        draft,
        selected=tuple(c for c in draft.selected if c.clip_id != winner_id) + (fragment_left, fragment_right),
    )

    # Re-run StoryValidator on the post-split draft, exactly as the real
    # pipeline does (post_selection_interior_gap_trim runs before it).
    revalidated = apply_final_story_coherence_validation(split_draft)
    diag = revalidated.diagnostics["final_story_coherence_validation"]
    assert diag["missing_idea_coverage"] == []
    assert diag["freeze_blocked"] is False

    plan = build_canonical_edit_plan(revalidated)
    assert any(
        idea.coverage_status == "complete" and idea.winning_clip_ids == (winner_id,)
        for idea in plan.ideas
    )


# 53. Distinct-addition guard refinement (D-048 FIX 1): a discourse marker
# ("Otro sintoma...") opening a high-specific-content-overlap retry of an
# earlier mention must not block a high-confidence same-idea merge --
# reached through the REAL take-grouping/idea-equivalence chain, not just
# take_grouping_provider.py's own unit tests. Generic vocabulary (facial/
# hand swelling), no Video00 phrase hardcoded -- reproduces D-047 Case 1's
# shape (same specific symptom AND location, marker as narrative framing
# only) rather than the founding D-039 incident's shape (a genuinely
# different body part), which stays blocked in this same module's own
# dedicated suite (test_cutsell_d048_fix1_distinct_addition_guard.py).

def test_distinct_addition_high_overlap_retry_merges_through_the_real_chain():
    monolith = _take(
        "monolith", 0.0, 5.0,
        "Empece a notar hinchazon en esta parte de la cara cerca de los ojos "
        "que yo pensaba que era cansancio.",
        complete=True,
    )
    retry = _take(
        "retry", 6.0, 11.0,
        "Otro sintoma fue hinchazon en esta parte de la cara cerca de los "
        "ojos que parecia cansancio.",
        complete=True,
    )

    draft, equivalence_diag, arbiter = _run_core((monolith, retry), oracle_pairs={("monolith", "retry")})

    assert equivalence_diag["status"] == "applied"
    assert equivalence_diag.get("distinct_addition_blocked", []) == []
    assert len(_kept(draft)) == 1  # one winning realization, not two "ideas"


# 54. Claim-coverage self-source trap (D-048 FIX 2): a richer diagnosis/
# treatment realization must not be discarded in favor of a vague,
# source-exclusive, incidental-temporal-aside candidate whose only
# "critical" claim is a bare negation riding on a bare year -- reached
# through the REAL take-grouping/idea-equivalence/take-judge/claim-coverage
# chain. Generic vocabulary, no Video00 phrase hardcoded -- reproduces
# D-047 Case 2's exact shape (the group's only critical claim is source-
# exclusive to the thin candidate).

def test_claim_coverage_self_source_trap_richer_winner_survives_through_the_real_chain():
    sibling = _take(
        "sibling", 0.0, 4.0,
        "Tuve problemas estomacales a un tiempo en donde se me hizo una "
        "endoscopia y me diagnosticaron.",
        complete=False,
    )
    rich = _take(
        "rich", 5.0, 9.0,
        "Tuve problemas de digestion en donde me hicieron una endoscopia y "
        "dijeron que tenia gastritis y me mandaron tres meses con pastillas.",
        complete=True,
    )
    vague = _take(
        "vague", 10.0, 14.0,
        "Tuve problemas de estomago en una temporada, en 2023, no se por "
        "que me pasaba eso.",
        complete=False,
    )

    draft, _, _ = _run_core(
        (sibling, rich, vague),
        oracle_pairs={("sibling", "rich"), ("sibling", "vague"), ("rich", "vague")},
    )

    assert _kept(draft) == {"rich"}
    assert _discarded(draft) == {"sibling", "vague"}
    diag = draft.diagnostics.get("claim_coverage_best_take") or {}
    assert diag.get("overrides", []) == []
    assert len(diag.get("suppressed_incidental_overrides", [])) == 1
    assert diag["suppressed_incidental_overrides"][0]["suppressed_new_winner_clip_id"] == "vague"
