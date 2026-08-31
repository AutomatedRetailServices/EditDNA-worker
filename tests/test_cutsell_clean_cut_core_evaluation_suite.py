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
level, and why.

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

from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, MediaSignals, SCHEMA_VERSION
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
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


def _run_core(takes: tuple[CandidateTake, ...], *, oracle_pairs=frozenset()):
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
    draft = apply_final_story_coherence_validation(draft, semantic_equivalence_arbiter=arbiter)
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
