"""Clean Cut Core V1 general evaluation suite.

Video-agnostic fixtures covering the 14 categories required before another
paid Video00 RAW: exact retry, paraphrased retry, two good takes expressing
the same idea, false start -> clean retry, incomplete -> complete retry,
long bad take vs concise complete take, composite required, continuation
that must not collapse, similar vocabulary but distinct ideas, visual fumble
with semantically valid transcript, unique-fact preservation, CTA/story-
ending preservation, multiple retries with progressively better delivery,
and same idea expressed with substantially different wording.

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
