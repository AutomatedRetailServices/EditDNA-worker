"""D-284 (RAW #119 audit): internal repeat trimming is limited to the
EXACT trailing case and misses a repeat cycle that continues into a
third, truncated re-recitation of the same phrase.

Real RAW #119 evidence (Modal run 35756418346): the final selected clip
contained several consecutive near-identical deliveries of a short
closing phrase ("Por eso cuidate. Alimentate bien ... haz ejercicio.")
back to back inside one candidate. `internal_repeat_trim.py`'s own
docstring already describes exactly this failure class ("ASR can keep a
failed restart inside the same candidate as valid speech"), but its
detector (`_trailing_repeat_start`) required the SECOND occurrence of the
repeated phrase to BE the true end of the take (or within one token of
it) -- a take with one clean cycle, one repeated cycle, and a further
TRUNCATED third cycle was never trimmed at all, because the third cycle's
tokens count as "more content after the second occurrence."

Fix (cutsell_worker/internal_repeat_trim.py): the tail-adjacency gate now
also accepts a remainder that is itself a token-for-token PREFIX of the
same repeated phrase (a partial third cycle), not just an empty/near-empty
remainder. Both existing safety properties are preserved exactly: the
phrase must still reoccur verbatim (never ordinary topic-adjacent
repetition), and the SAME required corroboration (a nearby physical reset
event OR an immediate substantive following take) still gates every trim
-- this generalization only changes what counts as "the end", not whether
corroboration is required. Fixtures are generic (no Video00 text) --
a sales CTA repeated phrase, matching the existing test file's own style.
"""
from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.internal_repeat_trim import trim_internal_repeated_restarts
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _word(text: str, start: float, end: float) -> Word:
    return Word(text, start, end)


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator records a sales CTA and retries the phrase twice more",
            dominant_style="talking_head",
            creator_intent="deliver one clean CTA",
            events=tuple(events),
            edit_mode="sales",
            sales_intent=1.0,
            main_topic="CTA",
            product_or_subject="product",
            story_logic="keep one clean CTA and remove the repeated restart cycles",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _take_from_tokens(tokens: tuple[str, ...]) -> CandidateTake:
    words = []
    t = 43.78
    for token in tokens:
        words.append(_word(token, t, t + 0.20))
        t += 0.23
    return CandidateTake(
        clip_id="cta", source_asset_id="src", source_order=0,
        start=43.78, end=t + 0.20, text=" ".join(tokens), words=tuple(words),
        signals=MediaSignals("src", 43.78, t + 0.20),
    )


_CYCLE = ("you", "see", "the", "orange", "shopping", "cart")


def _reset_context(take: CandidateTake, restart_index: int) -> WholeVideoContext:
    restart = take.words[restart_index].start
    return _context(TemporalEvent(
        "src", restart + 0.2, restart + 0.4,
        "body_reset_candidate", 0.98, "creator physically resets while restarting the phrase",
    ))


# ---------------------------------------------------------------------------
# Positive control: cycle 1 (clean) + cycle 2 (exact repeat) + cycle 3
# (truncated re-recitation of the SAME phrase) -- corroborated by a
# physical reset at the second cycle's start.
# ---------------------------------------------------------------------------

def test_truncated_third_cycle_is_trimmed_back_to_the_first_clean_cycle():
    take = _take_from_tokens(_CYCLE + _CYCLE + _CYCLE[:3])
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _reset_context(take, 6))

    assert len(trimmed) == 1
    assert trimmed[0].text == "you see the orange shopping cart"
    assert diagnostics[0]["reason"] == "internal_repeated_restart_trim_partial_remainder"


def test_immediate_following_take_can_corroborate_the_partial_remainder_case():
    take = _take_from_tokens(_CYCLE + _CYCLE + _CYCLE[:2])
    following = CandidateTake(
        clip_id="next", source_asset_id="src", source_order=0,
        start=take.end + 0.5, end=take.end + 3.0,
        text="but if you do not see it they probably sold out",
    )
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take, following), _context())

    assert trimmed[0].text == "you see the orange shopping cart"
    assert diagnostics[0]["following_clip_id"] == "next"
    assert diagnostics[0]["reason"] == "internal_repeated_restart_trim_partial_remainder"


def test_existing_exact_trailing_case_is_unaffected():
    # Regression guard: the original tail-exact shape (already covered by
    # tests/test_cutsell_internal_repeat_trim.py) still reports the
    # original reason string, not the new one.
    filler = ("might", "want", "to", "grab", "and", "if")
    take = _take_from_tokens(_CYCLE + filler + _CYCLE)
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _reset_context(take, len(_CYCLE) + len(filler)))

    assert diagnostics[0]["reason"] == "internal_trailing_repeated_restart_trim"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def test_genuinely_new_content_after_the_repeat_is_never_discarded():
    take = _take_from_tokens(_CYCLE + _CYCLE + ("today", "only", "while", "supplies", "last"))
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _reset_context(take, 6))

    assert trimmed == (take,)
    assert diagnostics == ()


def test_partial_remainder_without_corroboration_fails_open():
    take = _take_from_tokens(_CYCLE + _CYCLE + _CYCLE[:3])
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _context())

    assert trimmed == (take,)
    assert diagnostics == ()


def test_remainder_that_only_partially_matches_the_phrase_is_not_trimmed():
    # The remainder must be a genuine PREFIX of the repeated phrase, not
    # merely share some words with it.
    take = _take_from_tokens(_CYCLE + _CYCLE + ("cart", "shopping", "orange"))
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _reset_context(take, 6))

    assert trimmed == (take,)
    assert diagnostics == ()
