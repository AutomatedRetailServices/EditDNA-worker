"""D-046 FIX B -- good subspan preservation under a borderline attempt merge.

D-045 Case B: "Era como un rush, una alergia." was reconstructed as its own
delivery attempt in the last passing run, but a run-to-run ASR/timing
shift closed the ~0.76s gap between it and the following (bad) monolith
just enough that AttemptReconstructor fused them into one physical
attempt. The fused attempt correctly lost its Best Take contest as a
whole, destroying the good subspan fused inside it.

The fix does not change the merge decision itself (tightening the
boundary rule was evaluated and rejected -- too easily triggered by
ordinary multi-sentence delivery, i.e. a global fragment explosion, not a
targeted fix). Instead it ADDITIONALLY reconstructs the two sides of any
borderline internal gap (a real pause, both sides already independently
`complete_idea`) as extra standalone candidates, letting the existing
IdeaClusterer/BestTake/ClaimCoverage machinery decide their fate -- the
merged attempt itself is always still returned unchanged too.
"""
from cutsell_worker.attempt_reconstruction import (
    preserved_subspan_candidates,
    reconstruct_delivery_attempts,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, WholeVideoContext


def take(clip_id, text, start, end, *, source="src", complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def context():
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator recording",
            dominant_style="talking_head",
            creator_intent="natural delivery",
            events=(),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


_GOOD_MICRO_FRAGMENT = "También me salían espinillas, era como un rush, una alergia."
# Deliberately does NOT share its opening content tokens with the micro
# fragment above -- this isolates the new borderline-gap heuristic from
# the pre-existing, unrelated `_restart_evidence` ("lexical_restart")
# boundary rule, which fires independently of gap size whenever both
# sides open with the same two content words.
_BAD_MONOLITH = (
    "En esta parte de aquí, detrás de la oreja y todo el cuello, me salían "
    "espinillas que yo pensaba que era alergia, pero eran como espinillas de "
    "personas con problemas hormonales."
)


def _reconstruct_with_gap(gap_sec: float, *, left_complete=True, right_complete=True):
    left = take("left", _GOOD_MICRO_FRAGMENT, 0.0, 10.0, complete=left_complete)
    right = take("right", _BAD_MONOLITH, 10.0 + gap_sec, 28.0, complete=right_complete)
    attempts, diagnostics = reconstruct_delivery_attempts((left, right), context())
    preserved, audit = preserved_subspan_candidates((left, right), diagnostics)
    return attempts, diagnostics, preserved, audit


# --- independent micro-fragment must survive (the actual Case B shape) ---

def test_independent_micro_fragment_survives_a_borderline_merge():
    attempts, diagnostics, preserved, audit = _reconstruct_with_gap(0.76)

    assert len(attempts) == 1  # still fused into one attempt, unchanged
    assert "rush" in attempts[0].text

    assert len(preserved) == 2
    texts = {c.text for c in preserved}
    assert _GOOD_MICRO_FRAGMENT in texts
    assert _BAD_MONOLITH in texts
    assert len(audit) == 1
    assert audit[0]["gap_sec"] == 0.76


# --- just below / at / just above the preservation threshold ------------

def test_just_below_preservation_threshold_is_not_preserved():
    _, _, preserved, _ = _reconstruct_with_gap(0.50)
    assert preserved == ()


def test_exactly_at_preservation_threshold_is_preserved():
    _, _, preserved, _ = _reconstruct_with_gap(0.55)
    assert len(preserved) == 2


def test_just_above_preservation_threshold_is_preserved():
    _, _, preserved, _ = _reconstruct_with_gap(0.60)
    assert len(preserved) == 2


# --- +/-50ms / +/-100ms / +/-250ms jitter around the observed 0.76s gap --

def test_plus_minus_50ms_around_observed_gap_both_preserved():
    for gap in (0.71, 0.81):
        _, _, preserved, _ = _reconstruct_with_gap(gap)
        assert len(preserved) == 2, gap


def test_plus_minus_100ms_around_observed_gap_both_preserved():
    for gap in (0.66, 0.86):
        _, _, preserved, _ = _reconstruct_with_gap(gap)
        assert len(preserved) == 2, gap


def test_plus_minus_250ms_around_observed_gap_straddles_the_threshold():
    # 0.76 - 0.25 = 0.51s -- below the 0.55s floor, not preserved.
    _, _, below, _ = _reconstruct_with_gap(0.51)
    assert below == ()
    # 0.76 + 0.25 = 1.01s -- still below the 1.20s merge threshold, preserved.
    _, _, above, _ = _reconstruct_with_gap(1.01)
    assert len(above) == 2


def test_same_words_shifted_asr_timings_deterministic_within_preserve_zone():
    # Simulates run-to-run ASR jitter that shifts word timestamps slightly
    # but stays solidly within the preserve zone -- the outcome must not
    # flap based on immaterial sub-threshold timing noise.
    outcomes = [len(_reconstruct_with_gap(gap)[2]) for gap in (0.70, 0.72, 0.75, 0.78, 0.80)]
    assert outcomes == [2, 2, 2, 2, 2]


# --- a valid true continuation must still merge with nothing extra ------

def test_valid_true_continuation_is_not_treated_as_a_borderline_pair():
    # Right side is an incomplete continuation, not an independently
    # complete idea -- this is ordinary continuation speech, not two
    # fused ideas, and must not spuriously gain extra candidates.
    _, _, preserved, _ = _reconstruct_with_gap(0.76, right_complete=False)
    assert preserved == ()


def test_gap_at_or_above_merge_threshold_never_needs_preservation():
    # A gap this large already gets its own real boundary (two attempts),
    # so there is nothing to "preserve" -- both original candidates are
    # already independently present in `attempts`.
    attempts, diagnostics, preserved, _ = _reconstruct_with_gap(1.25)
    assert len(attempts) == 2
    assert preserved == ()


# --- no fragment explosion / no duplicate final realizations ------------

def test_no_fragment_explosion_with_multiple_internal_gaps_in_one_bucket():
    a = take("a", "Primera idea completa sobre este tema en particular.", 0.0, 5.0)
    b = take("b", "Segunda idea completa totalmente distinta de la anterior.", 5.6, 10.0)
    c = take("c", "Tercera idea completa que tampoco repite las anteriores.", 10.7, 15.0)
    d = take("d", "Cuarta idea completa para cerrar toda la explicacion.", 15.6, 20.0)

    attempts, diagnostics = reconstruct_delivery_attempts((a, b, c, d), context())
    assert len(attempts) == 1  # all four fused into one bucket (each gap < 1.20s)

    preserved, audit = preserved_subspan_candidates((a, b, c, d), diagnostics)

    # At most ONE split point per bucket -- exactly two extra candidates,
    # never one per internal gap (which would be 3 splits / 4+ extras here).
    assert len(preserved) == 2
    assert len(audit) == 1


def test_preserved_candidates_are_not_duplicates_of_the_fused_attempt():
    attempts, diagnostics, preserved, _ = _reconstruct_with_gap(0.76)

    fused_clip_id = attempts[0].clip_id
    fused_text = attempts[0].text
    for candidate in preserved:
        assert candidate.clip_id != fused_clip_id
        assert candidate.text != fused_text
    # The two preserved candidates are themselves distinct from each other.
    assert preserved[0].clip_id != preserved[1].clip_id
    assert preserved[0].text != preserved[1].text


def test_preserved_subspan_candidates_returns_nothing_for_an_unsplit_pool():
    # A pool with no merges at all (each take its own attempt) has nothing
    # to preserve -- the function must not invent candidates out of thin
    # air just because member_clip_ids happens to have length 1 per row.
    single = take("solo", "Una unica idea independiente y completa.", 0.0, 4.0)
    attempts, diagnostics = reconstruct_delivery_attempts((single,), context())
    preserved, audit = preserved_subspan_candidates((single,), diagnostics)
    assert preserved == ()
    assert audit == []
