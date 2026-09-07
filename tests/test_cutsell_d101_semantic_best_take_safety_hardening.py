"""D-101 FIX: SEMANTIC-BEST-TAKE SAFETY HARDENING (both proven root causes).

The P0 forensic (`docs/CUTSELL_FORENSIC_HEREDITARY_CANCER_CLUSTER_D101.md`)
found two independent defects inside `pipeline.py::_semantic_best_take`,
the SAME existing authority both times:

  ROOT CAUSE #1: `single_semantic_winner` trusted a lone Hybrid/Gemini
  "winner" label unconditionally, with none of the safety checks the
  `len(winners) != 1` branch already has -- letting a per-window
  classification error discard a realization carrying unique required
  meaning while keeping an unrelated one.

  ROOT CAUSE #2: `delivery_tie_break_among_survivors` picked among
  survivors by raw DeliveryScorer rank alone, with no awareness that one
  survivor could be an incomplete, literal content subset of another --
  letting a short prefix of a fuller passage outscore (and so discard)
  the complete passage that contains and completes it.

Both fixes are SAFETY VETOES only, never a return to maximum semantic
coverage: they never manufacture a composite and never restore a losing
realization merely because it carries extra low-value/supporting content.
Generic (English) fixtures only -- no Video00 wording, per the module
docstring conventions already used by every other test file in this
family (D-082/D-063/D-100).

`_semantic_best_take`'s return value flows DIRECTLY into
`TakeGroup.selected_clip_id` at its one production call site in
`build_flow_b_draft` with no further override before that assignment --
so proving behavior at this function IS proving it through the
authoritative selection path, the same precedent D-082's own test suite
(`test_cutsell_d082_non_decisive_semantic_fallback.py`) already
established for this exact function.
"""
from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import (
    _exclude_incomplete_subset_losers,
    _is_incomplete_content_subset,
    _semantic_best_take,
    _single_winner_safety_veto,
)


def take(clip_id: str, text: str, *, complete_idea: bool | None = True, start: float = 0.0) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 4.0,
        text=text,
        complete_idea=complete_idea,
    )


def ranked(*pairs: tuple[str, float]) -> tuple[RankedTake, ...]:
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


# ---------------------------------------------------------------------------
# ROOT CAUSE #1: single_semantic_winner safety veto
# ---------------------------------------------------------------------------

_NO_DIAGNOSIS = "The technician walked us through the whole intake process from start to finish."
_HAS_DIAGNOSIS = "The specialist confirmed she was diagnosed with a rare autoimmune condition."


def test_a_positive_control_unsafe_single_winner_is_vetoed_and_required_meaning_survives():
    """BEFORE (pre-D-101): `single_semantic_winner` would return "no_diagnosis"
    unconditionally, losing the unique required diagnosis meaning. AFTER:
    the veto fires and the diagnosis-carrying candidate wins through the
    fallback ladder's own CRITICAL_COVERAGE_DOMINANCE step -- required
    meaning survives the FULL authoritative selection path, not just the
    local veto helper."""
    no_diagnosis = take("no_diagnosis", _NO_DIAGNOSIS)
    has_diagnosis = take("has_diagnosis", _HAS_DIAGNOSIS, start=5.0)
    decisions = {"no_diagnosis": ("winner", 0.95), "has_diagnosis": ("alternate", 0.70)}
    r = ranked(("no_diagnosis", 10.0), ("has_diagnosis", 5.0))

    selected, preferred, reason = _semantic_best_take(
        (no_diagnosis, has_diagnosis), decisions, "no_diagnosis", r,
    )
    assert reason != "single_semantic_winner"
    assert selected == "has_diagnosis"


def test_a_veto_helper_reports_the_specific_reason_directly():
    no_diagnosis = take("no_diagnosis", _NO_DIAGNOSIS)
    has_diagnosis = take("has_diagnosis", _HAS_DIAGNOSIS, start=5.0)
    veto = _single_winner_safety_veto("no_diagnosis", (no_diagnosis, has_diagnosis), None)
    assert veto == "winner_missing_unique_critical_claim"


def test_a_negative_control_genuine_safe_single_winner_fast_path_unaffected():
    """A single winner whose own text ALREADY carries every CRITICAL claim
    in the family must not be vetoed -- no unnecessary keep-both/fallback
    behavior, byte-identical to pre-D-101 for the common, safe case."""
    winner = take("winner", _HAS_DIAGNOSIS)
    alternate = take("alternate", "It was a really long day at the clinic.", start=5.0)
    decisions = {"winner": ("winner", 0.95), "alternate": ("failed", 0.5)}
    r = ranked(("winner", 10.0), ("alternate", 3.0))

    selected, preferred, reason = _semantic_best_take((winner, alternate), decisions, "winner", r)
    assert reason == "single_semantic_winner"
    assert selected == "winner"
    assert _single_winner_safety_veto("winner", (winner, alternate), None) is None


def test_a_extra_supporting_only_content_never_triggers_the_veto():
    """The veto is a safety net, not a return to maximum coverage: a losing
    realization that merely repeats the winner's own CRITICAL content plus
    extra SUPPORTING/low-value filler must not veto the fast path."""
    winner = take("winner", _HAS_DIAGNOSIS)
    loser = take(
        "loser",
        "It was a really long day at the clinic before the specialist confirmed "
        "she was diagnosed with a rare autoimmune condition.",
        start=5.0,
    )
    decisions = {"winner": ("winner", 0.95), "loser": ("failed", 0.6)}
    r = ranked(("winner", 10.0), ("loser", 5.0))
    selected, preferred, reason = _semantic_best_take((winner, loser), decisions, "winner", r)
    assert reason == "single_semantic_winner"
    assert selected == "winner"


def test_a_winner_carrying_delete_recommended_evidence_is_vetoed():
    a = take("a", "Prices went up a little this year across the board.")
    b = take("b", "The refund policy covers a full year from purchase.", start=5.0)
    decisions = {"a": ("winner", 0.95), "b": ("failed", 0.5)}
    veto = _single_winner_safety_veto("a", (a, b), {"a": True})
    assert veto == "winner_carries_delete_recommended_evidence"


def test_a_winner_explicitly_incomplete_is_vetoed():
    a = take("a", "So then after that we", complete_idea=False)
    b = take("b", "So then after that we tried a completely different approach entirely.", start=5.0)
    veto = _single_winner_safety_veto("a", (a, b), None)
    assert veto == "winner_is_explicitly_incomplete"


def test_a_contradicting_members_are_vetoed():
    a = take("a", "The results came back with no signs of infection at all.")
    b = take("b", "The results came back showing no infection was ever present initially.", start=5.0)
    # Craft a genuine number mismatch (contradiction primitive fires on
    # differing number SETS, see contradiction_signal.detect_text_contradiction).
    a2 = take("a2", "The discount applies at a rate of 5 percent storewide.")
    b2 = take("b2", "The discount applies at a rate of 5 percent, though loyalty members get 10 percent.", start=5.0)
    from cutsell_worker.contradiction_signal import any_pair_contradicts
    assert any_pair_contradicts([a2.text, b2.text])
    veto = _single_winner_safety_veto("a2", (a2, b2), None)
    assert veto == "members_contradict"


# ---------------------------------------------------------------------------
# ROOT CAUSE #2: incomplete strict prefix/subset never defeats its own
# complete realization purely on delivery tie-break score
# ---------------------------------------------------------------------------

_SHORT = "The warranty covers repairs for two years."
_LONG = "The warranty covers repairs for two years but batteries come with a separate manufacturer guarantee."


def test_b_positive_control_incomplete_subset_cannot_win_the_tie_break_purely_on_score():
    short = take("short", _SHORT)
    long_ = take("long", _LONG, start=5.0)
    decisions = {"short": ("keep", 0.5), "long": ("keep", 0.5)}
    # The short prefix scores HIGHER -- pre-D-101 this alone would win.
    r = ranked(("short", 10.0), ("long", 5.0))
    selected, preferred, reason = _semantic_best_take((short, long_), decisions, "short", r)
    assert selected == "long"
    assert reason == "delivery_tie_break_among_survivors"


def test_b_helper_detects_the_literal_contiguous_subset_relationship():
    short = take("short", _SHORT)
    long_ = take("long", _LONG, start=5.0)
    assert _is_incomplete_content_subset(short, long_) is True
    assert _is_incomplete_content_subset(long_, short) is False


def test_b_negative_control_1_two_complete_alternatives_longer_does_not_automatically_win():
    """Neither candidate's text is a literal contiguous subset of the
    other -- protection must not apply; the HIGHER-scoring candidate wins
    on ordinary delivery merit, whichever one that is."""
    a = take("a", "We tested three different formulas before landing on this one.")
    b = take("b", "This formula uses natural ingredients sourced locally from small farms nearby.", start=5.0)
    assert _is_incomplete_content_subset(a, b) is False
    assert _is_incomplete_content_subset(b, a) is False
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    selected, _p, reason = _semantic_best_take((a, b), decisions, "a", ranked(("a", 10.0), ("b", 5.0)))
    assert selected == "a"
    assert reason == "delivery_tie_break_among_survivors"


def test_b_negative_control_2_complementary_realizations_never_collapse_as_prefix_retry():
    c = take("c", "The jacket comes in three different colors for everyone.")
    d = take("d", "The jacket is also machine washable and very durable.", start=5.0)
    assert _is_incomplete_content_subset(c, d) is False
    assert _is_incomplete_content_subset(d, c) is False
    decisions = {"c": ("keep", 0.5), "d": ("keep", 0.5)}
    selected, _p, reason = _semantic_best_take((c, d), decisions, "d", ranked(("c", 3.0), ("d", 10.0)))
    assert selected == "d"
    assert reason == "delivery_tie_break_among_survivors"


def test_b_negative_control_3_same_opening_different_idea_no_subset_inferred():
    e = take("e", "When my contract ended I spoke with my doctor about several options.")
    f = take("f", "When my contract ended I decided to travel instead for a while.", start=5.0)
    assert _is_incomplete_content_subset(e, f) is False
    assert _is_incomplete_content_subset(f, e) is False
    decisions = {"e": ("keep", 0.5), "f": ("keep", 0.5)}
    selected, _p, reason = _semantic_best_take((e, f), decisions, "e", ranked(("e", 10.0), ("f", 3.0)))
    assert selected == "e"
    assert reason == "delivery_tie_break_among_survivors"


def test_b_negative_control_4_contradiction_suppresses_the_protection_longer_not_automatic():
    """`long` textually contains `short` but they disagree on a number
    (contradiction) -- the subset protection must be suppressed entirely,
    never assume the fuller candidate is automatically the safe one. The
    existing step-5 contradiction guard actually catches this pair first,
    keeping the safe pre-D-101 default rather than either candidate being
    forced to win."""
    short = take("short", "The warranty covers repairs for 2 years.")
    long_ = take("long", "The warranty covers repairs for 2 years but batteries are only covered for 1 year.", start=5.0)
    assert _is_incomplete_content_subset(short, long_) is True
    from cutsell_worker.contradiction_signal import any_pair_contradicts
    assert any_pair_contradicts([short.text, long_.text])
    decisions = {"short": ("keep", 0.5), "long": ("keep", 0.5)}
    selected, _p, reason = _semantic_best_take(
        (short, long_), decisions, "short", ranked(("short", 10.0), ("long", 5.0)),
    )
    # Never automatically "long wins" -- the contradiction leaves the
    # family at its safe, pre-existing fallback instead.
    assert selected == "short"
    assert reason == "unresolved_contradiction"


def test_b_exclusion_helper_fails_open_never_excludes_everyone():
    short = take("short", _SHORT)
    long_ = take("long", _LONG, start=5.0)
    by_id = {"short": short, "long": long_}
    # Even if (hypothetically) both directions matched, the helper must
    # never return an empty pool.
    survivors = _exclude_incomplete_subset_losers(["short", "long"], by_id)
    assert survivors  # non-empty
    assert "long" in survivors


def test_b_richer_but_independent_content_still_competes_normally():
    """A short, COMPLETE candidate that is not textually contained in the
    longer one competes purely on delivery merit -- no subset relation,
    no protection, whichever already-existing behavior applies."""
    a = take("a", "The team spent the whole afternoon rehearsing the new routine.", complete_idea=True)
    b = take("b", "The crew filmed the scene twice before moving to the next location.", start=5.0, complete_idea=True)
    assert _is_incomplete_content_subset(a, b) is False
    decisions = {"a": ("keep", 0.5), "b": ("keep", 0.5)}
    selected, _p, reason = _semantic_best_take((a, b), decisions, "b", ranked(("a", 3.0), ("b", 9.0)))
    assert selected == "b"
    assert reason == "delivery_tie_break_among_survivors"
