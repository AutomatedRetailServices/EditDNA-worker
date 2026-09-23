from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_alternate_integrity import suppress_stranded_hybrid_alternates
from cutsell_worker.semantic_best_take_integrity import _prefer_clear_nonfailed_peer


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        complete_idea=complete,
        signals=MediaSignals("src", float(start), float(end)),
    )


def test_video00_failed_local_retry_yields_to_one_clear_keep_peer():
    failed_local = _take(
        "failed-local",
        235.85,
        250.05,
        "Tuve problemas estomacales a un tiempo, en donde se me hizo una endoscopía y me diagnosticaron con... Tuve problemas de estómago en una temporada.",
    )
    clean_peer = _take(
        "clean-peer",
        258.57,
        269.37,
        "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo, pero tenía gastritis y me mandaron tres meses con pastillas.",
    )

    preferred = _prefer_clear_nonfailed_peer(
        (failed_local, clean_peer),
        {
            "failed-local": ("failed", 0.85),
            "clean-peer": ("keep", 0.90),
        },
        "failed-local",
    )

    assert preferred == "clean-peer"


def test_semantic_tie_break_remains_fail_open_with_two_clear_peers():
    failed_local = _take("failed", 0, 5, "broken retry")
    peer_a = _take("a", 6, 12, "first usable delivery")
    peer_b = _take("b", 13, 19, "second usable delivery")

    preferred = _prefer_clear_nonfailed_peer(
        (failed_local, peer_a, peer_b),
        {
            "failed": ("failed", 0.90),
            "a": ("keep", 0.90),
            "b": ("winner", 0.92),
        },
        "failed",
    )

    assert preferred is None


def test_video04_open_alternate_and_corrected_suffix_are_suppressed_before_winner():
    prefix = _take(
        "open-alt",
        24.70,
        36.58,
        "somehow through electromagnetic forces, this little doodad is honestly a pretty competitive price and for the peace of mind that you get to know that your pet is truly having the most hygienic fountain experience,",
    )
    suffix = _take("suffix-alt", 38.66, 39.26, "priceless.")
    winner = _take(
        "winner",
        41.66,
        57.40,
        "And through electromagnetic forces, it makes this spin really fast. It shoots the water up this pipe into here and it flows. It is a really great price, especially for the peace of mind you get from this, which is priceless. Check it out.",
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (prefix, suffix, winner),
        (
            ("open-alt", "alternate", 0.80),
            ("suffix-alt", "alternate", 0.75),
            ("winner", "winner", 0.95),
        ),
    )

    assert kept == (winner,)
    assert {take.clip_id for take in removed} == {"open-alt", "suffix-alt"}
    reasons = {item["clip_id"]: item["reason"] for item in diagnostics}
    assert reasons["open-alt"] == "semantic_alternate_open_retry_before_winner"
    assert reasons["suffix-alt"] == "semantic_alternate_corrected_suffix_repeated_by_winner"


def test_unique_open_alternate_without_substantive_overlap_remains_fail_open():
    unique = _take(
        "unique",
        0.0,
        10.0,
        "This paragraph explains a completely separate hygiene benefit for the pet owner,",
    )
    winner = _take(
        "winner",
        15.0,
        28.0,
        "The magnetic motor spins the fountain and the product is a great price. Check it out.",
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (unique, winner),
        (("unique", "alternate", 0.90), ("winner", "winner", 0.96)),
    )

    assert kept == (unique, winner)
    assert removed == ()
    assert diagnostics == ()


def test_short_punchline_is_not_removed_without_removed_open_sibling():
    punchline = _take("punchline", 10.0, 10.7, "Priceless.")
    winner = _take(
        "winner",
        13.0,
        25.0,
        "The complete final delivery ends by saying the result is priceless.",
    )

    kept, removed, diagnostics = suppress_stranded_hybrid_alternates(
        (punchline, winner),
        (("punchline", "alternate", 0.80), ("winner", "winner", 0.95)),
    )

    assert kept == (punchline, winner)
    assert removed == ()
    assert diagnostics == ()
