from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.semantic_fragment_guard import remove_semantic_fragment_debris
from cutsell_worker.session_boundaries import safe_group_takes_by_sessions


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


def test_related_video04_story_paragraphs_are_not_one_retry_family():
    takes = (
        _take(
            "mold-hook",
            0.0,
            6.36,
            "Your pet fountain probably has mold growing inside the pump. You can access every single piece that the water touches in this fountain.",
        ),
        _take(
            "internal-pump",
            6.54,
            17.50,
            "There is no internal pump that you have to kind of worry is there mold or bacteria building up in there. The magic happens from this little doodad, which is magnetic. This little doodad is magnetic and somehow through",
        ),
        _take(
            "self-correction",
            24.70,
            39.26,
            "somehow through electromagnetic forces, this little doodad is honestly a pretty competitive price and for the peace of mind that you get to know that your pet is truly having the most hygienic fountain experience, worthless. Oh, priceless.",
        ),
        _take(
            "mechanism-close",
            41.66,
            57.40,
            "And through electromagnetic forces, it makes this spin really fast. It shoots the water up this pipe into here and it flows. I will have this linked below. It is a really great price, especially for the peace of mind you get from this, which is priceless. Check it out. You will love it. Thank me later.",
        ),
    )

    result = safe_group_takes_by_sessions(None, takes, None)

    assert len(result.groups) == 4
    assert all(len(group) == 1 for group in result.groups)


def test_true_video00_reformulated_retry_still_groups():
    first = _take(
        "first",
        82.40,
        90.36,
        "Al terminar mi contrato, hablé con mi ginecóloga y le pedí todos los test que ella pudiera imaginarse o que me pudiera indicar.",
    )
    retry = _take(
        "retry",
        95.58,
        107.48,
        "Al terminar mi contrato, cambié de ginecóloga y le pedí que me hiciera un test de todo lo que ella se pudiera imaginar y me pudiese indicar. Ahí me mandó a hacer sonografías.",
    )

    result = safe_group_takes_by_sessions(None, (first, retry), None)

    assert len(result.groups) == 1
    assert set(result.groups[0]) == {"first", "retry"}


def test_video02_unique_story_paragraphs_do_not_become_alternates_of_each_other():
    intro = _take(
        "intro",
        0.62,
        35.44,
        "You know those videos so you can tell that the creators are actually having fun. My favorite second video is with Blaine O'Connor and we decided to go with the classic delivery pizza guy.",
    )
    outfit = _take(
        "outfit",
        36.82,
        54.40,
        "An outfit that I cut the sleeves out to make it more sexy and at the end we went all in. We ordered a pizza and we could barely keep a straight face while filming.",
    )
    fun = _take(
        "fun",
        66.10,
        76.38,
        "The thing is that this video is so sexy and so funny. I think it was just as fun to make and to do the action and everything.",
    )

    result = safe_group_takes_by_sessions(None, (intro, outfit, fun), None)

    assert len(result.groups) == 3
    assert {group[0] for group in result.groups} == {"intro", "outfit", "fun"}


def test_failed_micro_fragment_at_medium_confidence_is_removed():
    broken = _take("tired", 6.33, 7.61, "you're tired", complete=False)

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("tired", "failed", 0.75),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_failed_micro_fragment"


def test_failed_repetition_survives_provider_variance_but_not_final_cleanup():
    broken = _take(
        "non-gmo",
        16.84,
        22.84,
        "non-gmo non-gmo non-gmo gluten-free and be they're not eating if they're not eating",
        complete=False,
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("non-gmo", "failed", 0.70),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_failed_repetition_pathology"


def test_low_confidence_alternate_with_exact_repeated_clause_is_not_final_delivery():
    broken = _take(
        "eating-loop",
        22.84,
        27.54,
        "if they're not eating if they're not eating health already worried that their kid",
        complete=False,
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("eating-loop", "alternate", 0.65),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_nonwinner_repetition_pathology"


def test_high_confidence_failed_short_self_talk_is_removed():
    broken = _take("character", 109.79, 111.05, "trying to say in character", complete=False)

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("character", "failed", 0.92),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_failed_short_fragment"


def test_winner_is_never_removed_only_for_repetition_shape():
    rhetorical = _take(
        "winner",
        0.0,
        5.0,
        "day by day day by day this gets easier for every family",
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (rhetorical,),
        (("winner", "winner", 0.99),),
    )

    assert kept == (rhetorical,)
    assert removed == ()
    assert diagnostics == ()


def test_short_incomplete_tail_is_not_merged_into_good_video03_middle():
    good = _take(
        "good-middle",
        26.35,
        44.31,
        "A mí me sorprende la capacidad que tiene la piel de recuperarse, pero más me siguen sorprendiendo estos productos. Esta crema es mágica, tiene componentes que de verdad te protegen, te reparan.",
        complete=True,
    )
    bad_tail = _take(
        "bad-tail",
        44.33,
        47.67,
        "la barrera cutánea te la te hace como",
        complete=False,
    )

    attempts, diagnostics = reconstruct_delivery_attempts((good, bad_tail), None)

    assert len(attempts) == 2
    assert attempts[0].clip_id == "good-middle"
    assert attempts[1].clip_id == "bad-tail"
    assert diagnostics["boundaries"][0]["reason"] == "short_incomplete_suffix"


def test_short_incomplete_piece_can_still_merge_forward_into_continuation():
    complete = _take("complete", 0.0, 8.0, "This first thought is complete and useful.")
    fragment = _take("fragment", 8.02, 9.2, "because the", complete=False)
    continuation = _take("continuation", 9.22, 12.0, "reason matters for the final result.", complete=True)

    attempts, diagnostics = reconstruct_delivery_attempts((complete, fragment, continuation), None)

    assert len(attempts) == 2
    assert attempts[0].clip_id == "complete"
    assert attempts[1].start == fragment.start
    assert attempts[1].end == continuation.end
    assert "because the" in attempts[1].text
    assert "reason matters" in attempts[1].text
    assert any(item["reason"] == "short_incomplete_suffix" for item in diagnostics["boundaries"])
