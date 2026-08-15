from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.take_grouping_provider import safe_group_takes
from cutsell_worker.take_judge import rank_takes


def take(clip_id, start, end, text, *, complete=True, signals=None):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
        signals=signals,
    )


def test_local_path_absorbs_interstitial_false_start_into_retry_group():
    takes = (
        take("a", 0.0, 4.0, "the popular crop black jeans are finally back in stock"),
        take("debris", 4.4, 5.4, "crop crop", complete=False),
        take("b", 6.0, 10.2, "the popular crop black jeans are finally back in stock"),
    )

    result = safe_group_takes(None, takes)

    assert result.groups == (("a", "debris", "b"),)
    assert "interstitial_retry_debris_absorbed" in result.reason


def test_serial_weak_retries_collapse_into_following_full_take():
    takes = (
        take("short", 0.0, 2.4, "the popular croc"),
        take("repeat", 3.0, 5.8, "crop popular crop popular crop popular"),
        take("meta", 6.5, 10.5, "the popular crop black jeans okay now whole sentence okay"),
        take("full", 11.5, 17.0, "the popular crop black denim jeans are back in stock anything with pockets is a win for me"),
    )

    result = safe_group_takes(None, takes)

    assert result.groups == (("short", "repeat", "meta", "full"),)
    assert "serial_retry_envelope_collapsed" in result.reason


def test_local_path_keeps_distinct_nearby_content_as_separate_groups():
    takes = (
        take("a", 0.0, 4.0, "the popular crop black jeans are finally back in stock"),
        take("b", 5.0, 9.0, "this jacket has a removable hood and two inside pockets"),
    )

    result = safe_group_takes(None, takes)

    assert result.groups == (("a",), ("b",))


def test_best_take_penalizes_material_prefix_even_if_complete_idea_is_wrongly_true():
    fragment = take("fragment", 0.0, 2.0, "the popular crop black jeans")
    full = take("full", 2.5, 7.0, "the popular crop black jeans are finally back in stock today")

    ranked = rank_takes((fragment, full))

    assert ranked[0].clip_id == "full"
    by_id = {item.clip_id: item for item in ranked}
    assert "material_prefix_fragment_penalty" in by_id["fragment"].reason
    assert by_id["full"].score > by_id["fragment"].score


def test_best_take_prefers_complete_natural_delivery_over_product_drop_retry():
    bad = MediaSignals(
        "src", 0.0, 4.0,
        audio_quality=0.85,
        face_visibility=0.95,
        eye_contact=0.35,
        framing_quality=0.85,
        product_visibility=0.20,
        motion_stability=0.25,
        continuity=0.35,
        visual_fumble=0.90,
        expression_naturalness=0.20,
        gesture_naturalness=0.15,
        delivery_energy=0.30,
        distraction_risk=0.90,
    )
    good = MediaSignals(
        "src", 5.0, 9.0,
        audio_quality=0.85,
        face_visibility=0.95,
        eye_contact=0.90,
        framing_quality=0.90,
        product_visibility=0.80,
        motion_stability=0.90,
        continuity=0.90,
        visual_fumble=0.05,
        expression_naturalness=0.90,
        gesture_naturalness=0.85,
        delivery_energy=0.85,
        distraction_risk=0.05,
    )
    takes = (
        take("drop", 0.0, 4.0, "this bonder seal keeps everything locked in place", signals=bad),
        take("clean", 5.0, 9.0, "this bonder seal keeps everything locked in place", signals=good),
    )

    grouping = safe_group_takes(None, takes)
    assert grouping.groups == (("drop", "clean"),)

    ranked = rank_takes(takes)
    assert ranked[0].clip_id == "clean"
    assert ranked[0].score > ranked[1].score
