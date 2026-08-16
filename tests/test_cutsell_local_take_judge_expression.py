from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.take_judge import rank_takes, score_take


def take(
    clip_id,
    *,
    expression=0.5,
    gesture=0.5,
    energy=0.5,
    fumble=0.0,
    distraction=0.0,
    product=0.7,
    motion=0.8,
    text="This product is amazing",
    start=0.0,
    end=2.0,
):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        signals=MediaSignals(
            source_asset_id="src",
            start=start,
            end=end,
            silence_ratio=0.0,
            audio_quality=0.8,
            face_visibility=0.9,
            eye_contact=0.8,
            framing_quality=0.8,
            product_visibility=product,
            motion_stability=motion,
            continuity=0.8,
            visual_fumble=fumble,
            expression_naturalness=expression,
            gesture_naturalness=gesture,
            delivery_energy=energy,
            distraction_risk=distraction,
        ),
    )


def test_natural_expression_and_delivery_win_when_other_signals_match():
    natural = take("natural", expression=0.9, gesture=0.85, energy=0.85)
    flat = take("flat", expression=0.35, gesture=0.35, energy=0.35)

    ranked = rank_takes((flat, natural))

    assert ranked[0].clip_id == "natural"
    assert ranked[0].score > ranked[1].score


def test_visual_fumble_and_distraction_penalize_apparently_complete_take():
    clean = take("clean", expression=0.8, gesture=0.8, energy=0.8)
    broken = take(
        "broken",
        expression=0.8,
        gesture=0.8,
        energy=0.8,
        fumble=0.9,
        distraction=0.9,
    )

    assert score_take(clean).score > score_take(broken).score


def test_product_drop_face_reaction_loses_retry_group_by_clear_margin():
    stable = take(
        "stable",
        expression=0.72,
        gesture=0.70,
        energy=0.70,
        fumble=0.15,
        distraction=0.12,
        product=0.72,
        motion=0.74,
    )
    dropped = take(
        "dropped",
        expression=0.28,
        gesture=0.26,
        energy=0.45,
        fumble=0.86,
        distraction=0.78,
        product=0.22,
        motion=0.30,
    )

    ranked = rank_takes((dropped, stable))

    assert ranked[0].clip_id == "stable"
    assert ranked[0].score - ranked[1].score >= 0.20


def test_low_product_visibility_alone_does_not_trigger_failure_penalty():
    talking_head = take(
        "talking-head",
        expression=0.82,
        gesture=0.76,
        energy=0.72,
        fumble=0.05,
        distraction=0.05,
        product=0.20,
        motion=0.78,
    )
    visible = take(
        "visible",
        expression=0.82,
        gesture=0.76,
        energy=0.72,
        fumble=0.05,
        distraction=0.05,
        product=0.80,
        motion=0.78,
    )

    # Product visibility contributes only its normal small ranking weight; the
    # interaction penalty is reserved for an actual multimodal failure.
    assert score_take(visible).score - score_take(talking_head).score < 0.05


def test_text_timing_fallback_still_works_without_visual_signals():
    candidate = CandidateTake(
        clip_id="text-only",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=2.0,
        text="This product is amazing",
        signals=None,
    )

    ranked = score_take(candidate)

    assert ranked.reason == "text_timing_baseline"
    assert ranked.score == 1.0


def test_single_character_asr_drift_short_prefix_loses_to_full_retry():
    short = take(
        "short",
        text="the popular croc",
        start=0.0,
        end=1.4,
        expression=0.82,
        gesture=0.82,
        energy=0.82,
    )
    full = take(
        "full",
        text="the popular crop black denim jeans are back in stock anything with pockets is a win for me",
        start=2.0,
        end=8.0,
        expression=0.72,
        gesture=0.72,
        energy=0.72,
    )

    ranked = rank_takes((short, full))
    by_id = {item.clip_id: item for item in ranked}

    assert ranked[0].clip_id == "full"
    assert "material_prefix_fragment_penalty" in by_id["short"].reason


def test_multiple_asr_differences_do_not_trigger_fuzzy_prefix_penalty():
    concise = take(
        "concise",
        text="the popular clock",
        start=0.0,
        end=1.4,
        expression=0.85,
        gesture=0.85,
        energy=0.85,
    )
    longer = take(
        "longer",
        text="the popular crop black denim jeans are back in stock anything with pockets is a win for me",
        start=2.0,
        end=8.0,
        expression=0.60,
        gesture=0.60,
        energy=0.60,
    )

    ranked = rank_takes((concise, longer))
    by_id = {item.clip_id: item for item in ranked}

    assert "material_prefix_fragment_penalty" not in by_id["concise"].reason
