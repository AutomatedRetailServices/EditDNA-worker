from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.take_judge import rank_takes, score_take


def take(clip_id, *, expression=0.5, gesture=0.5, energy=0.5, fumble=0.0, distraction=0.0):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=2.0,
        text="This product is amazing",
        signals=MediaSignals(
            source_asset_id="src",
            start=0.0,
            end=2.0,
            silence_ratio=0.0,
            audio_quality=0.8,
            face_visibility=0.9,
            eye_contact=0.8,
            framing_quality=0.8,
            product_visibility=0.7,
            motion_stability=0.8,
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
