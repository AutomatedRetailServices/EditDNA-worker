from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.local_retry_grouping import _adjacent_reformulated_retries
from cutsell_worker.session_boundaries import safe_group_takes_by_sessions
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


def test_serial_retry_envelope_allows_one_plausible_trailing_partial_before_full_take():
    takes = (
        take("short1", 0.0, 2.2, "the popular croc"),
        take("short2", 3.0, 4.9, "the popular croc"),
        take("short3", 5.6, 7.1, "the popular croc"),
        take("repeat", 7.8, 12.3, "crop popular crop popular crop popular"),
        take("partial", 14.8, 19.3, "the popular crop black jeans okay now we hold"),
        take("full", 21.7, 27.7, "the popular crop black denim jeans are back in stock anything with pockets is a win for me"),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("short1", "short2", "short3", "repeat", "partial", "full"),)
    assert "serial_retry_envelope_collapsed" in result.reason


def test_local_path_keeps_distinct_nearby_content_as_separate_groups():
    takes = (
        take("a", 0.0, 4.0, "the popular crop black jeans are finally back in stock"),
        take("b", 5.0, 9.0, "this jacket has a removable hood and two inside pockets"),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("a",), ("b",))


def test_spanish_reformulated_retry_with_same_opening_collapses():
    takes = (
        take(
            "first", 82.8, 90.6,
            "Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que ella pudiera imaginarse o indicar",
        ),
        take(
            "retry", 95.5, 104.3,
            "Al terminar mi contrato cambié de ginecóloga y le pedí todos los test que ella pudiera imaginar e indicar",
        ),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("first", "retry"),)


def test_reformulated_retry_fallback_merges_when_seed_groups_missed_it():
    takes = (
        take(
            "first", 82.8, 90.6,
            "Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que ella pudiera imaginarse o indicar",
        ),
        take(
            "retry", 95.5, 104.3,
            "Al terminar mi contrato cambié de ginecóloga y le pedí todos los test que ella pudiera imaginar e indicar",
        ),
    )
    groups, changed = _adjacent_reformulated_retries((("first",), ("retry",)), takes)
    assert changed is True
    assert groups == (("first", "retry"),)


def test_reformulated_retry_rule_does_not_merge_distinct_spanish_thoughts():
    takes = (
        take("first", 10.0, 15.0, "Al terminar mi contrato hablé con mi ginecóloga sobre mis análisis de rutina"),
        take("distinct", 18.0, 24.0, "Después regresé al barco y organicé mis maletas para el viaje siguiente"),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("first",), ("distinct",))
    groups, changed = _adjacent_reformulated_retries((("first",), ("distinct",)), takes)
    assert changed is False
    assert groups == (("first",), ("distinct",))


def test_same_opening_without_enough_shared_content_remains_separate():
    takes = (
        take("first", 10.0, 15.0, "After my contract ended I asked my doctor for every possible blood test"),
        take("distinct", 18.0, 24.0, "After my contract ended I flew home and spent a week visiting my family"),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("first",), ("distinct",))


def test_english_reformulated_retry_with_shared_opening_collapses():
    takes = (
        take("first", 10.0, 16.0, "When I tried this moisturizer every morning my skin stayed hydrated all day"),
        take("retry", 20.0, 26.0, "When I tried this moisturizer each morning my skin stayed hydrated through the whole day"),
    )
    result = safe_group_takes(None, takes)
    assert result.groups == (("first", "retry"),)


def test_best_take_penalizes_material_prefix_even_if_complete_idea_is_wrongly_true():
    fragment = take("fragment", 0.0, 2.0, "the popular crop black jeans")
    full = take("full", 2.5, 7.0, "the popular crop black jeans are finally back in stock today")
    ranked = rank_takes((fragment, full))
    assert ranked[0].clip_id == "full"
    by_id = {item.clip_id: item for item in ranked}
    assert "material_prefix_fragment_penalty" in by_id["fragment"].reason
    assert by_id["full"].score > by_id["fragment"].score


def test_best_take_penalizes_repetitive_restart_when_fuller_retry_exists():
    repeat = take("repeat", 0.0, 4.5, "crop popular crop popular crop popular")
    full = take("full", 5.0, 11.0, "the popular crop black denim jeans are back in stock anything with pockets is a win for me")
    ranked = rank_takes((repeat, full))
    assert ranked[0].clip_id == "full"
    by_id = {item.clip_id: item for item in ranked}
    assert "repetitive_restart_fragment_penalty" in by_id["repeat"].reason
    assert by_id["full"].score > by_id["repeat"].score


def test_repeated_phrase_without_fuller_related_retry_is_not_penalized():
    slogan = take("slogan", 0.0, 4.0, "go go go go get yours")
    distinct = take("distinct", 5.0, 9.0, "this jacket has a removable hood and pockets")
    ranked = rank_takes((slogan, distinct))
    by_id = {item.clip_id: item for item in ranked}
    assert "repetitive_restart_fragment_penalty" not in by_id["slogan"].reason


def test_best_take_penalizes_restart_tail_when_full_retry_follows():
    partial = take("partial", 0.0, 4.54, "the popular crop black jeans okay now we hold")
    full = take("full", 6.0, 12.0, "the popular crop black denim jeans are back in stock anything with pockets is a win for me")
    ranked = rank_takes((partial, full))
    by_id = {item.clip_id: item for item in ranked}
    assert ranked[0].clip_id == "full"
    assert "restart_tail_fragment_penalty" in by_id["partial"].reason
    assert by_id["full"].score > by_id["partial"].score


def test_okay_now_at_start_of_valid_take_is_not_restart_tail_penalized():
    valid = take("valid", 0.0, 4.0, "okay now let us look at the zipper and inside pockets")
    other = take("other", 5.0, 10.0, "the lining is soft and the sleeve length is perfect for me")
    ranked = rank_takes((valid, other))
    by_id = {item.clip_id: item for item in ranked}
    assert "restart_tail_fragment_penalty" not in by_id["valid"].reason


def test_best_take_prefers_complete_natural_delivery_over_product_drop_retry():
    bad = MediaSignals(
        "src", 0.0, 4.0, audio_quality=0.85, face_visibility=0.95, eye_contact=0.35,
        framing_quality=0.85, product_visibility=0.20, motion_stability=0.25,
        continuity=0.35, visual_fumble=0.90, expression_naturalness=0.20,
        gesture_naturalness=0.15, delivery_energy=0.30, distraction_risk=0.90,
    )
    good = MediaSignals(
        "src", 5.0, 9.0, audio_quality=0.85, face_visibility=0.95, eye_contact=0.90,
        framing_quality=0.90, product_visibility=0.80, motion_stability=0.90,
        continuity=0.90, visual_fumble=0.05, expression_naturalness=0.90,
        gesture_naturalness=0.85, delivery_energy=0.85, distraction_risk=0.05,
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


def test_session_scoped_production_path_uses_installed_retry_integrity():
    takes = (
        take("short1", 0.0, 2.2, "the popular croc"),
        take("short2", 3.0, 4.9, "the popular croc"),
        take("short3", 5.6, 7.1, "the popular croc"),
        take("repeat", 7.8, 12.3, "crop popular crop popular crop popular"),
        take("partial", 14.8, 19.3, "the popular crop black jeans okay now we hold"),
        take("full", 21.7, 27.7, "the popular crop black denim jeans are back in stock anything with pockets is a win for me"),
    )
    result = safe_group_takes_by_sessions(None, takes, None)
    assert result.groups == (("short1", "short2", "short3", "repeat", "partial", "full"),)
    assert "serial_retry_envelope_collapsed" in result.reason
