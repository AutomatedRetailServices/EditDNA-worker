"""D-277: V1 Manual Timeline Composition Contract -- pure-type/validator
tests. Stage 46's own 22-case synthetic model matrix, plus a few
additional cases for the identity/order-independence and audio-mode
firewall properties this module's own docstring promises. No I/O, no
ffmpeg, no microphone, no AI engine, no mobile UI -- every test here
exercises `cutsell_worker.timeline_composition` alone.
"""
import pytest

from cutsell_worker import timeline_composition as tc


def _asset(asset_id="a1", duration_sec=30.0, identity=None):
    return tc.TimelineAssetReference(
        asset_id=asset_id,
        source_media_identity=identity or f"s3://bucket/{asset_id}.mp4",
        duration_sec=duration_sec,
    )


def _broll(placement_id="b1", start=5.0, end=10.0, src_in=0.0, src_out=5.0,
           audio_mode=tc.TimelineAudioMode.KEEP_PRIMARY_VOICE, asset=None):
    return tc.BrollPlacement(
        placement_id=placement_id, asset=asset or _asset(),
        timeline_start_sec=start, timeline_end_sec=end,
        source_in_sec=src_in, source_out_sec=src_out, audio_mode=audio_mode,
    )


def _vo(placement_id="v1", start=5.0, end=10.0, src_in=0.0, src_out=5.0, asset=None):
    return tc.VoiceOverPlacement(
        placement_id=placement_id, asset=asset or _asset(asset_id="vo1"),
        timeline_start_sec=start, timeline_end_sec=end,
        source_in_sec=src_in, source_out_sec=src_out,
    )


def _base_composition(duration_sec=30.0):
    return tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION,
        base_edit_identity="ai_edit_abc123",
        timeline_duration_sec=duration_sec,
    )


# =============================================================================
# 1. base edit only
# =============================================================================

def test_01_base_edit_only_is_valid_with_no_placements():
    comp = _base_composition()
    result = tc.validate_composition(comp)
    assert result.valid is True
    assert result.errors == ()
    assert comp.broll_placements == ()
    assert comp.voice_over_placements == ()


# =============================================================================
# 2. one B-roll keep primary voice
# =============================================================================

def test_02_one_broll_keep_primary_voice():
    comp = _base_composition()
    new_comp, result = tc.add_broll(comp, _broll(audio_mode=tc.TimelineAudioMode.KEEP_PRIMARY_VOICE))
    assert result.valid is True
    assert len(new_comp.broll_placements) == 1
    assert new_comp.broll_placements[0].audio_mode == tc.TimelineAudioMode.KEEP_PRIMARY_VOICE
    # Stage 6/8: original A-roll voice is never cut by a mere KEEP_PRIMARY_VOICE placement.
    assert tc.caption_source_for_region(new_comp, 7.0) == "ORIGINAL_PRIMARY_VOICE"


# =============================================================================
# 3. B-roll muted
# =============================================================================

def test_03_broll_muted_audio():
    comp = _base_composition()
    new_comp, result = tc.add_broll(comp, _broll(audio_mode=tc.TimelineAudioMode.MUTE_BROLL_AUDIO))
    assert result.valid is True
    assert new_comp.broll_placements[0].audio_mode == tc.TimelineAudioMode.MUTE_BROLL_AUDIO
    # Stage 35: MUTE_BROLL_AUDIO ignores B-roll source audio -- captions still
    # come from the (silent-under-video, still audible) primary voice.
    assert tc.caption_source_for_region(new_comp, 7.0) == "ORIGINAL_PRIMARY_VOICE"


# =============================================================================
# 4. B-roll source audio
# =============================================================================

def test_04_broll_source_audio():
    comp = _base_composition()
    new_comp, result = tc.add_broll(comp, _broll(audio_mode=tc.TimelineAudioMode.USE_BROLL_AUDIO))
    assert result.valid is True
    # Stage 12/32/13: USE_BROLL_AUDIO suppresses the primary voice for that
    # region and becomes the caption source.
    assert tc.caption_source_for_region(new_comp, 7.0) == "BROLL_SOURCE_AUDIO"
    assert tc.caption_source_for_region(new_comp, 2.0) == "ORIGINAL_PRIMARY_VOICE"


# =============================================================================
# 5. moved B-roll
# =============================================================================

def test_05_moved_broll_changes_interval_not_asset():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(start=5.0, end=10.0, src_in=0.0, src_out=5.0))
    moved, result = tc.move_broll(comp, "b1", timeline_start_sec=15.0, timeline_end_sec=20.0)
    assert result.valid is True
    placement = moved.broll_placements[0]
    assert placement.timeline_start_sec == 15.0 and placement.timeline_end_sec == 20.0
    # Stage 19: source bytes/asset/source_in/source_out unchanged by a move.
    assert placement.source_in_sec == 0.0 and placement.source_out_sec == 5.0
    assert placement.asset.asset_id == "a1"


# =============================================================================
# 6. trimmed B-roll
# =============================================================================

def test_06_trimmed_broll_changes_source_window():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(start=5.0, end=10.0, src_in=0.0, src_out=5.0))
    trimmed, result = tc.trim_broll(
        comp, "b1", source_in_sec=1.0, source_out_sec=3.0,
        timeline_start_sec=5.0, timeline_end_sec=7.0,
    )
    assert result.valid is True
    placement = trimmed.broll_placements[0]
    assert placement.source_in_sec == 1.0 and placement.source_out_sec == 3.0
    assert placement.timeline_end_sec - placement.timeline_start_sec == 2.0


# =============================================================================
# 7. replaced B-roll
# =============================================================================

def test_07_replaced_broll_substitutes_asset_identity():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(start=5.0, end=10.0))
    new_placement = _broll(placement_id="b1", start=5.0, end=10.0,
                            asset=_asset(asset_id="a2", identity="s3://bucket/a2.mp4"))
    replaced, result = tc.replace_broll(comp, "b1", new_placement)
    assert result.valid is True
    assert replaced.broll_placements[0].asset.asset_id == "a2"
    assert len(replaced.broll_placements) == 1


def test_07b_replace_broll_rejects_mismatched_placement_id():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(start=5.0, end=10.0))
    mismatched = _broll(placement_id="wrong-id", start=5.0, end=10.0)
    result_comp, result = tc.replace_broll(comp, "b1", mismatched)
    assert result_comp is None
    assert result.valid is False


# =============================================================================
# 8. deleted B-roll
# =============================================================================

def test_08_deleted_broll_removed_from_composition():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(start=5.0, end=10.0))
    deleted, result = tc.delete_broll(comp, "b1")
    assert result.valid is True
    assert deleted.broll_placements == ()


# =============================================================================
# 9. multiple non-overlapping B-rolls
# =============================================================================

def test_09_multiple_non_overlapping_brolls_allowed():
    comp = _base_composition()
    comp, r1 = tc.add_broll(comp, _broll(placement_id="b1", start=0.0, end=5.0))
    comp, r2 = tc.add_broll(comp, _broll(placement_id="b2", start=5.0, end=10.0))
    comp, r3 = tc.add_broll(comp, _broll(placement_id="b3", start=12.0, end=15.0))
    assert r1.valid and r2.valid and r3.valid
    assert len(comp.broll_placements) == 3


# =============================================================================
# 10. VO insertion
# =============================================================================

def test_10_voice_over_insertion():
    comp = _base_composition()
    new_comp, result = tc.add_voice_over(comp, _vo(start=5.0, end=10.0))
    assert result.valid is True
    assert len(new_comp.voice_over_placements) == 1
    assert tc.caption_source_for_region(new_comp, 7.0) == "VOICE_OVER"


# =============================================================================
# 11. VO move
# =============================================================================

def test_11_voice_over_move():
    comp = _base_composition()
    comp, _ = tc.add_voice_over(comp, _vo(start=5.0, end=10.0))
    moved, result = tc.move_voice_over(comp, "v1", timeline_start_sec=15.0, timeline_end_sec=18.0)
    assert result.valid is True
    p = moved.voice_over_placements[0]
    assert p.timeline_start_sec == 15.0 and p.timeline_end_sec == 18.0


# =============================================================================
# 12. VO trim
# =============================================================================

def test_12_voice_over_trim():
    comp = _base_composition()
    comp, _ = tc.add_voice_over(comp, _vo(start=5.0, end=10.0, src_in=0.0, src_out=5.0))
    trimmed, result = tc.trim_voice_over(
        comp, "v1", source_in_sec=1.0, source_out_sec=4.0,
        timeline_start_sec=5.0, timeline_end_sec=8.0,
    )
    assert result.valid is True
    p = trimmed.voice_over_placements[0]
    assert p.source_in_sec == 1.0 and p.source_out_sec == 4.0


# =============================================================================
# 13. VO replacement
# =============================================================================

def test_13_voice_over_replacement():
    comp = _base_composition()
    comp, _ = tc.add_voice_over(comp, _vo(start=5.0, end=10.0))
    new_vo = _vo(placement_id="v1", start=5.0, end=10.0, asset=_asset(asset_id="vo2"))
    replaced, result = tc.replace_voice_over(comp, "v1", new_vo)
    assert result.valid is True
    assert replaced.voice_over_placements[0].asset.asset_id == "vo2"


# =============================================================================
# 14. VO deletion
# =============================================================================

def test_14_voice_over_deletion_restores_primary_voice_captions():
    comp = _base_composition()
    comp, _ = tc.add_voice_over(comp, _vo(start=5.0, end=10.0))
    assert tc.caption_source_for_region(comp, 7.0) == "VOICE_OVER"
    deleted, result = tc.delete_voice_over(comp, "v1")
    assert result.valid is True
    assert deleted.voice_over_placements == ()
    assert tc.caption_source_for_region(deleted, 7.0) == "ORIGINAL_PRIMARY_VOICE"


# =============================================================================
# 15. invalid negative interval
# =============================================================================

def test_15_invalid_negative_timeline_interval_rejected():
    comp = _base_composition()
    bad = _broll(start=-2.0, end=5.0)
    result_comp, result = tc.add_broll(comp, bad)
    assert result_comp is None
    assert result.valid is False
    assert any("timeline_start_sec" in e for e in result.errors)


def test_15b_invalid_negative_source_in_rejected():
    comp = _base_composition()
    bad = _broll(src_in=-1.0, src_out=3.0)
    result_comp, result = tc.add_broll(comp, bad)
    assert result_comp is None
    assert result.valid is False
    assert any("source_in_sec" in e for e in result.errors)


# =============================================================================
# 16. out-of-bounds asset
# =============================================================================

def test_16_out_of_bounds_source_window_rejected():
    comp = _base_composition()
    short_asset = _asset(duration_sec=3.0)
    bad = _broll(src_in=0.0, src_out=5.0, asset=short_asset)  # exceeds 3.0s asset
    result_comp, result = tc.add_broll(comp, bad)
    assert result_comp is None
    assert result.valid is False
    assert any("exceeds asset duration" in e for e in result.errors)


def test_16b_out_of_bounds_timeline_duration_rejected():
    comp = _base_composition(duration_sec=8.0)
    bad = _broll(start=5.0, end=10.0)  # exceeds 8.0s timeline
    result_comp, result = tc.add_broll(comp, bad)
    assert result_comp is None
    assert result.valid is False
    assert any("exceeds timeline duration" in e for e in result.errors)


# =============================================================================
# 17. overlap conflict
# =============================================================================

def test_17_overlapping_broll_placements_rejected():
    comp = _base_composition()
    comp, r1 = tc.add_broll(comp, _broll(placement_id="b1", start=0.0, end=5.0))
    assert r1.valid
    result_comp, result = tc.add_broll(comp, _broll(placement_id="b2", start=3.0, end=8.0))
    assert result_comp is None
    assert result.valid is False
    assert any("overlap" in e for e in result.errors)


def test_17b_overlapping_voice_over_placements_rejected():
    comp = _base_composition()
    comp, r1 = tc.add_voice_over(comp, _vo(placement_id="v1", start=0.0, end=5.0))
    assert r1.valid
    result_comp, result = tc.add_voice_over(comp, _vo(placement_id="v2", start=4.0, end=9.0))
    assert result_comp is None
    assert result.valid is False
    assert any("overlap" in e for e in result.errors)


def test_17c_broll_and_voice_over_may_coexist_over_same_interval():
    """B-roll (visual layer) and voice-over (audio layer) are
    independent -- they are never checked against each other for
    overlap."""
    comp = _base_composition()
    comp, r1 = tc.add_broll(comp, _broll(placement_id="b1", start=0.0, end=5.0))
    comp, r2 = tc.add_voice_over(comp, _vo(placement_id="v1", start=0.0, end=5.0))
    assert r1.valid and r2.valid
    assert len(comp.broll_placements) == 1
    assert len(comp.voice_over_placements) == 1


# =============================================================================
# 18. timeline identity deterministic
# =============================================================================

def test_18_timeline_identity_deterministic_for_identical_state():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(placement_id="b1", start=0.0, end=5.0))
    id_a = tc.compute_timeline_identity(comp)
    id_b = tc.compute_timeline_identity(comp)
    assert id_a == id_b
    assert id_a.startswith("timeline_")


# =============================================================================
# 19. identity changes on edit
# =============================================================================

def test_19_identity_changes_after_an_edit():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(placement_id="b1", start=0.0, end=5.0))
    id_before = tc.compute_timeline_identity(comp)
    moved, _ = tc.move_broll(comp, "b1", timeline_start_sec=10.0, timeline_end_sec=15.0)
    id_after = tc.compute_timeline_identity(moved)
    assert id_before != id_after


# =============================================================================
# 20. path-independent identity (order-independent here, and never
# derived from a filesystem path in the first place)
# =============================================================================

def test_20_identity_independent_of_placement_insertion_order():
    comp_a = _base_composition()
    comp_a, _ = tc.add_broll(comp_a, _broll(placement_id="b1", start=0.0, end=5.0))
    comp_a, _ = tc.add_broll(comp_a, _broll(placement_id="b2", start=6.0, end=9.0))

    comp_b = _base_composition()
    comp_b, _ = tc.add_broll(comp_b, _broll(placement_id="b2", start=6.0, end=9.0))
    comp_b, _ = tc.add_broll(comp_b, _broll(placement_id="b1", start=0.0, end=5.0))

    assert tc.compute_timeline_identity(comp_a) == tc.compute_timeline_identity(comp_b)


def test_20b_asset_reference_never_carries_a_filesystem_path_field():
    # Structural guarantee: TimelineAssetReference has exactly the three
    # documented fields, none named/shaped like a local path.
    fields = tc.TimelineAssetReference.__dataclass_fields__.keys()
    assert set(fields) == {"asset_id", "source_media_identity", "duration_sec"}


# =============================================================================
# 21. underlying A-roll restored after B-roll deletion
# =============================================================================

def test_21_underlying_a_roll_restored_after_broll_deletion():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll(audio_mode=tc.TimelineAudioMode.USE_BROLL_AUDIO))
    assert tc.caption_source_for_region(comp, 7.0) == "BROLL_SOURCE_AUDIO"
    deleted, result = tc.delete_broll(comp, "b1")
    assert result.valid is True
    # Stage 22: primary A-roll remains intact underneath -- once the B-roll
    # placement is gone, the region reverts to the original primary voice.
    assert tc.caption_source_for_region(deleted, 7.0) == "ORIGINAL_PRIMARY_VOICE"


# =============================================================================
# 22. primary A-roll mode agnostic
# =============================================================================

def test_22_timeline_agnostic_to_primary_a_roll_visual_mode():
    """Stage 42: the timeline contract must not know or care whether
    base_edit_identity refers to a talking-head, faceless-product,
    hands/product, or demo-action primary edit -- there is no field
    anywhere in TimelineComposition that encodes visual mode."""
    for base_identity in ("ai_edit_talking_head", "ai_edit_faceless_product",
                           "ai_edit_hands_product", "ai_edit_demo_action"):
        comp = tc.TimelineComposition(
            contract_version=tc.TIMELINE_CONTRACT_VERSION,
            base_edit_identity=base_identity,
            timeline_duration_sec=30.0,
        )
        new_comp, result = tc.add_broll(comp, _broll())
        assert result.valid is True
        assert new_comp.base_edit_identity == base_identity
    fields = tc.TimelineComposition.__dataclass_fields__.keys()
    assert "visual_mode" not in fields and "scene_type" not in fields


# =============================================================================
# Additional coverage: unknown placement_id operations fail closed,
# revision identity distinctness, audio-mode vocabulary is exactly 3
# values (Stage 7's own "smallest V1 set").
# =============================================================================

def test_move_unknown_broll_placement_fails_closed():
    comp = _base_composition()
    result_comp, result = tc.move_broll(comp, "does-not-exist", 0.0, 5.0)
    assert result_comp is None
    assert result.valid is False


def test_delete_unknown_voice_over_placement_fails_closed():
    comp = _base_composition()
    result_comp, result = tc.delete_voice_over(comp, "does-not-exist")
    assert result_comp is None
    assert result.valid is False


def test_revision_identity_distinct_from_base_edit_identity():
    comp = _base_composition()
    comp, _ = tc.add_broll(comp, _broll())
    revision = tc.derive_revision_identity(comp)
    assert revision.base_edit_identity == "ai_edit_abc123"
    assert revision.identity != revision.base_edit_identity
    assert revision.contract_version == tc.TIMELINE_CONTRACT_VERSION


def test_audio_mode_vocabulary_is_exactly_three_values():
    assert {m.value for m in tc.TimelineAudioMode} == {
        "KEEP_PRIMARY_VOICE", "USE_BROLL_AUDIO", "MUTE_BROLL_AUDIO",
    }


def test_voice_over_placement_carries_no_audio_mode_field():
    fields = tc.VoiceOverPlacement.__dataclass_fields__.keys()
    assert "audio_mode" not in fields


def test_timeline_contract_version_is_one():
    assert tc.TIMELINE_CONTRACT_VERSION == 1


def test_composition_and_placements_are_frozen_dataclasses():
    comp = _base_composition()
    with pytest.raises(Exception):
        comp.timeline_duration_sec = 999.0  # type: ignore[misc]
    placement = _broll()
    with pytest.raises(Exception):
        placement.timeline_start_sec = 999.0  # type: ignore[misc]


def test_duplicate_placement_id_rejected_by_validate_composition():
    comp = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION,
        base_edit_identity="ai_edit_abc123",
        timeline_duration_sec=30.0,
        broll_placements=(
            _broll(placement_id="dup", start=0.0, end=2.0),
            _broll(placement_id="dup", start=10.0, end=12.0),
        ),
    )
    result = tc.validate_composition(comp)
    assert result.valid is False
    assert any("duplicate broll placement_id" in e for e in result.errors)
