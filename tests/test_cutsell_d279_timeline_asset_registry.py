"""D-279 -- V1 Manual Timeline Asset Ingest / Persistence Contract.

Pure-type/pure-function coverage. No I/O, no network, no ffmpeg.
"""
from __future__ import annotations

import pytest

from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_composition as tc
from cutsell_worker import timeline_composition_executor as tce


def _asset(
    asset_id="asset_1",
    user_id="user_a",
    project_id="project_a",
    role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
    media_kind=reg.TimelineMediaKind.VIDEO,
    status=reg.TimelineAssetQualificationStatus.READY,
    has_audio=True,
    duration_sec=4.0,
    **overrides,
):
    fields = dict(
        asset_id=asset_id,
        ownership=reg.AssetOwnershipScope(user_id=user_id, project_id=project_id),
        role=role,
        media_kind=media_kind,
        source_media_identity=f"{asset_id}-identity",
        storage_reference=f"s3://bucket/{asset_id}.mp4",
        duration_sec=duration_sec,
        has_audio=has_audio,
        qualification_status=status,
    )
    fields.update(overrides)
    return reg.TimelineMediaAsset(**fields)


def _scope(user_id="user_a", project_id="project_a"):
    return reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)


# ---------------------------------------------------------------------------
# AssetOwnershipScope
# ---------------------------------------------------------------------------

def test_ownership_scope_rejects_empty_fields():
    with pytest.raises(ValueError):
        reg.AssetOwnershipScope(user_id="", project_id="p1")
    with pytest.raises(ValueError):
        reg.AssetOwnershipScope(user_id="u1", project_id="")


def test_ownership_scope_is_frozen():
    scope = _scope()
    with pytest.raises(Exception):
        scope.user_id = "other"  # type: ignore[misc]


def test_authorize_asset_access_requires_full_scope_match():
    a = _scope("user_a", "project_a")
    assert reg.authorize_asset_access(requesting=a, record_ownership=_scope("user_a", "project_a"))
    assert not reg.authorize_asset_access(requesting=a, record_ownership=_scope("user_a", "project_b"))
    assert not reg.authorize_asset_access(requesting=a, record_ownership=_scope("user_b", "project_a"))


def test_assert_asset_access_none_requesting_is_local_dev_bypass():
    reg.assert_asset_access(requesting=None, record_ownership=_scope())  # must not raise


def test_assert_asset_access_raises_permission_error_on_mismatch():
    with pytest.raises(PermissionError):
        reg.assert_asset_access(requesting=_scope("user_b", "project_a"), record_ownership=_scope("user_a", "project_a"))


def test_validate_content_sha256_source_rejects_etag_derivation():
    with pytest.raises(ValueError):
        reg.validate_content_sha256_source("deadbeef", derived_from_etag=True)


def test_validate_content_sha256_source_accepts_non_etag_value():
    assert reg.validate_content_sha256_source("deadbeef", derived_from_etag=False) == "deadbeef"


# ---------------------------------------------------------------------------
# TimelineMediaAsset construction + projections
# ---------------------------------------------------------------------------

def test_asset_rejects_empty_identity_and_nonpositive_duration():
    with pytest.raises(ValueError):
        _asset(asset_id="")
    with pytest.raises(ValueError):
        _asset(duration_sec=0.0)


def test_to_asset_reference_never_carries_storage_reference():
    asset = _asset()
    ref = reg.to_asset_reference(asset)
    assert isinstance(ref, tc.TimelineAssetReference)
    assert ref.asset_id == asset.asset_id
    assert ref.duration_sec == asset.duration_sec
    assert not hasattr(ref, "storage_reference")


def test_client_safe_asset_view_excludes_storage_reference():
    asset = _asset(technical_metadata_reference="probe_ref_1")
    view = reg.client_safe_asset_view(asset)
    assert "storage_reference" not in view
    assert "technical_metadata_reference" not in view
    assert view["asset_id"] == asset.asset_id
    assert view["qualification_status"] == "READY"


def test_list_project_assets_excludes_other_users_and_projects_and_deleted():
    mine = _asset(asset_id="mine")
    other_user = _asset(asset_id="other_user", user_id="user_b")
    other_project = _asset(asset_id="other_project", project_id="project_b")
    deleted = _asset(asset_id="deleted", status=reg.TimelineAssetQualificationStatus.DELETED)
    pool = (mine, other_user, other_project, deleted)
    visible = reg.list_project_assets(requesting=_scope(), assets=pool)
    assert {a.asset_id for a in visible} == {"mine"}


def test_list_project_assets_can_include_deleted_when_asked():
    deleted = _asset(asset_id="deleted", status=reg.TimelineAssetQualificationStatus.DELETED)
    visible = reg.list_project_assets(requesting=_scope(), assets=(deleted,), include_deleted=True)
    assert {a.asset_id for a in visible} == {"deleted"}


def test_reconcile_asset_duration_server_authority_always_wins():
    assert reg.reconcile_asset_duration(client_reported_duration_sec=999.0, server_probed_duration_sec=4.2) == 4.2


# ---------------------------------------------------------------------------
# resolve_timeline_asset -- the secure resolver seam into D-278
# ---------------------------------------------------------------------------

def test_resolve_valid_broll_asset_succeeds():
    asset = _asset()
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_RESOLVED
    assert isinstance(result.resolved, tce.ResolvedTimelineAsset)
    assert result.resolved.asset_id == asset.asset_id
    assert result.resolved.local_path == "/tmp/x.mp4"
    assert result.resolved.has_audio is True


def test_resolve_missing_asset_is_asset_not_found():
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=None, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_FOUND
    assert result.resolved is None


def test_resolve_deleted_asset_indistinguishable_from_not_found():
    asset = _asset(status=reg.TimelineAssetQualificationStatus.DELETED)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_FOUND


def test_resolve_wrong_user_is_asset_not_owned():
    asset = _asset(user_id="user_a")
    result = reg.resolve_timeline_asset(
        requesting=_scope("user_b", "project_a"), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_resolve_wrong_project_is_asset_not_owned_no_cross_project_reuse():
    asset = _asset(project_id="project_a")
    result = reg.resolve_timeline_asset(
        requesting=_scope("user_a", "project_b"), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_resolve_not_ready_asset_is_asset_not_ready():
    asset = _asset(status=reg.TimelineAssetQualificationStatus.QUALIFYING)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_READY


def test_resolve_wrong_media_kind_is_media_unsupported():
    asset = _asset(media_kind=reg.TimelineMediaKind.AUDIO)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.m4a",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_MEDIA_UNSUPPORTED


def test_resolve_use_broll_audio_on_silent_asset_fails_has_no_audio():
    asset = _asset(has_audio=False)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
        required_audio_mode=tc.TimelineAudioMode.USE_BROLL_AUDIO,
    )
    assert result.outcome == reg.ASSET_HAS_NO_AUDIO


def test_resolve_keep_primary_voice_on_silent_asset_still_succeeds():
    asset = _asset(has_audio=False)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/x.mp4",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
        required_audio_mode=tc.TimelineAudioMode.KEEP_PRIMARY_VOICE,
    )
    assert result.outcome == reg.ASSET_RESOLVED


def test_resolve_voice_over_asset_succeeds_for_audio_kind():
    asset = _asset(asset_id="vo_1", role=reg.TimelineAssetRole.VOICE_OVER, media_kind=reg.TimelineMediaKind.AUDIO, duration_sec=2.0)
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="/tmp/vo.m4a",
        required_media_kind=reg.TimelineMediaKind.AUDIO,
    )
    assert result.outcome == reg.ASSET_RESOLVED


def test_resolve_blank_local_path_is_asset_not_found():
    asset = _asset()
    result = reg.resolve_timeline_asset(
        requesting=_scope(), asset=asset, local_path="   ",
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_FOUND


# ---------------------------------------------------------------------------
# validate_timeline_against_registry
# ---------------------------------------------------------------------------

def _composition_with_broll(broll_asset_ref, audio_mode=tc.TimelineAudioMode.KEEP_PRIMARY_VOICE):
    base = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION,
        base_edit_identity="base_1",
        timeline_duration_sec=10.0,
    )
    placement = tc.BrollPlacement(
        placement_id="broll_1",
        asset=broll_asset_ref,
        timeline_start_sec=2.0,
        timeline_end_sec=6.0,
        source_in_sec=0.0,
        source_out_sec=4.0,
        audio_mode=audio_mode,
    )
    new_comp, result = tc.add_broll(base, placement)
    assert result.valid, result.errors
    return new_comp


def test_validate_timeline_against_registry_accepts_valid_placement():
    asset = _asset()
    comp = _composition_with_broll(reg.to_asset_reference(asset))
    result = reg.validate_timeline_against_registry(
        requesting=_scope(), composition=comp, assets_by_id={asset.asset_id: asset},
    )
    assert result.valid


def test_validate_timeline_against_registry_flags_missing_asset():
    ref = tc.TimelineAssetReference(asset_id="ghost", source_media_identity="ghost-id", duration_sec=4.0)
    comp = _composition_with_broll(ref)
    result = reg.validate_timeline_against_registry(requesting=_scope(), composition=comp, assets_by_id={})
    assert not result.valid
    assert any(reg.ASSET_NOT_FOUND in e for e in result.errors)


def test_validate_timeline_against_registry_flags_wrong_owner():
    asset = _asset(user_id="user_b")
    comp = _composition_with_broll(reg.to_asset_reference(asset))
    result = reg.validate_timeline_against_registry(
        requesting=_scope("user_a", "project_a"), composition=comp, assets_by_id={asset.asset_id: asset},
    )
    assert not result.valid
    assert any(reg.ASSET_NOT_OWNED in e for e in result.errors)


def test_validate_timeline_against_registry_flags_not_ready():
    asset = _asset(status=reg.TimelineAssetQualificationStatus.QUALIFYING)
    comp = _composition_with_broll(reg.to_asset_reference(asset))
    result = reg.validate_timeline_against_registry(
        requesting=_scope(), composition=comp, assets_by_id={asset.asset_id: asset},
    )
    assert not result.valid
    assert any(reg.ASSET_NOT_READY in e for e in result.errors)


def test_validate_timeline_against_registry_flags_use_broll_audio_on_silent_asset():
    asset = _asset(has_audio=False)
    comp = _composition_with_broll(reg.to_asset_reference(asset), audio_mode=tc.TimelineAudioMode.USE_BROLL_AUDIO)
    result = reg.validate_timeline_against_registry(
        requesting=_scope(), composition=comp, assets_by_id={asset.asset_id: asset},
    )
    assert not result.valid
    assert any(reg.ASSET_HAS_NO_AUDIO in e for e in result.errors)


# ---------------------------------------------------------------------------
# Asset delete vs placement delete
# ---------------------------------------------------------------------------

def test_is_asset_referenced_true_for_broll_and_false_otherwise():
    asset = _asset()
    comp = _composition_with_broll(reg.to_asset_reference(asset))
    assert reg.is_asset_referenced(asset_id=asset.asset_id, composition=comp)
    assert not reg.is_asset_referenced(asset_id="unrelated", composition=comp)


def test_delete_unreferenced_asset_succeeds():
    asset = _asset()
    result = reg.delete_timeline_asset(requesting=_scope(), asset=asset, referencing_compositions=())
    assert result.outcome == reg.ASSET_DELETE_SUCCEEDED
    assert result.asset.qualification_status == reg.TimelineAssetQualificationStatus.DELETED


def test_delete_referenced_asset_is_blocked():
    asset = _asset()
    comp = _composition_with_broll(reg.to_asset_reference(asset))
    result = reg.delete_timeline_asset(requesting=_scope(), asset=asset, referencing_compositions=(comp,))
    assert result.outcome == reg.ASSET_REFERENCED


def test_delete_missing_asset_is_asset_not_found():
    result = reg.delete_timeline_asset(requesting=_scope(), asset=None)
    assert result.outcome == reg.ASSET_NOT_FOUND


def test_delete_wrong_owner_is_asset_not_owned():
    asset = _asset(user_id="user_b")
    result = reg.delete_timeline_asset(requesting=_scope("user_a", "project_a"), asset=asset)
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_placement_delete_never_touches_asset_library():
    """delete_broll on the D-277 composition removes only the reference;
    this module has no notion of that operation cascading into the asset
    registry -- confirmed structurally by checking `delete_broll` itself
    never imports this module."""
    import inspect
    source = inspect.getsource(tc)
    assert "timeline_asset_registry" not in source


def test_rerecord_voice_over_creates_new_identity_and_preserves_old():
    previous = _asset(asset_id="vo_old", role=reg.TimelineAssetRole.VOICE_OVER, media_kind=reg.TimelineMediaKind.AUDIO, duration_sec=2.0)
    new_asset = reg.rerecord_voice_over_asset(
        previous_asset=previous, new_asset_id="vo_new", new_storage_reference="s3://bucket/vo_new.m4a",
        new_source_media_identity="vo_new-identity", new_duration_sec=3.0,
    )
    assert new_asset.asset_id == "vo_new"
    assert new_asset.replaces_asset_id == "vo_old"
    assert new_asset.qualification_status == reg.TimelineAssetQualificationStatus.UPLOADED
    # the old asset object itself is untouched (never mutated -- frozen dataclass)
    assert previous.qualification_status == reg.TimelineAssetQualificationStatus.READY


def test_rerecord_voice_over_rejects_same_asset_id():
    previous = _asset(asset_id="vo_old", role=reg.TimelineAssetRole.VOICE_OVER, media_kind=reg.TimelineMediaKind.AUDIO, duration_sec=2.0)
    with pytest.raises(ValueError):
        reg.rerecord_voice_over_asset(
            previous_asset=previous, new_asset_id="vo_old", new_storage_reference="s3://bucket/vo_old.m4a",
            new_source_media_identity="vo_old-identity", new_duration_sec=3.0,
        )


def test_broll_replace_never_deletes_old_asset_from_library():
    old_asset = _asset(asset_id="broll_old")
    new_asset = _asset(asset_id="broll_new")
    comp = _composition_with_broll(reg.to_asset_reference(old_asset))
    original_placement = comp.broll_placements[0]
    new_placement = tc.BrollPlacement(
        placement_id=original_placement.placement_id,
        asset=reg.to_asset_reference(new_asset),
        timeline_start_sec=original_placement.timeline_start_sec,
        timeline_end_sec=original_placement.timeline_end_sec,
        source_in_sec=original_placement.source_in_sec,
        source_out_sec=original_placement.source_out_sec,
        audio_mode=original_placement.audio_mode,
    )
    replaced, result = tc.replace_broll(comp, placement_id="broll_1", new_placement=new_placement)
    assert result.valid, result.errors
    # both library entries remain independently valid/undeleted -- replace
    # only ever swaps the placement's own reference.
    assert old_asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    assert new_asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    assert not reg.is_asset_referenced(asset_id=old_asset.asset_id, composition=replaced)
    assert reg.is_asset_referenced(asset_id=new_asset.asset_id, composition=replaced)


# ---------------------------------------------------------------------------
# Timeline persistence + optimistic concurrency
# ---------------------------------------------------------------------------

def _empty_composition():
    return tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=10.0,
    )


def test_build_persistence_record_identity_matches_derive_revision_identity():
    comp = _empty_composition()
    record = reg.build_persistence_record(ownership=_scope(), composition=comp)
    assert record.timeline_revision_identity == tc.derive_revision_identity(comp).identity


def test_persistence_record_rejects_mismatched_revision_identity():
    comp = _empty_composition()
    with pytest.raises(ValueError):
        reg.TimelinePersistenceRecord(
            contract_version=1, ownership=_scope(), composition=comp, timeline_revision_identity="wrong",
        )


def test_save_timeline_revision_first_save_never_conflicts():
    comp = _empty_composition()
    result = reg.save_timeline_revision(
        ownership=_scope(), composition=comp, current_record=None, expected_revision_identity=None,
    )
    assert result.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    assert result.record is not None


def test_save_timeline_revision_rejects_invalid_composition():
    bad_placement = tc.BrollPlacement(
        placement_id="p1",
        asset=tc.TimelineAssetReference(asset_id="a", source_media_identity="a-id", duration_sec=4.0),
        timeline_start_sec=-1.0,
        timeline_end_sec=6.0,
        source_in_sec=0.0,
        source_out_sec=4.0,
    )
    invalid = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=10.0,
        broll_placements=(bad_placement,),
    )
    result = reg.save_timeline_revision(
        ownership=_scope(), composition=invalid, current_record=None, expected_revision_identity=None,
    )
    assert result.outcome == reg.TIMELINE_INVALID


def test_save_timeline_revision_stale_expected_revision_is_rejected():
    comp = _empty_composition()
    first = reg.build_persistence_record(ownership=_scope(), composition=comp)
    changed = comp.__class__(
        contract_version=comp.contract_version, base_edit_identity=comp.base_edit_identity,
        timeline_duration_sec=comp.timeline_duration_sec + 1.0,
    )
    result = reg.save_timeline_revision(
        ownership=_scope(), composition=changed, current_record=first, expected_revision_identity="stale_value",
    )
    assert result.outcome == reg.TIMELINE_REVISION_CONFLICT


def test_save_timeline_revision_correct_expected_revision_succeeds():
    comp = _empty_composition()
    first = reg.build_persistence_record(ownership=_scope(), composition=comp)
    changed = comp.__class__(
        contract_version=comp.contract_version, base_edit_identity=comp.base_edit_identity,
        timeline_duration_sec=comp.timeline_duration_sec + 1.0,
    )
    result = reg.save_timeline_revision(
        ownership=_scope(), composition=changed, current_record=first,
        expected_revision_identity=first.timeline_revision_identity,
    )
    assert result.outcome == reg.TIMELINE_SAVE_SUCCEEDED


def test_save_timeline_revision_wrong_owner_of_current_record_is_not_owned():
    comp = _empty_composition()
    first = reg.build_persistence_record(ownership=_scope("user_a", "project_a"), composition=comp)
    result = reg.save_timeline_revision(
        ownership=_scope("user_b", "project_a"), composition=comp, current_record=first,
        expected_revision_identity=first.timeline_revision_identity,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_reopen_roundtrip_preserves_revision_identity():
    comp = _empty_composition()
    saved = reg.save_timeline_revision(
        ownership=_scope(), composition=comp, current_record=None, expected_revision_identity=None,
    ).record
    # simulate a reopen: read the persisted record back, save again unchanged
    reopened = reg.save_timeline_revision(
        ownership=_scope(), composition=saved.composition, current_record=saved,
        expected_revision_identity=saved.timeline_revision_identity,
    )
    assert reopened.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    assert reopened.record.timeline_revision_identity == saved.timeline_revision_identity


# ---------------------------------------------------------------------------
# Export reproducibility -- exact revision only
# ---------------------------------------------------------------------------

def test_resolve_timeline_export_matches_exact_revision():
    comp = _empty_composition()
    record = reg.build_persistence_record(ownership=_scope(), composition=comp)
    result = reg.resolve_timeline_export(
        requesting=_scope(), record=record, expected_revision_identity=record.timeline_revision_identity,
    )
    assert result.outcome == reg.TIMELINE_EXPORT_RESOLVED
    assert result.composition == comp


def test_resolve_timeline_export_rejects_stale_revision():
    comp = _empty_composition()
    record = reg.build_persistence_record(ownership=_scope(), composition=comp)
    result = reg.resolve_timeline_export(
        requesting=_scope(), record=record, expected_revision_identity="stale",
    )
    assert result.outcome == reg.TIMELINE_REVISION_CONFLICT


def test_resolve_timeline_export_rejects_wrong_owner():
    comp = _empty_composition()
    record = reg.build_persistence_record(ownership=_scope("user_a", "project_a"), composition=comp)
    result = reg.resolve_timeline_export(
        requesting=_scope("user_b", "project_a"), record=record,
        expected_revision_identity=record.timeline_revision_identity,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_resolve_timeline_export_missing_record_is_timeline_invalid():
    result = reg.resolve_timeline_export(requesting=_scope(), record=None, expected_revision_identity="anything")
    assert result.outcome == reg.TIMELINE_INVALID


# ---------------------------------------------------------------------------
# Structural / source-inspection checks
# ---------------------------------------------------------------------------

def test_module_never_imports_ai_or_selection_authorities():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(reg))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {
                "best_take_authority", "deterministic_best_take_authority", "selection_freeze",
                "boundary_engine", "render", "visual_finishing", "audio_finishing_executor",
            }


def test_module_performs_no_filesystem_or_network_io():
    import inspect
    source = inspect.getsource(reg)
    for forbidden in ("open(", "subprocess", "boto3", "requests.", "socket.", "os.remove", "os.path.exists"):
        assert forbidden not in source


def test_error_vocabulary_is_the_fixed_closed_set():
    expected = {
        "ASSET_NOT_FOUND", "ASSET_NOT_OWNED", "ASSET_NOT_READY", "ASSET_MEDIA_UNSUPPORTED",
        "ASSET_HAS_NO_AUDIO", "TIMELINE_REVISION_CONFLICT", "TIMELINE_INVALID", "ASSET_REFERENCED",
    }
    actual = {
        reg.ASSET_NOT_FOUND, reg.ASSET_NOT_OWNED, reg.ASSET_NOT_READY, reg.ASSET_MEDIA_UNSUPPORTED,
        reg.ASSET_HAS_NO_AUDIO, reg.TIMELINE_REVISION_CONFLICT, reg.TIMELINE_INVALID, reg.ASSET_REFERENCED,
    }
    assert actual == expected
