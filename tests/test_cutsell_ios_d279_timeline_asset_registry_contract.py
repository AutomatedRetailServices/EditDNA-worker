from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIMELINE_ASSETS = ROOT / "mobile/ios/CutSell/TimelineAssets.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
VISUAL_TIMELINE = ROOT / "mobile/ios/CutSell/VisualTimelineView.swift"


def test_ios_models_match_d279_client_safe_asset_contract():
    source = TIMELINE_ASSETS.read_text()
    for field in (
        'case assetID = "asset_id"',
        'case mediaKind = "media_kind"',
        'case durationSec = "duration_sec"',
        'case hasAudio = "has_audio"',
        'case qualificationStatus = "qualification_status"',
        'case replacesAssetID = "replaces_asset_id"',
    ):
        assert field in source

    # D-279 deliberately excludes server storage/materialization references.
    assert "storage_reference" not in source
    assert "technical_metadata_reference" not in source


def test_ios_registry_load_is_project_and_user_scoped():
    source = TIMELINE_ASSETS.read_text()
    assert '"/v1/projects/\\(projectID)/timeline-assets"' in source
    assert 'URLQueryItem(name: "user_id", value: userID)' in source
    assert "guard response.projectID == projectID" in source


def test_ios_filters_only_ready_assets_into_editable_role_catalogs():
    source = TIMELINE_ASSETS.read_text()
    assert "$0.isReady && $0.role == .supplementalBroll && $0.mediaKind == .video" in source
    assert "$0.isReady && $0.role == .voiceOver && $0.mediaKind == .audio" in source
    assert "$0.isReady && $0.role == .primarySource && $0.mediaKind == .video" in source


def test_draft_load_fetches_d279_registry_with_existing_draft():
    source = VIEW_MODEL.read_text()
    assert "async let draftRequest" in source
    assert "async let assetRequest = TimelineAssetRegistryClient.list" in source
    assert source.index("snapshot = loadedDraft") < source.index("let loadedAssets = try await assetRequest")
    assert "timelineAssetLibrary = loadedAssets" in source


def test_preview_assets_are_not_misrepresented_as_d279_registry_assets():
    asset_source = TIMELINE_ASSETS.read_text()
    view_source = VISUAL_TIMELINE.read_text()
    assert "enum SourcePreviewAssetCatalog" in asset_source
    assert "SourcePreviewAssetCatalog.build(from: model.snapshot)" in view_source
    assert "TimelineAssetCatalog" not in asset_source
    assert "TimelineAssetCatalog" not in view_source
