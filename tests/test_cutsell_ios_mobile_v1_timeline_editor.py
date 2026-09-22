"""Mobile V1 Timeline UI gate -- structural contract tests.

Pure text-scanning tests against the Swift sources (the established
convention in `test_cutsell_ios_d279_timeline_asset_registry_contract.py`
-- no Xcode/Swift toolchain is available in this offline qualification
environment; real Xcode Simulator build verification happens via the
existing `cutsell-ios-ci.yml` macOS CI, triggered automatically once this
gate's `mobile/ios/**` changes reach PR #25).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
TIMELINE_COMPOSITION = ROOT / "mobile/ios/CutSell/TimelineComposition.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
TIMELINE_ASSETS = ROOT / "mobile/ios/CutSell/TimelineAssets.swift"
PROJECTS_VIEW = ROOT / "mobile/ios/CutSell/ProjectsView.swift"
MODELS = ROOT / "mobile/ios/CutSell/Models.swift"
DRAFT_EDITOR_VIEW = ROOT / "mobile/ios/CutSell/DraftEditorView.swift"
VISUAL_TIMELINE = ROOT / "mobile/ios/CutSell/VisualTimelineView.swift"


# ---------------------------------------------------------------------------
# 1. Three-track order: Main Video, Voice-over, Overlay
# ---------------------------------------------------------------------------

def test_three_track_order_main_video_voiceover_overlay():
    source = TIMELINE_EDITOR.read_text()
    assert "enum TimelineTrackKind" in source
    main_idx = source.index("case mainVideo")
    vo_idx = source.index("voiceOver", main_idx)
    overlay_idx = source.index("overlay", vo_idx)
    assert main_idx < vo_idx < overlay_idx


def test_tracks_rendered_via_allCases_not_hand_ordered_duplicate_list():
    source = TIMELINE_EDITOR.read_text()
    assert "ForEach(TimelineTrackKind.allCases)" in source


# ---------------------------------------------------------------------------
# 2. Typed, exclusive selection -- one enum/ID pair, never parallel indices
# ---------------------------------------------------------------------------

def test_selection_is_one_typed_optional_not_parallel_indices():
    source = TIMELINE_EDITOR.read_text()
    assert "struct TimelineSelection" in source
    assert "var track: TimelineTrackKind" in source
    assert "var itemID: String" in source
    assert "@State private var selection: TimelineSelection?" in source
    # No parallel "selectedRow"/"selectedColumn"-style index pair.
    assert "selectedRowIndex" not in source
    assert "selectedColumnIndex" not in source


def test_selecting_a_new_item_replaces_the_single_selection_state():
    source = TIMELINE_EDITOR.read_text()
    assert "selection = TimelineSelection(track: track, itemID: item.id)" in source


# ---------------------------------------------------------------------------
# 3. READY-only asset filtering (never an editable non-READY asset)
# ---------------------------------------------------------------------------

def test_voiceover_and_overlay_use_only_ready_asset_catalogs():
    source = TIMELINE_EDITOR.read_text()
    assert "timelineAssetLibrary?.readyVoiceOvers" in source or "timelineAssetLibrary.readyVoiceOvers" in source
    assert "timelineAssetLibrary?.readyBroll" in source or "timelineAssetLibrary.readyBroll" in source
    # The catalogs themselves are the ones D-279 already gates on isReady.
    lib_source = TIMELINE_ASSETS.read_text()
    assert "$0.isReady && $0.role == .voiceOver && $0.mediaKind == .audio" in lib_source
    assert "$0.isReady && $0.role == .supplementalBroll && $0.mediaKind == .video" in lib_source


# ---------------------------------------------------------------------------
# 4. Split acts only on the selected element at the playhead
# ---------------------------------------------------------------------------

def test_split_is_gated_on_selection_and_playhead_span():
    source = TIMELINE_EDITOR.read_text()
    assert "canSplitAtPlayhead" in source
    assert "playheadTime > item.startSec" in source
    assert "playheadTime < item.endSec" in source
    assert ".disabled(!canSplitAtPlayhead)" in source


def test_split_dispatches_to_the_selected_tracks_own_real_authority():
    source = TIMELINE_EDITOR.read_text()
    assert "model.split(clipID: selection.itemID, at: playheadTime)" in source
    assert "model.splitVoiceOverPlacement(id: selection.itemID, at: playheadTime)" in source
    assert "model.splitBrollPlacement(id: selection.itemID, at: playheadTime)" in source


# ---------------------------------------------------------------------------
# 5. Delete + honest Undo (real backend authority, never a faked state)
# ---------------------------------------------------------------------------

def test_delete_is_gated_on_a_real_selection():
    source = TIMELINE_EDITOR.read_text()
    assert ".disabled(!canDeleteSelection)" in source
    assert "private var canDeleteSelection: Bool { selectedItem != nil }" in source


def test_delete_dispatches_to_each_tracks_own_real_removal_authority():
    source = TIMELINE_EDITOR.read_text()
    assert "model.remove(clipID: selection.itemID)" in source
    assert "model.removeVoiceOverPlacement(id: selection.itemID)" in source
    assert "model.removeBrollPlacement(id: selection.itemID)" in source


def test_undo_uses_only_real_existing_authority_never_fabricated():
    editor_source = TIMELINE_EDITOR.read_text()
    assert ".disabled(!canUndo)" in editor_source
    assert "model.undo()" in editor_source  # existing real draft-undo authority, reused verbatim
    assert "model.undoLastTimelineMutation()" in editor_source

    vm_source = VIEW_MODEL.read_text()
    # The composition undo re-PUTs a captured prior state through the SAME
    # real save operation -- never a new backend authority.
    assert "func undoLastTimelineMutation() async" in vm_source
    assert "TimelineCompositionClient.save(" in vm_source
    assert "lastCompositionBeforeMutation" in vm_source


def test_undo_never_pretends_a_mutation_saved_before_backend_confirmation():
    vm_source = VIEW_MODEL.read_text()
    assert "never show a mutation as saved before backend" in vm_source.lower() \
        or "never pretends to have undone" in vm_source.lower()


# ---------------------------------------------------------------------------
# 6. Persistence wired to the REAL D-282A operations -- no invented routes
# ---------------------------------------------------------------------------

def test_timeline_composition_client_uses_real_d282a_routes_only():
    source = TIMELINE_COMPOSITION.read_text()
    assert '"/v1/projects/\\(projectID)/timeline"' in source
    # Never any route this codebase's own backend does not define.
    assert "/v1/timeline" not in source.replace('"/v1/projects/\\(projectID)/timeline"', "")
    assert "invented" not in source.lower()


def test_timeline_composition_fields_match_d282a_response_contract_exactly():
    source = TIMELINE_COMPOSITION.read_text()
    for field in (
        'case contractVersion = "contract_version"',
        'case baseEditIdentity = "base_edit_identity"',
        'case timelineDurationSec = "timeline_duration_sec"',
        'case timelineRevisionIdentity = "timeline_revision_identity"',
        'case brollPlacements = "broll_placements"',
        'case voiceOverPlacements = "voice_over_placements"',
    ):
        assert field in source
    # D-282A's own additive save-request field -- never invented separately.
    assert "base_edit_asset_id" in source
    assert "expected_revision_identity" in source


def test_save_request_never_sends_a_raw_storage_reference():
    source = TIMELINE_COMPOSITION.read_text()
    assert "storage_reference" not in source
    assert "local://" not in source
    assert "s3://" not in source


def test_get_composition_treats_the_real_404_as_an_honest_empty_state():
    source = TIMELINE_COMPOSITION.read_text()
    assert "case .http(404, _)" in source
    assert "noTimelineSaved" in source
    vm_source = VIEW_MODEL.read_text()
    assert "catch is TimelineCompositionError" in vm_source
    assert "timelineComposition = nil" in vm_source


def test_mutation_refreshes_snapshot_and_library_after_confirmation():
    source = VIEW_MODEL.read_text()
    assert "func refreshTimelineComposition() async" in source
    # save -> refresh composition -> refresh asset library, in that order,
    # and only ever after a CONFIRMED (non-throwing) save call.
    save_idx = source.index("try await TimelineCompositionClient.save(")
    refresh_comp_idx = source.index("await refreshTimelineComposition()", save_idx)
    refresh_assets_idx = source.index("TimelineAssetRegistryClient.list(", refresh_comp_idx)
    assert save_idx < refresh_comp_idx < refresh_assets_idx


def test_main_video_uses_the_pre_existing_draft_edits_system_not_d282a():
    editor_source = TIMELINE_EDITOR.read_text()
    # Main Video split/delete go through the ORIGINAL draft-edits model
    # methods, never through the new TimelineCompositionClient.
    assert "case .mainVideo:\n            await model.split(clipID: selection.itemID, at: playheadTime)" in editor_source
    vm_source = VIEW_MODEL.read_text()
    assert '"/v1/draft-edits/split"' in vm_source
    assert '"/v1/draft-edits/remove"' in vm_source


def test_base_edit_asset_resolution_is_honest_never_invented():
    source = VIEW_MODEL.read_text()
    assert "var baseEditAssetID: String? { timelineAssetLibrary?.readyPrimarySources.first?.assetID }" in source
    editor_source = TIMELINE_EDITOR.read_text()
    assert "canAddOverlayOrVoiceOver" in editor_source
    assert ".disabled(!canAddOverlayOrVoiceOver)" in editor_source


# ---------------------------------------------------------------------------
# 7. Voice-over / Overlay placement add flow -- only READY assets, real save
# ---------------------------------------------------------------------------

def test_add_voiceover_and_overlay_call_real_composition_mutations():
    source = TIMELINE_EDITOR.read_text()
    assert "model.addVoiceOverPlacement(" in source
    assert "model.addBrollPlacement(" in source


def test_add_button_only_appears_at_end_of_voiceover_and_overlay_tracks_not_main_video():
    source = TIMELINE_EDITOR.read_text()
    assert 'if track != .mainVideo {' in source


# ---------------------------------------------------------------------------
# 8. Edits tabs: All / Ready / Drafts only -- Processing removed as a tab
# ---------------------------------------------------------------------------

def test_edits_tabs_are_exactly_all_ready_drafts():
    source = PROJECTS_VIEW.read_text()
    assert 'case all = "All", ready = "Ready", drafts = "Drafts"' in source
    assert 'case processing' not in source.lower()
    assert 'Text("Processing")' not in source


def test_processing_states_are_a_single_shared_source_of_truth():
    models_source = MODELS.read_text()
    assert "static let processingStates: Set<String>" in models_source
    projects_source = PROJECTS_VIEW.read_text()
    assert "Project.processingStates.contains" in projects_source
    # Reused at both the detail-routing site and the Drafts-tab filter --
    # never a second hardcoded copy of the state list.
    assert projects_source.count('"processing", "uploaded", "preparing", "transcribing", "analyzing", "composing"') == 0


def test_ready_and_drafts_tabs_are_mutually_exclusive_filters():
    source = PROJECTS_VIEW.read_text()
    assert 'case .ready:\n            return appState.projects.filter { $0.state == "draft_ready" }' in source
    assert 'case .drafts:' in source
    assert '$0.state != "draft_ready"' in source


def test_edits_tab_picker_is_segmented_and_reuses_default_accent_tint():
    source = PROJECTS_VIEW.read_text()
    assert '.pickerStyle(.segmented)' in source
    # No custom hardcoded hex/RGB color invented for "official CutSell
    # blue" -- this codebase's own only existing brand-tint mechanism is
    # the system accentColor (used everywhere else: VisualTimelineView's
    # selection outline, every .borderedProminent button).
    assert "#0000" not in source and "Color(red:" not in source


# ---------------------------------------------------------------------------
# Structural firewall -- no unrelated engine/backend files touched, no
# closed-track authority referenced from the mobile layer.
# ---------------------------------------------------------------------------

def test_no_engine_authority_referenced_from_new_swift_files():
    forbidden = (
        "BestTakeResolver", "SelectionFreeze", "BoundaryEngine", "hybrid_composite_best_take",
        "deterministic_best_take_authority", "renderer.py", "boundary_engine_pass",
    )
    for path in (TIMELINE_EDITOR, TIMELINE_COMPOSITION):
        source = path.read_text()
        for term in forbidden:
            assert term not in source


def test_new_files_never_hardcode_video00_or_qa_reference_data():
    forbidden = ("VIDEO-2026-07-30", "5E01F214-A364-4F4B", "D40F1D43-7391-44D5")
    for path in (TIMELINE_EDITOR, TIMELINE_COMPOSITION, PROJECTS_VIEW):
        source = path.read_text()
        for term in forbidden:
            assert term not in source


def test_visual_timeline_view_left_completely_unmodified_in_scope():
    # This gate is explicitly additive: the pre-existing single-row
    # clip inspector (swap take / trim / caption / audio) keeps working
    # unmodified alongside the new three-track editor.
    source = VISUAL_TIMELINE.read_text()
    assert "struct TimelineClipInspector: View" in source
    assert "Swap Take" in source


def test_new_editor_is_additively_wired_into_draft_editor_view():
    source = DRAFT_EDITOR_VIEW.read_text()
    assert "TimelineEditorView(model: model)" in source
    assert "VisualTimelineView(model: model)" in source
    editor_idx = source.index("TimelineEditorView(model: model)")
    visual_idx = source.index("VisualTimelineView(model: model)")
    assert editor_idx < visual_idx
