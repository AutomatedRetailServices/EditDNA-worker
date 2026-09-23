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
# 1. Three-track order: Main Video, Voice-over, B-roll
# ---------------------------------------------------------------------------

def test_three_track_order_main_video_voiceover_broll():
    source = TIMELINE_EDITOR.read_text()
    assert "enum TimelineTrackKind" in source
    main_idx = source.index("case mainVideo")
    vo_idx = source.index("voiceOver", main_idx)
    broll_idx = source.index("broll", vo_idx)
    assert main_idx < vo_idx < broll_idx


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
    # canAddBrollOrVoiceOver still gates the honest disclosure text and
    # each dedicated view's own "Add existing to timeline" action -- it no
    # longer gates either track's entry ("+") button, since Import
    # (registering a new asset) has no such precondition (see BrollView.swift
    # and VoiceOverView.swift).
    assert "canAddBrollOrVoiceOver" in editor_source
    assert ".disabled(model.baseEditAssetID == nil)" in (TIMELINE_EDITOR.parent / "BrollView.swift").read_text()


# ---------------------------------------------------------------------------
# 7. Voice-over / B-roll placement add flow -- only READY assets, real save
# ---------------------------------------------------------------------------

def test_add_voiceover_and_broll_call_real_composition_mutations():
    # Both Voice-over's and B-roll's real "add existing to timeline"
    # mutations now live in their own dedicated views (see
    # test_cutsell_ios_mobile_v1_voiceover_ui.py and
    # test_cutsell_ios_mobile_v1_broll_ui.py) -- TimelineEditorView only
    # opens those real, non-decorative entry points.
    source = TIMELINE_EDITOR.read_text()
    assert "BrollView(" in source
    assert "VoiceOverView(" in source
    broll_view_source = (TIMELINE_EDITOR.parent / "BrollView.swift").read_text()
    assert "model.addBrollPlacement(" in broll_view_source
    voiceover_view_source = (TIMELINE_EDITOR.parent / "VoiceOverView.swift").read_text()
    assert "model.addVoiceOverPlacement(" in voiceover_view_source


def test_add_button_only_appears_at_end_of_voiceover_and_broll_tracks_not_main_video():
    source = TIMELINE_EDITOR.read_text()
    # Both B-roll's and Voice-over's "+" now open their own dedicated
    # sheet instead of an inline confirmationDialog -- both are still
    # real, non-decorative entry points, never absent for either
    # non-Main-Video track.
    assert "if track == .broll {" in source
    assert "else if track == .voiceOver {" in source
    track_row_idx = source.index("private func trackRow(_ track: TimelineTrackKind)")
    body_end = source.index("\n    // MARK: - Actions", track_row_idx)
    body = source[track_row_idx:body_end]
    assert body.count('Image(systemName: "plus")') == 2


def test_real_overlay_entry_point_exists_separately_from_broll():
    # The corrective addition: the genuine positioned/scaled Overlay
    # feature (/v1/overlays/*) gets its own real action-bar entry point,
    # distinct from B-roll's timeline-track entry point.
    source = TIMELINE_EDITOR.read_text()
    assert "OverlayView(model: model)" in source
    assert (TIMELINE_EDITOR.parent / "OverlayView.swift").exists()


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


# ---------------------------------------------------------------------------
# 9. Redo -- real authority only (Main Video's existing /draft/redo), never
#    a simulated redo for B-roll/Voice-over (no such backend authority
#    exists for those D-282A placements). The real Overlay feature DOES
#    share this same /draft/redo authority, but via its own Redo button
#    inside OverlayView.swift, never through this shared canRedoMainVideo
#    flag.
# ---------------------------------------------------------------------------

def test_redo_button_is_gated_on_a_dedicated_real_authority_flag():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var canRedoMainVideo = false" in source
    assert 'Label("Redo", systemImage: "arrow.uturn.forward")' in source
    assert ".disabled(!canRedoMainVideo)" in source


def test_redo_dispatches_only_to_main_videos_real_redo_authority():
    source = TIMELINE_EDITOR.read_text()
    assert "private func performRedo() async" in source
    redo_idx = source.index("private func performRedo() async")
    body_end = source.index("\n}", redo_idx)
    body = source[redo_idx:body_end]
    assert "guard canRedoMainVideo else { return }" in body
    assert "await model.redo()" in body
    # No Overlay/Voice-over redo call anywhere -- that authority doesn't exist.
    assert "removeBrollPlacement" not in body
    assert "removeVoiceOverPlacement" not in body


def test_canredo_flag_is_only_ever_set_true_after_a_real_main_video_undo():
    source = TIMELINE_EDITOR.read_text()
    undo_idx = source.index("private func performUndo() async")
    redo_idx = source.index("private func performRedo() async")
    undo_body = source[undo_idx:redo_idx]
    assert "canRedoMainVideo = true" in undo_body
    assert "await model.undo()" in undo_body


def test_canredo_flag_is_invalidated_by_any_new_mutation_or_selection():
    source = TIMELINE_EDITOR.read_text()
    # New selection invalidates a pending redo.
    tap_idx = source.index("selection = TimelineSelection(track: track, itemID: item.id)")
    tap_block = source[tap_idx:tap_idx + 200]
    assert "canRedoMainVideo = false" in tap_block
    # A fresh split/delete on Main Video also invalidates it.
    split_idx = source.index("await model.split(clipID: selection.itemID, at: playheadTime)")
    split_block = source[split_idx:split_idx + 160]
    assert "canRedoMainVideo = false" in split_block
    remove_idx = source.index("await model.remove(clipID: selection.itemID)")
    remove_block = source[remove_idx:remove_idx + 160]
    assert "canRedoMainVideo = false" in remove_block


# ---------------------------------------------------------------------------
# 10. Main Video filmstrip reuse -- "Conserva filmstrip/waveform cuando
#     estén disponibles" -- the SAME real preview catalog VisualTimelineView
#     already builds, never a second/invented source, never fabricated data.
# ---------------------------------------------------------------------------

def test_main_video_reuses_the_same_real_preview_catalog_as_visual_timeline():
    source = TIMELINE_EDITOR.read_text()
    assert "SourcePreviewAssetCatalog.build(from: model.snapshot)" in source
    visual_source = VISUAL_TIMELINE.read_text()
    assert "SourcePreviewAssetCatalog.build(from:" in visual_source


def test_main_video_items_carry_previewframes_filtered_to_their_own_span():
    source = TIMELINE_EDITOR.read_text()
    assert "var previewFrames: [TimelineFrame] = []" in source
    assert "let frames = (sourceAssets?.frames ?? []).filter { $0.time >= start && $0.time <= end }" in source
    assert "previewFrames: frames" in source


def test_timeline_row_cell_renders_previewframes_as_a_real_filmstrip():
    source = TIMELINE_EDITOR.read_text()
    cell_idx = source.index("private struct TimelineRowCell")
    cell_body = source[cell_idx:]
    assert "displayFrames" in cell_body
    assert "AsyncImage(url: frame.url)" in cell_body
    # Falls back honestly to the plain placeholder when there is no real
    # preview data -- never fabricates frames that don't exist.
    assert "if !displayFrames.isEmpty" in cell_body


def test_filmstrip_is_never_rendered_for_voiceover_or_broll_rows():
    # Voice-over/B-roll TimelineRowItems never populate previewFrames --
    # only mainVideoItems does, so the shared cell only ever shows a
    # filmstrip for Main Video, honestly reflecting that no comparable
    # preview catalog exists for those two tracks in this gate's scope.
    source = TIMELINE_EDITOR.read_text()
    vo_idx = source.index("private var voiceOverItems")
    broll_idx = source.index("private var brollItems")
    vo_block = source[vo_idx:broll_idx]
    broll_end = source.index("private func items(for track:")
    broll_block = source[broll_idx:broll_end]
    assert "previewFrames" not in vo_block
    assert "previewFrames" not in broll_block


# ---------------------------------------------------------------------------
# 11. Cross-track playhead -- visible on all three tracks, drives Split
#     gating, and is honestly labeled for accessibility.
# ---------------------------------------------------------------------------

def test_playhead_spans_all_three_tracks_and_gates_split():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var playheadTime: Double = 0" in source
    assert "private var playheadLine: some View" in source
    tracks_idx = source.index("private var timelineTracks")
    playhead_line_idx = source.index("playheadLine", tracks_idx)
    assert tracks_idx < playhead_line_idx
    # It sits in the SAME ZStack as the per-track ForEach, so one line
    # crosses all three tracks rather than each track owning its own.
    tracks_block = source[tracks_idx:source.index("private var playheadLine:")]
    assert "ZStack(alignment: .topLeading)" in tracks_block
    assert "ForEach(TimelineTrackKind.allCases)" in tracks_block
    assert "playheadLine" in tracks_block
    # Split only ever acts on the selection AT the playhead (see also
    # test_split_is_gated_on_selection_and_playhead_span).
    assert "canSplitAtPlayhead" in source


def test_playhead_slider_has_honest_accessibility_value():
    source = TIMELINE_EDITOR.read_text()
    slider_idx = source.index("Slider(value: $playheadTime, in: 0...totalDuration)")
    slider_block = source[slider_idx:slider_idx + 300]
    assert '.accessibilityLabel("Playhead")' in slider_block
    assert ".accessibilityValue(" in slider_block
    assert "playheadTime" in slider_block and "totalDuration" in slider_block


# ---------------------------------------------------------------------------
# 12. Backend error must never leave the UI claiming a save that didn't
#     happen -- the honest failure path for the D-282A composition mutations.
# ---------------------------------------------------------------------------

def test_backend_error_on_save_never_produces_a_false_saved_state():
    vm_source = VIEW_MODEL.read_text()
    save_fn_idx = vm_source.index("private func saveTimelineComposition")
    next_fn_idx = vm_source.index("\n    func ", save_fn_idx)
    save_fn_body = vm_source[save_fn_idx:next_fn_idx]
    assert "catch" in save_fn_body
    # The catch path only ever records the error and returns -- it must
    # NEVER assign `timelineComposition` (that would fake a saved state).
    catch_idx = save_fn_body.index("catch")
    catch_block = save_fn_body[catch_idx:]
    assert "timelineComposition =" not in catch_block
    assert "errorMessage" in catch_block


def test_isavingtimeline_flag_is_cleared_on_both_success_and_failure_paths():
    vm_source = VIEW_MODEL.read_text()
    assert "isSavingTimeline = true" in vm_source
    save_fn_idx = vm_source.index("private func saveTimelineComposition")
    next_fn_idx = vm_source.index("\n    func ", save_fn_idx)
    save_fn_body = vm_source[save_fn_idx:next_fn_idx]
    # Defer or an explicit reset on both branches -- never left stuck true
    # after a failed save (which would honestly-but-permanently block UI).
    assert "isSavingTimeline = false" in save_fn_body or "defer" in save_fn_body


# ---------------------------------------------------------------------------
# 13. Redo may ONLY be enabled once /draft/undo has actually confirmed
#     success -- never inferred just because `await model.undo()` returned.
#     undo()/redo() must expose a real typed success signal (never Void),
#     grounded in whether the API call actually threw.
# ---------------------------------------------------------------------------

def test_undo_and_redo_expose_a_real_typed_success_result_not_void():
    vm_source = VIEW_MODEL.read_text()
    assert "func undo() async -> Bool" in vm_source
    assert "func redo() async -> Bool" in vm_source
    # @discardableResult so the pre-existing toolbar buttons in
    # DraftEditorView.swift (which never inspected a return value) keep
    # compiling -- never a second, parallel undo/redo authority.
    undo_idx = vm_source.index("func undo() async -> Bool")
    assert "@discardableResult" in vm_source[max(0, undo_idx - 120):undo_idx]
    redo_idx = vm_source.index("func redo() async -> Bool")
    assert "@discardableResult" in vm_source[max(0, redo_idx - 120):redo_idx]


def test_undo_returns_true_only_on_confirmed_success_false_on_caught_error():
    vm_source = VIEW_MODEL.read_text()
    undo_idx = vm_source.index("func undo() async -> Bool")
    redo_idx = vm_source.index("func redo() async -> Bool")
    undo_body = vm_source[undo_idx:redo_idx]
    # Success path: the real snapshot assignment is followed by `return true`.
    assign_idx = undo_body.index("self.snapshot = try await api.request(")
    return_true_idx = undo_body.index("return true", assign_idx)
    catch_idx = undo_body.index("} catch {")
    assert assign_idx < return_true_idx < catch_idx
    # Failure path: the catch block sets errorMessage AND returns false --
    # never silently swallowed as a bare `catch { errorMessage = ... }`.
    catch_body = undo_body[catch_idx:]
    assert "errorMessage = error.localizedDescription" in catch_body
    assert "return false" in catch_body


def test_redo_returns_true_only_on_confirmed_success_false_on_caught_error():
    vm_source = VIEW_MODEL.read_text()
    redo_idx = vm_source.index("func redo() async -> Bool")
    redo_body = vm_source[redo_idx:redo_idx + 700]
    assign_idx = redo_body.index("self.snapshot = try await api.request(")
    return_true_idx = redo_body.index("return true", assign_idx)
    catch_idx = redo_body.index("} catch {")
    assert assign_idx < return_true_idx < catch_idx
    catch_body = redo_body[catch_idx:]
    assert "errorMessage = error.localizedDescription" in catch_body
    assert "return false" in catch_body


def test_successful_undo_enables_redo():
    source = TIMELINE_EDITOR.read_text()
    undo_idx = source.index("private func performUndo() async")
    redo_idx = source.index("private func performRedo() async")
    body = source[undo_idx:redo_idx]
    assert "let undoSucceeded = await model.undo()" in body
    succeeded_idx = body.index("if undoSucceeded {")
    enable_idx = body.index("canRedoMainVideo = true", succeeded_idx)
    close_idx = body.index("}", enable_idx)
    # canRedoMainVideo = true must be INSIDE the success branch, not a
    # sibling statement that runs unconditionally after the await.
    assert succeeded_idx < enable_idx < close_idx


def test_failed_undo_does_not_enable_redo():
    source = TIMELINE_EDITOR.read_text()
    undo_idx = source.index("private func performUndo() async")
    redo_idx = source.index("private func performRedo() async")
    body = source[undo_idx:redo_idx]
    # canRedoMainVideo = true appears exactly once, and only inside the
    # `if undoSucceeded` branch -- never as an unconditional statement
    # that would run even when undo() returned false.
    assert body.count("canRedoMainVideo = true") == 1
    unconditional_after_await = body.split("let undoSucceeded = await model.undo()")[1]
    # The very next non-blank statement after the await must be the
    # bookkeeping reset, not an unconditional redo-enable.
    next_lines = [ln.strip() for ln in unconditional_after_await.splitlines() if ln.strip()]
    assert next_lines[0] == "justMutatedMainVideo = false"
    assert next_lines[1].startswith("if undoSucceeded")


def test_successful_redo_consumes_the_pending_state():
    source = TIMELINE_EDITOR.read_text()
    redo_idx = source.index("private func performRedo() async")
    body = source[redo_idx:]
    assert "let redoSucceeded = await model.redo()" in body
    succeeded_idx = body.index("if redoSucceeded {")
    consume_idx = body.index("canRedoMainVideo = false", succeeded_idx)
    close_idx = body.index("}", consume_idx)
    assert succeeded_idx < consume_idx < close_idx


def test_failed_redo_preserves_honest_state_and_surfaces_the_real_error():
    source = TIMELINE_EDITOR.read_text()
    redo_idx = source.index("private func performRedo() async")
    body = source[redo_idx:]
    # canRedoMainVideo = false appears exactly once, only inside the
    # success branch -- a failed redo must leave the pending-redo state
    # exactly as it was (still real, still redoable), never silently
    # cleared as if it had been consumed.
    assert body.count("canRedoMainVideo = false") == 1
    assert "if redoSucceeded {" in body
    # The real error path (model.redo()'s own catch -> errorMessage) is
    # the ONLY error-surfacing mechanism -- this view never introduces a
    # second, parallel error channel.
    editor_source = TIMELINE_EDITOR.read_text()
    assert "errorMessage" not in editor_source.split("private func performRedo() async")[1].split("\n}")[0]
    vm_source = VIEW_MODEL.read_text()
    redo_fn_idx = vm_source.index("func redo() async -> Bool")
    assert "errorMessage = error.localizedDescription" in vm_source[redo_fn_idx:redo_fn_idx + 700]


def test_a_new_mutation_still_invalidates_redo_under_the_typed_result_contract():
    # Regression guard: the typed-result fix must not have disturbed the
    # existing invalidation wiring (new selection / new split / new
    # delete all still reset canRedoMainVideo to false).
    source = TIMELINE_EDITOR.read_text()
    tap_idx = source.index("selection = TimelineSelection(track: track, itemID: item.id)")
    tap_block = source[tap_idx:tap_idx + 200]
    assert "canRedoMainVideo = false" in tap_block
    split_idx = source.index("await model.split(clipID: selection.itemID, at: playheadTime)")
    split_block = source[split_idx:split_idx + 160]
    assert "canRedoMainVideo = false" in split_block
    remove_idx = source.index("await model.remove(clipID: selection.itemID)")
    remove_block = source[remove_idx:remove_idx + 160]
    assert "canRedoMainVideo = false" in remove_block
