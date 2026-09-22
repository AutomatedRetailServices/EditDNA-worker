"""Mobile V1 Voice-over UI gate -- structural contract tests.

Pure text-scanning tests against the Swift sources (the established
convention in this repo -- no Xcode/Swift toolchain is available in this
offline qualification environment; real Xcode Simulator build verification
happens via the existing `cutsell-ios-ci.yml` macOS CI, dispatched manually
for this isolated feature branch).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICEOVER_VIEW = ROOT / "mobile/ios/CutSell/VoiceOverView.swift"
IMPORT_MANAGER = ROOT / "mobile/ios/CutSell/VoiceOverImportManager.swift"
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
TIMELINE_COMPOSITION = ROOT / "mobile/ios/CutSell/TimelineComposition.swift"
MAIN_APP = ROOT / "cutsell_app/timeline_routes.py"
ASSET_REGISTRY_STORE = ROOT / "cutsell_worker/timeline_asset_registry_store.py"
UPLOADS = ROOT / "cutsell_worker/uploads.py"
TIMELINE_COMPOSITION_PY = ROOT / "cutsell_worker/timeline_composition.py"


# ---------------------------------------------------------------------------
# 1. Entry point from Timeline -- real, non-decorative, additive.
# ---------------------------------------------------------------------------

def test_voiceover_entry_point_wired_additively_into_timeline_editor():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var showVoiceOverView = false" in source
    assert "showVoiceOverPicker" not in source
    assert ".sheet(isPresented: $showVoiceOverView)" in source
    assert "VoiceOverView(" in source


def test_voiceover_entry_point_never_gated_on_placement_precondition():
    # Import has no real base-edit-asset precondition -- the entry button
    # itself must never be disabled on canAddOverlayOrVoiceOver.
    source = TIMELINE_EDITOR.read_text()
    button_idx = source.index('.accessibilityIdentifier("timeline.voiceOverButton")')
    block_start = source.rindex("Button {", 0, button_idx)
    block = source[block_start:button_idx]
    assert "canAddOverlayOrVoiceOver" not in block


def test_voiceover_sheet_passes_current_selection_honestly():
    source = TIMELINE_EDITOR.read_text()
    sheet_idx = source.index(".sheet(isPresented: $showVoiceOverView)")
    sheet_end = source.index("\n        }", sheet_idx)
    sheet_body = source[sheet_idx:sheet_end]
    assert "selection?.track == .voiceOver ? selection?.itemID : nil" in sheet_body


# ---------------------------------------------------------------------------
# 2. Import -- real, complete, already-registered upload+registration
#    routes; never a fabricated route or persistence.
# ---------------------------------------------------------------------------

def test_import_manager_uses_only_real_registered_routes():
    source = IMPORT_MANAGER.read_text()
    assert '"/v1/projects/\\(projectID)/timeline-uploads"' in source
    assert '"/v1/projects/\\(projectID)/timeline-assets"' in source
    routes_source = MAIN_APP.read_text()
    assert '@router.post("/{project_id}/timeline-uploads"' in routes_source
    assert '@router.post("/{project_id}/timeline-assets"' in routes_source


def test_import_uses_real_media_class_and_role_strings():
    source = IMPORT_MANAGER.read_text()
    assert 'media_class: "voice_over"' in source
    assert 'role: "VOICE_OVER"' in source
    assert 'media_kind: "AUDIO"' in source
    # Confirm these are the backend's own real enum values, not guessed.
    upload_reg = ROOT.joinpath("cutsell_worker/timeline_upload_registration.py").read_text()
    assert 'MEDIA_CLASS_VOICE_OVER = "voice_over"' in upload_reg


def test_import_ingestion_is_documented_as_synchronous_never_a_fake_poll():
    source = IMPORT_MANAGER.read_text()
    assert "synchronously" in source.lower() or "no polling" in source.lower()
    store_source = ASSET_REGISTRY_STORE.read_text()
    assert "status = reg.TimelineAssetQualificationStatus.READY" in store_source


def test_import_never_calls_apiclient_upload_helper_or_invents_a_session_holder():
    source = IMPORT_MANAGER.read_text()
    # The real S3 presigned-POST needs multipart/form-data with `fields` --
    # APIClient.upload() is a mismatched single-header PUT helper.
    assert "api.upload(" not in source
    assert "APIClientSessionHolder" not in source


def test_recording_is_documented_as_not_implemented_never_simulated():
    source = IMPORT_MANAGER.read_text()
    assert "microphone" in source.lower()
    assert "not implemented" in source.lower() or "never" in source.lower()
    # No AVFoundation recording symbols anywhere.
    forbidden = ("AVAudioRecorder", "AVAudioSession.sharedInstance", "startRecording")
    for term in forbidden:
        assert term not in source
        assert term not in VOICEOVER_VIEW.read_text()


def test_import_flow_uses_real_view_model_method_never_touches_session_directly():
    view_source = VOICEOVER_VIEW.read_text()
    assert "model.importVoiceOverAudio(fileURL: url)" in view_source
    assert "session.userID" not in view_source
    assert "session.accessToken" not in view_source
    vm_source = VIEW_MODEL.read_text()
    assert "func importVoiceOverAudio(fileURL: URL) async" in vm_source
    assert "VoiceOverImportManager.shared.importAudio(" in vm_source


# ---------------------------------------------------------------------------
# 3. Track display, select, add existing, split (edit), delete -- real
#    authority reused, never duplicated/invented.
# ---------------------------------------------------------------------------

def test_track_reads_the_same_real_composition_placements():
    source = VOICEOVER_VIEW.read_text()
    assert "model.timelineComposition?.voiceOverPlacements" in source


def test_add_existing_uses_real_placement_authority():
    source = VOICEOVER_VIEW.read_text()
    assert "model.addVoiceOverPlacement(" in source
    assert ".disabled(model.baseEditAssetID == nil)" in source


def test_split_is_the_real_edit_authority_gated_on_a_genuine_span():
    source = VOICEOVER_VIEW.read_text()
    assert "model.splitVoiceOverPlacement(id: id, at: splitTime)" in source
    assert "func canSplitAtSplitTime" in source
    assert "placement.timelineStartSec + 0.05" in source
    assert "placement.timelineEndSec - 0.05" in source


def test_delete_uses_real_removal_authority():
    source = VOICEOVER_VIEW.read_text()
    assert "model.removeVoiceOverPlacement(id: id)" in source


# ---------------------------------------------------------------------------
# 4. Undo -- real authority, AND the root-cause fix: a failed revert must
#    never silently discard the one recorded undo snapshot.
# ---------------------------------------------------------------------------

def test_undo_uses_real_captured_snapshot_authority():
    source = VOICEOVER_VIEW.read_text()
    assert "model.canUndoTimelineMutation" in source
    assert "model.undoLastTimelineMutation()" in source
    vm_source = VIEW_MODEL.read_text()
    assert "func undoLastTimelineMutation() async" in vm_source


def test_save_timeline_composition_returns_a_real_typed_success_result():
    vm_source = VIEW_MODEL.read_text()
    assert "private func saveTimelineComposition(" in vm_source
    sig_idx = vm_source.index("private func saveTimelineComposition(")
    sig_end = vm_source.index(") async -> Bool", sig_idx)
    assert sig_idx < sig_end
    # @discardableResult immediately precedes it, so the many existing
    # call sites that ignore the return value keep compiling unchanged.
    preceding = vm_source[max(0, sig_idx - 200):sig_idx]
    assert "@discardableResult" in preceding


def test_undo_snapshot_is_only_cleared_on_confirmed_success():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func undoLastTimelineMutation() async")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert "let succeeded = await saveTimelineComposition(" in fn_body
    succeeded_idx = fn_body.index("if succeeded {")
    clear_idx = fn_body.index("lastCompositionBeforeMutation = nil", succeeded_idx)
    assert succeeded_idx < clear_idx
    # The old unconditional-clear-before-save pattern must be gone.
    assert "lastCompositionBeforeMutation = nil\n        await saveTimelineComposition" not in vm_source


def test_failed_revert_leaves_the_snapshot_intact_for_a_retry():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func undoLastTimelineMutation() async")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    # Exactly one clear, and it is INSIDE the success branch -- never a
    # sibling unconditional statement that would run on failure too.
    assert fn_body.count("lastCompositionBeforeMutation = nil") == 1


# ---------------------------------------------------------------------------
# 5. Volume/fade-in/fade-out/pan -- honestly disabled, never simulated.
# ---------------------------------------------------------------------------

def test_no_audio_shaping_field_exists_in_the_real_contract():
    routes_source = MAIN_APP.read_text()
    class_idx = routes_source.index("class VoiceOverPlacementModel")
    class_end = routes_source.index("class TimelineGetResponse")
    class_body = routes_source[class_idx:class_end]
    for forbidden in ("volume", "fade", "pan", "audio_mode"):
        assert forbidden not in class_body.lower()


def test_audio_controls_are_shown_disabled_never_wired_to_a_mutation():
    source = VOICEOVER_VIEW.read_text()
    assert 'Section("Audio (not yet supported)")' in source
    section_idx = source.index('Section("Audio (not yet supported)")')
    section_end = source.index("\n                }", source.index("Section", section_idx + 10))
    section_body = source[section_idx:section_end]
    assert section_body.count(".disabled(true)") >= 4
    # None of these sliders call any mutation method.
    for forbidden in ("setVolume", "setFade", "setPan", "updateAudioMode"):
        assert forbidden not in section_body


def test_v1_audio_behavior_is_documented_as_fixed_not_configurable():
    py_source = TIMELINE_COMPOSITION_PY.read_text()
    assert "not configurable" in py_source.lower()
    swift_source = VOICEOVER_VIEW.read_text()
    assert "not configurable" in swift_source.lower() or "no audio-shaping field" in swift_source.lower()


# ---------------------------------------------------------------------------
# 6. Empty / saving / ready / error states -- explicit, derived from real
#    signals only.
# ---------------------------------------------------------------------------

def test_stage_is_derived_from_real_signals_only():
    source = VOICEOVER_VIEW.read_text()
    assert "enum VoiceOverStage { case empty, saving, ready, error }" in source
    stage_idx = source.index("private var stage: VoiceOverStage")
    stage_end = source.index("\n    }", stage_idx)
    stage_body = source[stage_idx:stage_end]
    assert "model.errorMessage" in stage_body
    assert "model.isSavingTimeline" in stage_body
    assert "placements.isEmpty" in stage_body
    assert "readyAssets.isEmpty" in stage_body


def test_all_four_stages_render_something_real():
    source = VOICEOVER_VIEW.read_text()
    for case in (".empty", ".saving", ".ready", ".error"):
        assert f"case {case}:" in source


def test_backend_errors_are_surfaced_via_the_real_error_alert():
    source = VOICEOVER_VIEW.read_text()
    assert ".alert(\"CutSell\"" in source
    assert "model.errorMessage" in source


# ---------------------------------------------------------------------------
# 7. No Video00 hardcoding; no invented timestamps/presets.
# ---------------------------------------------------------------------------

def test_no_video00_or_qa_reference_data_hardcoded():
    forbidden = ("VIDEO-2026-07-30", "5E01F214-A364-4F4B", "D40F1D43-7391-44D5")
    for path in (VOICEOVER_VIEW, IMPORT_MANAGER):
        source = path.read_text()
        for term in forbidden:
            assert term not in source
