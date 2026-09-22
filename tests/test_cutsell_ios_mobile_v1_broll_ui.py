"""Mobile V1 B-roll UI gate -- structural contract tests.

CORRECTIVE HISTORY: this file was originally
`test_cutsell_ios_mobile_v1_overlay_ui.py`, testing a screen mislabeled
"Overlay" that actually controlled `BrollPlacement` (a full-frame visual
replacement clip -- no position, scale, rotation, or opacity). Renamed to
match what it actually tests. The GENUINE, positioned/scaled Overlay
feature (`/v1/overlays/*`) is tested separately in
`test_cutsell_ios_mobile_v1_overlay_ui.py`; see also
`test_no_broll_screen_ever_claims_positioned_overlay_capability` below,
the regression guard against re-introducing this exact mislabeling.

Pure text-scanning tests against the Swift sources (the established
convention in this repo -- no Xcode/Swift toolchain is available in this
offline qualification environment; real Xcode Simulator build verification
happens via the existing `cutsell-ios-ci.yml` macOS CI, dispatched manually
for this isolated feature branch).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROLL_VIEW = ROOT / "mobile/ios/CutSell/BrollView.swift"
IMPORT_MANAGER = ROOT / "mobile/ios/CutSell/BrollImportManager.swift"
LEGACY_UPLOAD_MANAGER = ROOT / "mobile/ios/CutSell/OverlayUploadManager.swift"
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
TIMELINE_COMPOSITION = ROOT / "mobile/ios/CutSell/TimelineComposition.swift"
MAIN_APP = ROOT / "cutsell_app/timeline_routes.py"
ASSET_REGISTRY_STORE = ROOT / "cutsell_worker/timeline_asset_registry_store.py"
TIMELINE_COMPOSITION_PY = ROOT / "cutsell_worker/timeline_composition.py"
TIMELINE_COMPOSITION_EXECUTOR_PY = ROOT / "cutsell_worker/timeline_composition_executor.py"


# ---------------------------------------------------------------------------
# 1. Entry point from Timeline -- real, non-decorative, additive.
# ---------------------------------------------------------------------------

def test_broll_entry_point_wired_additively_into_timeline_editor():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var showBrollView = false" in source
    assert "showBrollPicker" not in source
    assert ".sheet(isPresented: $showBrollView)" in source
    assert "BrollView(" in source


def test_broll_entry_point_never_gated_on_placement_precondition():
    # Import has no real base-edit-asset precondition -- the entry button
    # itself must never be disabled on canAddBrollOrVoiceOver.
    source = TIMELINE_EDITOR.read_text()
    button_idx = source.index('.accessibilityIdentifier("timeline.brollButton")')
    block_start = source.rindex("Button {", 0, button_idx)
    block = source[block_start:button_idx]
    assert "canAddBrollOrVoiceOver" not in block


def test_broll_sheet_passes_current_selection_honestly():
    source = TIMELINE_EDITOR.read_text()
    sheet_idx = source.index(".sheet(isPresented: $showBrollView)")
    sheet_end = source.index("\n        }", sheet_idx)
    sheet_body = source[sheet_idx:sheet_end]
    assert "selection?.track == .broll ? selection?.itemID : nil" in sheet_body


def test_track_kind_is_honestly_named_broll_not_overlay():
    source = TIMELINE_EDITOR.read_text()
    assert "case mainVideo, voiceOver, broll" in source
    assert 'case .broll: return "B-roll"' in source
    # No stray old case name.
    assert "case mainVideo, voiceOver, overlay" not in source


# ---------------------------------------------------------------------------
# 2. Import -- real, complete, already-registered upload+registration
#    routes; never a fabricated route or persistence; never the unrelated
#    legacy draft-based overlay system.
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
    assert 'media_class: "video"' in source
    assert 'role: "SUPPLEMENTAL_BROLL"' in source
    assert 'media_kind: "VIDEO"' in source
    routes_source = MAIN_APP.read_text()
    assert "def _media_class_for(" in routes_source


def test_import_never_uses_the_unrelated_legacy_overlay_upload_route():
    # cutsell_app/overlay_routes.py's /v1/overlays/uploads/presign is a
    # DIFFERENT, real, ACTIVE feature (the genuine positioned Overlay --
    # see test_cutsell_ios_mobile_v1_overlay_ui.py) -- never conflated with
    # this D-282A TimelineComposition B-roll track.
    source = IMPORT_MANAGER.read_text()
    assert source.count("api.request(") == 2
    assert "OverlayUploadManager.shared" not in source
    assert ": OverlayUploadManager" not in source
    assert LEGACY_UPLOAD_MANAGER.exists()


def test_import_ingestion_is_documented_as_synchronous_never_a_fake_poll():
    source = IMPORT_MANAGER.read_text()
    assert "synchronous" in source.lower()
    store_source = ASSET_REGISTRY_STORE.read_text()
    assert "def create_video_timeline_asset(" in store_source


def test_import_never_calls_apiclient_upload_helper_or_invents_a_session_holder():
    source = IMPORT_MANAGER.read_text()
    assert "api.upload(" not in source
    assert "APIClientSessionHolder" not in source


def test_only_video_is_offered_never_a_fabricated_image_broll_path():
    source = IMPORT_MANAGER.read_text()
    assert "allowedContentTypes" not in source  # picker lives in the View
    view_source = BROLL_VIEW.read_text()
    assert "allowedContentTypes: [.movie]" in view_source
    assert "[.image]" not in view_source
    registry_source = ROOT.joinpath("cutsell_worker/timeline_asset_registry.py").read_text()
    kind_idx = registry_source.index("class TimelineMediaKind")
    kind_body = registry_source[kind_idx:kind_idx + 200]
    assert "IMAGE" not in kind_body


def test_import_flow_uses_real_view_model_method_never_touches_session_directly():
    view_source = BROLL_VIEW.read_text()
    assert "model.importBrollVideo(fileURL: url)" in view_source
    assert "session.userID" not in view_source
    assert "session.accessToken" not in view_source
    vm_source = VIEW_MODEL.read_text()
    assert "func importBrollVideo(fileURL: URL) async" in vm_source
    assert "BrollImportManager.shared.importVideo(" in vm_source


# ---------------------------------------------------------------------------
# 3. Track display, select, add existing, split (edit), delete -- real
#    authority reused, never duplicated/invented.
# ---------------------------------------------------------------------------

def test_track_reads_the_same_real_composition_placements():
    source = BROLL_VIEW.read_text()
    assert "model.timelineComposition?.brollPlacements" in source


def test_add_existing_uses_real_placement_authority():
    source = BROLL_VIEW.read_text()
    assert "model.addBrollPlacement(" in source
    assert ".disabled(model.baseEditAssetID == nil)" in source


def test_split_is_the_real_edit_authority_gated_on_a_genuine_span():
    source = BROLL_VIEW.read_text()
    assert "model.splitBrollPlacement(id: id, at: splitTime)" in source
    assert "func canSplitAtSplitTime" in source
    assert "placement.timelineStartSec + 0.05" in source
    assert "placement.timelineEndSec - 0.05" in source


def test_delete_uses_real_removal_authority():
    source = BROLL_VIEW.read_text()
    assert "model.removeBrollPlacement(id: id)" in source


# ---------------------------------------------------------------------------
# 4. Audio mode -- the ONE real, backend-accepted editable field besides
#    timing. Also the regression guard for the pre-existing raw-value bug
#    this gate found and fixed (MUTE_BOTH -> MUTE_BROLL_AUDIO).
# ---------------------------------------------------------------------------

def test_timeline_audio_mode_enum_matches_the_real_backend_values_exactly():
    swift_source = TIMELINE_COMPOSITION.read_text()
    assert 'case keepPrimaryVoice = "KEEP_PRIMARY_VOICE"' in swift_source
    assert 'case useBrollAudio = "USE_BROLL_AUDIO"' in swift_source
    assert 'case muteBrollAudio = "MUTE_BROLL_AUDIO"' in swift_source
    assert "MUTE_BOTH" not in swift_source
    assert "muteBoth" not in swift_source
    py_source = TIMELINE_COMPOSITION_PY.read_text()
    assert 'KEEP_PRIMARY_VOICE = "KEEP_PRIMARY_VOICE"' in py_source
    assert 'USE_BROLL_AUDIO = "USE_BROLL_AUDIO"' in py_source
    assert 'MUTE_BROLL_AUDIO = "MUTE_BROLL_AUDIO"' in py_source


def test_set_broll_audio_mode_is_a_real_typed_mutation():
    vm_source = VIEW_MODEL.read_text()
    assert "func setBrollAudioMode(id: String, mode: TimelineAudioMode) async" in vm_source
    fn_idx = vm_source.index("func setBrollAudioMode(id: String, mode: TimelineAudioMode) async")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert "await saveTimelineComposition(" in fn_body


def test_audio_mode_picker_offers_exactly_the_three_real_values():
    source = BROLL_VIEW.read_text()
    assert "model.setBrollAudioMode(id:" in source
    for case in ("TimelineAudioMode.keepPrimaryVoice", "TimelineAudioMode.useBrollAudio", "TimelineAudioMode.muteBrollAudio"):
        assert case in source


def test_audio_mode_render_nuance_is_documented_honestly():
    executor_source = TIMELINE_COMPOSITION_EXECUTOR_PY.read_text()
    assert "MUTE_BROLL_AUDIO both leave the primary" in executor_source


# ---------------------------------------------------------------------------
# 5. B-roll has NO position/scale/rotation/opacity/keyframe control at all
#    -- not even shown disabled. This is the corrective fix itself: the
#    prior gate showed these disabled inside the (mislabeled) Overlay
#    screen; they never belonged to B-roll's real contract, so they must
#    not appear here in ANY form.
# ---------------------------------------------------------------------------

def test_no_transform_or_keyframe_field_exists_in_the_real_contract():
    routes_source = MAIN_APP.read_text()
    class_idx = routes_source.index("class BrollPlacementModel")
    class_end = routes_source.index("class VoiceOverPlacementModel")
    class_body = routes_source[class_idx:class_end]
    for forbidden in ("position", "scale", "rotation", "opacity", "keyframe", " x:", " y:", "width"):
        assert forbidden not in class_body.lower()


def test_no_position_scale_rotation_opacity_field_exists_on_the_broll_dataclass():
    py_source = TIMELINE_COMPOSITION_PY.read_text()
    class_idx = py_source.index("class BrollPlacement:")
    class_end = py_source.index("class VoiceOverPlacement:")
    class_body = py_source[class_idx:class_end]
    for forbidden in ("position", "scale", "rotation", "opacity", "keyframe"):
        assert forbidden not in class_body.lower()


def test_broll_view_shows_no_transform_control_not_even_disabled():
    # The corrective fix: unlike the prior mislabeled screen, BrollView must
    # not show ANY position/scale/rotation/opacity/keyframe control, since
    # none of those belong to B-roll's real contract at all.
    source = BROLL_VIEW.read_text()
    for forbidden in (
        "Position, scale", "\"Rotation\"", "\"Opacity\"", "\"Keyframes\"",
        "Slider(value: .constant", "not yet supported",
    ):
        assert forbidden not in source


def test_no_broll_screen_ever_claims_positioned_overlay_capability():
    # Regression guard (directive item 8): re-labeling B-roll's full-frame
    # replacement as a positioned "Overlay" must never happen again. The
    # real per-placement fields on BrollPlacement are exactly these six --
    # x/y/width/rotation/opacity are not among them, in Swift OR Python.
    swift_source = TIMELINE_COMPOSITION.read_text()
    struct_idx = swift_source.index("struct BrollPlacement")
    struct_end = swift_source.index("\n}", struct_idx)
    struct_body = swift_source[struct_idx:struct_end]
    for forbidden in ("var x:", "var y:", "var width:", "var rotation:", "var opacity:"):
        assert forbidden not in struct_body
    assert "var audioMode:" in struct_body


# ---------------------------------------------------------------------------
# 6. Undo -- real authority reused (no new mechanism invented for B-roll).
# ---------------------------------------------------------------------------

def test_undo_uses_real_captured_snapshot_authority():
    source = BROLL_VIEW.read_text()
    assert "model.canUndoTimelineMutation" in source
    assert "model.undoLastTimelineMutation()" in source


def test_audio_mode_change_participates_in_the_real_undo_snapshot():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func setBrollAudioMode(id: String, mode: TimelineAudioMode) async")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert "capturePreviousForUndo: true" in fn_body


# ---------------------------------------------------------------------------
# 7. Empty / importing / ready / error states -- explicit, derived from
#    real signals only.
# ---------------------------------------------------------------------------

def test_stage_is_derived_from_real_signals_only():
    source = BROLL_VIEW.read_text()
    assert "enum BrollStage { case empty, importing, ready, error }" in source
    stage_idx = source.index("private var stage: BrollStage")
    stage_end = source.index("\n    }", stage_idx)
    stage_body = source[stage_idx:stage_end]
    assert "model.errorMessage" in stage_body
    assert "model.isSavingTimeline" in stage_body
    assert "placements.isEmpty" in stage_body
    assert "readyAssets.isEmpty" in stage_body


def test_all_four_stages_render_something_real():
    source = BROLL_VIEW.read_text()
    for case in (".empty", ".importing", ".ready", ".error"):
        assert f"case {case}:" in source


def test_backend_errors_are_surfaced_via_the_real_error_alert():
    source = BROLL_VIEW.read_text()
    assert '.alert("CutSell"' in source
    assert "model.errorMessage" in source


# ---------------------------------------------------------------------------
# 8. Preservation -- Captions, Voice-over, and the rest of the Timeline
#    gate stay intact.
# ---------------------------------------------------------------------------

def test_captions_and_voiceover_entry_points_are_preserved():
    source = TIMELINE_EDITOR.read_text()
    assert "CaptionsView(" in source
    assert "VoiceOverView(" in source
    assert (TIMELINE_EDITOR.parent / "CaptionsView.swift").exists()
    assert (TIMELINE_EDITOR.parent / "VoiceOverView.swift").exists()


def test_main_video_split_delete_undo_still_use_the_pre_existing_system():
    source = TIMELINE_EDITOR.read_text()
    assert "case .mainVideo:\n            await model.split(clipID: selection.itemID, at: playheadTime)" in source
    assert "case .mainVideo:\n            await model.remove(clipID: selection.itemID)" in source


# ---------------------------------------------------------------------------
# 9. No Video00 hardcoding; no invented timestamps/presets.
# ---------------------------------------------------------------------------

def test_no_video00_or_qa_reference_data_hardcoded():
    forbidden = ("VIDEO-2026-07-30", "5E01F214-A364-4F4B", "D40F1D43-7391-44D5")
    for path in (BROLL_VIEW, IMPORT_MANAGER):
        source = path.read_text()
        for term in forbidden:
            assert term not in source
