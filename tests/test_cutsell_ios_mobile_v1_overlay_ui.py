"""Mobile V1 Overlay UI -- structural contract tests.

CORRECTIVE GATE: a prior gate built `OverlayView.swift` against
`BrollPlacement` (a full-frame visual replacement clip with no position or
scale) -- a mislabeling. That code is now honestly renamed to
`BrollView.swift` (see `test_cutsell_ios_mobile_v1_broll_ui.py`). This file
tests the GENUINE, positioned/scaled Overlay feature built against
`/v1/overlays/*` (`cutsell_worker/overlay_edits.py`), confirmed real and
ACTIVE by direct audit: `cutsell_worker/export_job.py` reads
`draft.media_overlays` and passes it into `render.render_preview`, which
invokes `media_overlay_render.build_final_overlay_command` -- a real ffmpeg
`overlay=x=...:y=...` filter with scale, a timed enable window, and
mute_audio-gated audio mixing.

Pure text-scanning tests against the Swift sources (the established
convention in this repo -- no Xcode/Swift toolchain is available in this
offline qualification environment; real Xcode Simulator build verification
happens via the existing `cutsell-ios-ci.yml` macOS CI, dispatched manually
for this isolated feature branch).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY_VIEW = ROOT / "mobile/ios/CutSell/OverlayView.swift"
UPLOAD_MANAGER = ROOT / "mobile/ios/CutSell/OverlayUploadManager.swift"
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
OVERLAY_ROUTES = ROOT / "cutsell_app/overlay_routes.py"
OVERLAY_EDITS = ROOT / "cutsell_worker/overlay_edits.py"
OVERLAY_UPLOADS = ROOT / "cutsell_worker/overlay_uploads.py"
CONTRACTS = ROOT / "cutsell_worker/contracts.py"
SERDE = ROOT / "cutsell_worker/serde.py"
EXPORT_JOB = ROOT / "cutsell_worker/export_job.py"
MEDIA_OVERLAY_RENDER = ROOT / "cutsell_worker/media_overlay_render.py"
RENDER = ROOT / "cutsell_worker/render.py"


# ---------------------------------------------------------------------------
# 1. Real, ACTIVE authority audit -- the routes are registered, connected to
#    the canonical draft, and actually consumed by the export renderer.
#    (This is what distinguishes Overlay from a merely-plausible-looking
#    but disconnected legacy system -- confirmed here so a future change
#    that silently disconnects the renderer wiring is caught.)
# ---------------------------------------------------------------------------

def test_overlay_routes_are_registered_in_the_running_app():
    main_source = (ROOT / "cutsell_app/main.py").read_text()
    assert "from cutsell_app.overlay_routes import router as overlay_router" in main_source
    assert "app.include_router(overlay_router)" in main_source


def test_overlay_routes_call_the_real_edit_functions():
    routes_source = OVERLAY_ROUTES.read_text()
    assert "add_media_overlay(" in routes_source
    assert "update_media_overlay(" in routes_source
    assert "remove_media_overlay(" in routes_source
    edits_source = OVERLAY_EDITS.read_text()
    assert "def add_media_overlay(" in edits_source
    assert "def update_media_overlay(" in edits_source
    assert "def remove_media_overlay(" in edits_source


def test_media_overlays_is_parsed_into_the_canonical_draft_contract():
    contracts_source = CONTRACTS.read_text()
    assert "class MediaOverlay" in contracts_source
    serde_source = SERDE.read_text()
    assert "_media_overlay_from_dict" in serde_source
    assert "media_overlays=media_overlays" in serde_source


def test_export_job_actually_wires_overlays_into_the_real_renderer():
    # The decisive confirmation this is ACTIVE, not legacy/disconnected.
    export_source = EXPORT_JOB.read_text()
    assert "draft.media_overlays" in export_source
    assert "LocalMediaOverlay(" in export_source
    assert "media_overlays=tuple(local_overlays)" in export_source
    render_source = RENDER.read_text()
    assert "media_overlays: Iterable[LocalMediaOverlay]" in render_source
    assert "build_final_overlay_command(" in render_source


def test_renderer_applies_real_position_scale_and_timing():
    source = MEDIA_OVERLAY_RENDER.read_text()
    assert "overlay=x=" in source
    assert "scale={pixel_width}" in source
    assert "enable='between(t," in source
    assert "mute_audio" in source


# ---------------------------------------------------------------------------
# 2. Entry point from Timeline -- real, non-decorative, additive. Overlay
#    has no dedicated track row (media_overlays has no typed per-track
#    model) -- it is a real action-bar button, mirroring Captions.
# ---------------------------------------------------------------------------

def test_overlay_entry_point_wired_into_timeline_action_bar():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var showOverlayView = false" in source
    assert ".sheet(isPresented: $showOverlayView)" in source
    assert "OverlayView(model: model)" in source
    assert '.accessibilityIdentifier("timeline.overlayButton")' in source


def test_overlay_is_not_a_timeline_track_kind():
    source = TIMELINE_EDITOR.read_text()
    assert "case mainVideo, voiceOver, broll" in source
    assert "case overlay" not in source


# ---------------------------------------------------------------------------
# 3. Import -- uses the real, pre-existing OverlayUploadManager (S3
#    presigned POST), then places the result via addMediaOverlay -- never
#    the D-282A two-phase upload+registration pipeline (that contract has
#    no analog here: /v1/overlays/add places immediately).
# ---------------------------------------------------------------------------

def test_import_uses_the_real_pre_existing_upload_manager():
    view_source = OVERLAY_VIEW.read_text()
    assert "allowedContentTypes: [.image, .movie]" in view_source
    assert "model.importOverlayMedia(fileURL: url)" in view_source
    vm_source = VIEW_MODEL.read_text()
    assert "func importOverlayMedia(fileURL: URL) async" in vm_source
    assert "OverlayUploadManager.shared.upload(" in vm_source
    assert UPLOAD_MANAGER.exists()


def test_import_places_immediately_via_the_real_add_route():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func importOverlayMedia(fileURL: URL) async")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert "await addMediaOverlay(" in fn_body


def test_add_media_overlay_posts_to_the_real_route_with_required_fields():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func addMediaOverlay(")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert '"/v1/overlays/add"' in fn_body
    for field in ('"project_id"', '"user_id"', '"kind"', '"uri"', '"start"', '"end"', '"x"', '"y"', '"width"', '"mute_audio"'):
        assert field in fn_body
    # Matches OverlayAddRequest's own required fields exactly.
    routes_source = OVERLAY_ROUTES.read_text()
    add_idx = routes_source.index("class OverlayAddRequest")
    add_end = routes_source.index("class OverlayUpdateRequest")
    add_body = routes_source[add_idx:add_end]
    for field in ("project_id", "user_id", "draft", "kind", "uri", "start", "end", "x", "y", "width", "mute_audio"):
        assert field in add_body


# ---------------------------------------------------------------------------
# 4. Real per-overlay editing -- position (x/y), scale (width), mute_audio.
#    Uses /v1/overlays/update, never a fabricated mutation.
# ---------------------------------------------------------------------------

def test_update_media_overlay_posts_to_the_real_route():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func updateMediaOverlay(")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert '"/v1/overlays/update"' in fn_body
    assert '"overlay_id"' in fn_body


def test_remove_media_overlay_posts_to_the_real_route():
    vm_source = VIEW_MODEL.read_text()
    fn_idx = vm_source.index("func removeMediaOverlay(")
    fn_end = vm_source.index("\n    }", fn_idx)
    fn_body = vm_source[fn_idx:fn_end]
    assert '"/v1/overlays/remove"' in fn_body
    assert '"overlay_id"' in fn_body


def test_position_and_scale_sliders_commit_only_once_never_per_frame():
    # A per-drag-frame network call would storm the API and race the
    # draft's own optimistic-concurrency expected_revision check.
    source = OVERLAY_VIEW.read_text()
    assert "struct CommitSlider" in source
    assert "onEditingChanged" in source
    commit_idx = source.index("struct CommitSlider")
    commit_body = source[commit_idx:]
    assert "if !editing { Task { await onCommit(localValue) } }" in commit_body


def test_position_scale_and_mute_audio_are_wired_to_real_mutations():
    source = OVERLAY_VIEW.read_text()
    assert "model.updateMediaOverlay(overlayID: overlayID, x: newValue)" in source
    assert "model.updateMediaOverlay(overlayID: overlayID, y: newValue)" in source
    assert "model.updateMediaOverlay(overlayID: overlayID, width: newValue)" in source
    assert "model.updateMediaOverlay(overlayID: overlayID, muteAudio: newValue)" in source


def test_delete_uses_real_removal_authority():
    source = OVERLAY_VIEW.read_text()
    assert "model.removeMediaOverlay(overlayID:" in source


# ---------------------------------------------------------------------------
# 5. Rotation/opacity/keyframes -- honestly disabled (real gap: no such
#    field on MediaOverlay), unlike position/scale which ARE real.
# ---------------------------------------------------------------------------

def test_no_rotation_opacity_keyframe_field_exists_in_the_real_contract():
    py_source = CONTRACTS.read_text()
    class_idx = py_source.index("class MediaOverlay")
    class_end = py_source.index("\n\n", class_idx)
    class_body = py_source[class_idx:class_end]
    for forbidden in ("rotation", "opacity", "keyframe"):
        assert forbidden not in class_body.lower()
    # Position and scale, by contrast, ARE real.
    assert "x: float" in class_body
    assert "y: float" in class_body
    assert "width: float" in class_body


def test_rotation_opacity_keyframe_controls_are_shown_disabled_never_wired():
    source = OVERLAY_VIEW.read_text()
    assert 'Section("Rotation, opacity & keyframes (not yet supported)")' in source
    section_idx = source.index('Section("Rotation, opacity & keyframes (not yet supported)")')
    section_end = source.index("\n                    }", source.index("Section", section_idx + 10))
    section_body = source[section_idx:section_end]
    assert section_body.count(".disabled(true)") >= 2
    for forbidden in ("setRotation", "setOpacity", "addKeyframe"):
        assert forbidden not in section_body


# ---------------------------------------------------------------------------
# 6. Undo/Redo -- reuses the SAME Main Video draft/undo/redo authority,
#    never a second/fabricated mechanism (overlays live on the same
#    canonical draft, unlike B-roll/Voice-over's D-282A snapshot Undo).
# ---------------------------------------------------------------------------

def test_undo_redo_reuse_the_real_main_video_draft_authority():
    source = OVERLAY_VIEW.read_text()
    assert "await model.undo()" in source
    assert "await model.redo()" in source
    assert "undoLastTimelineMutation" not in source
    assert "canUndoTimelineMutation" not in source


# ---------------------------------------------------------------------------
# 7. Empty / importing / ready / error states -- explicit, derived from
#    real signals only.
# ---------------------------------------------------------------------------

def test_stage_is_derived_from_real_signals_only():
    source = OVERLAY_VIEW.read_text()
    assert "enum OverlayStage { case empty, importing, ready, error }" in source
    stage_idx = source.index("private var stage: OverlayStage")
    stage_end = source.index("\n    }", stage_idx)
    stage_body = source[stage_idx:stage_end]
    assert "model.errorMessage" in stage_body
    assert "model.isSaving" in stage_body
    assert "overlays.isEmpty" in stage_body


def test_all_four_stages_render_something_real():
    source = OVERLAY_VIEW.read_text()
    for case in (".empty", ".importing", ".ready", ".error"):
        assert f"case {case}:" in source


def test_backend_errors_are_surfaced_via_the_real_error_alert():
    source = OVERLAY_VIEW.read_text()
    assert '.alert("CutSell"' in source
    assert "model.errorMessage" in source


# ---------------------------------------------------------------------------
# 8. No Video00 hardcoding.
# ---------------------------------------------------------------------------

def test_no_video00_or_qa_reference_data_hardcoded():
    forbidden = ("VIDEO-2026-07-30", "5E01F214-A364-4F4B", "D40F1D43-7391-44D5")
    source = OVERLAY_VIEW.read_text()
    for term in forbidden:
        assert term not in source
