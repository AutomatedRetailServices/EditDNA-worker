"""Mobile V1 Captions UI gate -- structural contract tests.

Pure text-scanning tests against the Swift sources (the established
convention in this repo -- no Xcode/Swift toolchain is available in this
offline qualification environment; real Xcode Simulator build verification
happens via the existing `cutsell-ios-ci.yml` macOS CI, dispatched manually
for this isolated feature branch).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTIONS_VIEW = ROOT / "mobile/ios/CutSell/CaptionsView.swift"
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
MAIN_APP = ROOT / "cutsell_app/main.py"
CAPTION_SETTINGS = ROOT / "cutsell_worker/caption_settings.py"
DRAFT_EDITS = ROOT / "cutsell_worker/draft_edits.py"
RENDER = ROOT / "cutsell_worker/render.py"


# ---------------------------------------------------------------------------
# 1. Setup -> Creating -> Ready derived ONLY from real signals, never a
#    fabricated ASR/captions-generation job.
# ---------------------------------------------------------------------------

def test_stages_are_derived_from_real_signals_only():
    source = CAPTIONS_VIEW.read_text()
    assert "enum CaptionsStage { case setup, creating, ready }" in source
    stage_idx = source.index("private var stage: CaptionsStage")
    stage_end = source.index("\n    }", stage_idx)
    stage_body = source[stage_idx:stage_end]
    assert "model.isSaving" in stage_body
    assert "model.captionsEnabled" in stage_body
    # No invented job/polling/timer construct.
    forbidden = ("Timer(", "DispatchQueue", "pollCaption", "caption_job", "CaptionJob")
    for term in forbidden:
        assert term not in source


def test_creating_stage_is_documented_as_save_in_flight_not_asr_job():
    source = CAPTIONS_VIEW.read_text()
    lowered = source.lower()
    assert "asr" in lowered
    assert "not an asr" in lowered or "never an asr" in lowered or "not a fabricated" in lowered or "no separate captions-generation job" in lowered


def test_view_never_auto_enables_captions_on_appear():
    source = CAPTIONS_VIEW.read_text()
    on_appear_idx = source.index(".onAppear {")
    on_appear_end = source.index("\n            }", on_appear_idx)
    on_appear_body = source[on_appear_idx:on_appear_end]
    assert "setCaptionSettings(enabled: true)" not in on_appear_body
    assert "setCaptionSettings(" not in on_appear_body


# ---------------------------------------------------------------------------
# 2. Enable/disable + style -- real, pre-existing authority only.
# ---------------------------------------------------------------------------

def test_enable_toggle_uses_real_authority():
    source = CAPTIONS_VIEW.read_text()
    assert "model.captionsEnabled" in source
    assert "model.setCaptionSettings(enabled: newValue)" in source


def test_style_picker_offers_only_the_two_real_presets():
    source = CAPTIONS_VIEW.read_text()
    assert 'Text("Classic").tag("classic")' in source
    assert 'Text("Clean").tag("clean")' in source
    # No invented third preset.
    forbidden = ("Bold", "Minimal", "Neon", "Karaoke", "Highlight")
    for term in forbidden:
        assert f'.tag("{term.lower()}")' not in source
    settings_source = CAPTION_SETTINGS.read_text()
    assert 'CAPTION_PRESETS = {"classic", "clean"}' in settings_source


def test_style_change_uses_real_authority():
    source = CAPTIONS_VIEW.read_text()
    assert "model.setCaptionSettings(preset: newValue)" in source


# ---------------------------------------------------------------------------
# 3. Position -- honestly unsupported, never simulated.
# ---------------------------------------------------------------------------

def test_position_is_honestly_disabled_never_simulated():
    source = CAPTIONS_VIEW.read_text()
    assert '"Bottom (fixed)"' in source
    assert "Top/Center not yet supported" in source
    # No working position mutation call exists anywhere in the view --
    # only the identifier for the honestly-disabled row, never a real
    # setCaptionSettings(position:)-style call.
    assert "setCaptionSettings(position" not in source
    assert "@State private var position" not in source
    render_source = RENDER.read_text()
    # The real renderer hardcodes bottom alignment -- confirms no position
    # contract exists to wire up.
    assert "Alignment=2" in render_source
    settings_source = CAPTION_SETTINGS.read_text()
    assert "position" not in settings_source.lower()


# ---------------------------------------------------------------------------
# 4. Selected/All -- only shown where real authority actually exists for
#    that scope; the other scope is honestly marked unavailable.
# ---------------------------------------------------------------------------

def test_style_and_enabled_are_honestly_all_scope_only():
    source = CAPTIONS_VIEW.read_text()
    assert "applies to all clips" in source.lower() or "Style — applies to all clips" in source
    assert 'Selected\\" style scope is not available' in source


def test_text_editing_is_honestly_selected_scope_only():
    source = CAPTIONS_VIEW.read_text()
    assert "Text — selected clip only" in source
    assert "no real, coherent" in source.lower() or "no real authority" in source.lower() or "no real, coherent authority" in source.lower()


# ---------------------------------------------------------------------------
# 5. Manual text editing -- real per-clip authority only.
# ---------------------------------------------------------------------------

def test_manual_text_editing_uses_real_per_clip_authority():
    source = CAPTIONS_VIEW.read_text()
    assert "func saveText() async" in source
    save_idx = source.index("func saveText() async")
    save_end = source.index("\n    }", save_idx)
    save_body = source[save_idx:save_end]
    assert "model.editCaption(clipID: clipID, text: draftText)" in save_body
    vm_source = VIEW_MODEL.read_text()
    assert '"/v1/draft-edits/captions"' in vm_source


def test_text_field_is_disabled_until_a_real_clip_is_selected():
    source = CAPTIONS_VIEW.read_text()
    assert ".disabled(selectedClip == nil)" in source


# ---------------------------------------------------------------------------
# 6. Delete + Undo -- real authority only.
# ---------------------------------------------------------------------------

def test_delete_clears_caption_text_via_real_authority():
    source = CAPTIONS_VIEW.read_text()
    assert "func deleteText() async" in source
    delete_idx = source.index("func deleteText() async")
    delete_end = source.index("\n    }", delete_idx)
    delete_body = source[delete_idx:delete_end]
    assert 'model.editCaption(clipID: clipID, text: "")' in delete_body


def test_delete_button_is_gated_on_a_real_non_empty_caption():
    source = CAPTIONS_VIEW.read_text()
    assert '.disabled(selectedClip == nil || (selectedClip?["caption_text"]?.stringValue ?? "").isEmpty)' in source


def test_undo_uses_the_same_real_draft_undo_authority_never_inferred():
    source = CAPTIONS_VIEW.read_text()
    assert "private func performUndo() async" in source
    undo_idx = source.index("private func performUndo() async")
    undo_body = source[undo_idx:]
    assert "let succeeded = await model.undo()" in undo_body
    assert "if succeeded {" in undo_body
    vm_source = VIEW_MODEL.read_text()
    assert "func undo() async -> Bool" in vm_source


def test_no_redo_control_is_offered_for_captions():
    # This gate's own scope is explicitly Delete + Undo only -- no Redo
    # button/label/state, only the doc comment explaining why not.
    source = CAPTIONS_VIEW.read_text()
    assert 'Label("Redo"' not in source
    assert "canRedoCaptions" not in source
    assert "func performRedo" not in source


# ---------------------------------------------------------------------------
# 7. Backend errors are always visible; no silent failure.
# ---------------------------------------------------------------------------

def test_backend_errors_are_surfaced_via_the_real_error_alert():
    source = CAPTIONS_VIEW.read_text()
    assert 'Alert("CutSell"'.replace("Alert", ".alert") in source
    assert "model.errorMessage" in source


# ---------------------------------------------------------------------------
# 8. No invented routes -- only the two real, already-registered endpoints.
# ---------------------------------------------------------------------------

def test_only_real_registered_routes_are_referenced():
    vm_source = VIEW_MODEL.read_text()
    assert '"/v1/draft-edits/caption-settings"' in vm_source
    assert '"/v1/draft-edits/captions"' in vm_source
    main_source = MAIN_APP.read_text()
    assert '@app.post("/v1/draft-edits/captions")' in main_source
    assert '@app.post("/v1/draft-edits/caption-settings")' in main_source


def test_captions_view_calls_no_invented_route_directly():
    source = CAPTIONS_VIEW.read_text()
    # CaptionsView must dispatch exclusively through the view model's real
    # methods -- never construct its own APIClient.request call.
    assert "api.request(" not in source
    assert "APIClient" not in source


# ---------------------------------------------------------------------------
# 9. No Video00 hardcoding; no fabricated timestamps.
# ---------------------------------------------------------------------------

def test_no_video00_or_qa_reference_data_hardcoded():
    forbidden = ("VIDEO-2026-07-30", "5E01F214-A364-4F4B", "D40F1D43-7391-44D5")
    source = CAPTIONS_VIEW.read_text()
    for term in forbidden:
        assert term not in source


def test_no_fabricated_word_level_timestamps():
    # No functional word-timing identifier -- the doc comment's prose
    # mention of "karaoke" (explaining what is NOT supported) is fine.
    source = CAPTIONS_VIEW.read_text()
    forbidden = ("wordTimestamps", "highlightWord", "perWordTiming", "@State private var karaoke")
    for term in forbidden:
        assert term not in source


# ---------------------------------------------------------------------------
# 10. Real, additive entry point from the Timeline editor -- never
#     decorative, never a second competing authority.
# ---------------------------------------------------------------------------

def test_captions_button_is_wired_additively_into_timeline_editor():
    source = TIMELINE_EDITOR.read_text()
    assert "@State private var showCaptions = false" in source
    assert 'Label("Captions", systemImage: "captions.bubble")' in source
    assert "showCaptions = true" in source
    assert ".sheet(isPresented: $showCaptions) {" in source
    assert "CaptionsView(" in source


def test_captions_sheet_passes_main_video_selection_honestly():
    source = TIMELINE_EDITOR.read_text()
    sheet_idx = source.index(".sheet(isPresented: $showCaptions)")
    sheet_end = source.index("\n        }", sheet_idx)
    sheet_body = source[sheet_idx:sheet_end]
    assert "selection?.track == .mainVideo ? selection?.itemID : nil" in sheet_body
