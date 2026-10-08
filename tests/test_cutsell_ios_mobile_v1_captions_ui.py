"""Captions UI -- structural contract tests (Editor v2: panel under the video
in CaptionsPanelView.swift, "Fix words" sheet in CaptionsView.swift).

Pure text-scanning tests against the Swift sources (the established
convention in this repo -- no Xcode/Swift toolchain is available in this
offline qualification environment; real Xcode Simulator build verification
happens via the existing `cutsell-ios-ci.yml` macOS CI, dispatched manually
for this isolated feature branch).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTIONS_VIEW = ROOT / "mobile/ios/CutSell/CaptionsView.swift"
PANEL = ROOT / "mobile/ios/CutSell/CaptionsPanelView.swift"
DRAFT_EDITOR = ROOT / "mobile/ios/CutSell/DraftEditorView.swift"
PLAYBACK = ROOT / "mobile/ios/CutSell/DraftPlaybackView.swift"
TIMELINE_EDITOR = ROOT / "mobile/ios/CutSell/TimelineEditorView.swift"
VIEW_MODEL = ROOT / "mobile/ios/CutSell/DraftEditorViewModel.swift"
MAIN_APP = ROOT / "cutsell_app/main.py"
CAPTION_SETTINGS = ROOT / "cutsell_worker/caption_settings.py"
DRAFT_EDITS = ROOT / "cutsell_worker/draft_edits.py"
RENDER = ROOT / "cutsell_worker/render.py"


# ---------------------------------------------------------------------------
# 1. The "Fix words" sheet edits text only and never invents a job.
# ---------------------------------------------------------------------------

def test_fix_words_sheet_only_edits_text():
    source = CAPTIONS_VIEW.read_text()
    assert "setCaptionSettings(" not in source
    assert '.navigationTitle("Fix words")' in source
    forbidden = ("Timer(", "DispatchQueue", "pollCaption", "caption_job", "CaptionJob")
    for term in forbidden:
        assert term not in source
    lowered = source.lower()
    assert "asr" in lowered and "no separate captions-generation job" in lowered


def test_view_never_changes_settings_on_appear():
    source = CAPTIONS_VIEW.read_text()
    on_appear_idx = source.index(".onAppear {")
    on_appear_end = source.index("\n            }", on_appear_idx)
    assert "setCaptionSettings(" not in source[on_appear_idx:on_appear_end]
    panel = PANEL.read_text()
    on_appear_idx = panel.index(".onAppear {")
    on_appear_end = panel.index("\n        }", on_appear_idx)
    assert "setCaptionSettings(" not in panel[on_appear_idx:on_appear_end]


# ---------------------------------------------------------------------------
# 2. Panel under the video: On/Off, four looks, word colour, box, typeface --
#    the real, pre-existing caption-settings authority only.
# ---------------------------------------------------------------------------

def test_enable_toggle_uses_real_authority():
    source = PANEL.read_text()
    assert "model.captionsEnabled" in source
    assert "model.setCaptionSettings(enabled: newValue)" in source


def test_panel_offers_the_four_editor_v2_looks():
    source = PANEL.read_text()
    assert "private enum Look: String, CaseIterable { case classic, highlight, box, yellow }" in source
    for title in ("Classic", "Highlight", "Box", "Yellow"):
        assert f'return "{title}"' in source


def test_every_preset_the_panel_sends_is_a_real_server_preset():
    from cutsell_worker.caption_render import CAPTION_PRESETS
    import re
    source = PANEL.read_text()
    sent = set(re.findall(r'preset(?:: |\s=\s)"(\w+)"', source))
    sent |= set(re.findall(r'WordColour\(preset: "(\w+)"', source))
    sent |= set(re.findall(r'boxVariant\(preset: "(\w+)"', source))
    assert {"classic", "yellow", "box", "box_light", "highlight_green", "highlight_red", "highlight_blue"} <= sent
    assert sent <= set(CAPTION_PRESETS)


def test_word_colour_and_box_rows_show_only_for_their_look():
    source = PANEL.read_text()
    assert "if enabled && currentLook == .highlight {\n                wordColourRow" in source
    assert "} else if enabled && currentLook == .box {\n                boxVariantRow" in source
    assert '"Black box, white words"' in source and '"White box, black words"' in source
    assert '"Word color"' in source


def test_font_row_lists_the_nine_server_fonts_and_saves_through_real_authority():
    source = PANEL.read_text()
    assert "ForEach(CaptionFontCatalog.all)" in source
    assert "ScrollView(.horizontal" in source
    assert "model.setCaptionSettings(font: spec.key)" in source
    assert "model.setCaptionSettings(preset: preset)" in source


def test_settings_are_sent_to_the_real_route_with_every_field():
    vm = VIEW_MODEL.read_text()
    idx = vm.index("private func sendCaptionSettings(")
    body = vm[idx:vm.index("\n    }\n", idx)]
    for field in ("enabled", "preset", "font", "x", "y", "scale"):
        assert f'object["{field}"]' in body
    assert '"/v1/draft-edits/caption-settings"' in body
    assert "await autosave(edited)" in body
    # Server side accepts exactly these fields.
    main_source = MAIN_APP.read_text()
    request = main_source[main_source.index("class DraftCaptionSettingsRequest"):]
    request = request[:request.index("\n\n")]
    for field in ("enabled", "preset", "font", "x", "y", "scale"):
        assert f"    {field}: " in request


def test_choices_show_at_once_and_fall_back_to_the_server_on_failure():
    vm = VIEW_MODEL.read_text()
    assert "@Published private var captionOverrides" in vm
    assert "captionOverrides[key] ?? snapshot?.draft[key]" in vm
    # Saves run one after another on the latest revision; each choice is
    # dropped once its own save finishes (success or failure).
    assert "await previous?.value" in vm
    assert "where self.captionOverrideTokens[key] == token" in vm
    assert "self.captionOverrides[key] = nil" in vm


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
# 10. Entry points: the Captions button opens the panel under the video;
#     tapping the caption on the video (panel open) opens "Fix words".
# ---------------------------------------------------------------------------

def test_captions_button_opens_the_panel_under_the_video():
    timeline = TIMELINE_EDITOR.read_text()
    assert "var onOpenCaptions: () -> Void = {}" in timeline
    assert 'Label("Captions", systemImage: "captions.bubble")' in timeline
    assert "onOpenCaptions()" in timeline
    assert "CaptionsView(" not in timeline
    editor = DRAFT_EDITOR.read_text()
    assert "@State private var showCaptionsPanel = false" in editor
    assert "if showCaptionsPanel {\n                            CaptionsPanelView(model: model)" in editor
    assert editor.index("DraftPlaybackView(") < editor.index("CaptionsPanelView(") < editor.index("TimelineEditorView(")


def test_tapping_the_caption_opens_fix_words_for_that_clip():
    editor = DRAFT_EDITOR.read_text()
    assert "onCaptionTap: showCaptionsPanel ?" in editor
    assert "CaptionsView(model: model, initialClipID: captionTextClipID)" in editor
    playback = PLAYBACK.read_text()
    assert "onCaptionTap(cue.clipID)" in playback
