"""Editor v4 (Figma 04_Editor "Editor v4" 1-11, approved 13 Oct 2026) replaces Editor v2, plus the
build-11 fixes: preview sound, separate video/voice volumes, live/real voice-over waves, speed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_editor_screen_is_v4():
    body = read("DraftEditorView.swift").split("var body: some View {", 1)[1].split("@ViewBuilder", 1)[0]
    assert "EditorV4View(model: model, project: project, previewState: Self.probeState)" in body
    assert "#else\n        return nil" in read("DraftEditorView.swift")  # Release builds never open a probe state


def test_header_has_no_logo_and_controls_have_no_keyframe():
    v4 = read("EditorV4View.swift")
    header = v4.split("private var header: some View {", 1)[1].split("private var shownTitle", 1)[0]
    assert "BrandLogo" not in header
    assert '"Export"' in header and "renaming = true" in header
    controls = v4.split("private var controlsRow: some View {", 1)[1].split("@ViewBuilder", 1)[0]
    assert "EdKeyframe" not in controls
    for image in ("EdExpand", "EdUndo", "EdRedo"):
        assert f'"{image}"' in controls
    assert "CGSize(width: 266, height: 472)" in v4  # bigger video (Figma 266 x 472)


def test_toolbar_v4_order_and_icons():
    v4 = read("EditorV4View.swift")
    assert "case split, delete, mirror, rotate, overlay, voiceOver, captions, style" in v4
    for title in ('"Split"', '"Delete"', '"Mirror"', '"Rotate"', '"Overlay"', '"Voice Over"', '"Captions"', '"Style"'):
        assert title in v4, title
    for icon in ("Split", "Delete", "Mirror", "Rotate", "Overlay", "Voice", "Captions", "Style"):
        assert (APP / "Assets.xcassets" / "Figma" / f"Ed4{icon}.imageset" / f"Ed4{icon}@3x.png").exists(), icon
    assert "ScrollView(.horizontal" in v4  # slides sideways
    assert "active == tool ? CS.cyan" in v4  # active tool in cyan


def test_timeline_cursor_fixed_in_the_middle_with_zoom():
    tl = read("EditorV4Timeline.swift")
    assert "center + CGFloat(seconds - now) * pixelsPerSecond" in tl  # tracks slide under a fixed cursor
    assert "MagnificationGesture()" in tl and "pixelsPerSecond = max(Self.minZoom" in tl
    assert "playback.seek(to:" in tl
    # caption inside each clip, no separate captions track
    assert "ClipBlockV4(" in tl and "caption: caption" in tl
    assert "captionLane" not in tl
    # voice-over track always visible
    assert '"+  Voice over"' in tl and "VoiceBlockV4(" in tl
    assert '"+  Text · photo · B-roll"' in tl and '"brollBlock"' in tl


def test_caption_written_on_the_video_with_menu_2b():
    v4 = read("EditorV4View.swift")
    assert "startCaptionEditing(cue.clipID)" in v4 and "TextField(\"\", text: $captionText" in v4
    assert "model.editCaption(clipID: clipID, text: text)" in v4
    panels = read("EditorV4Panels.swift")
    for preset in ('"classic"', '"highlight_green"', '"highlight_red"', '"highlight_blue"', '"box"', '"box_light"', '"yellow"'):
        assert preset in panels, preset
    assert "CaptionFontCatalog.all" in panels and '"Word color"' in panels
    assert "model.applyCaptionSettings(enabled: $0)" in panels


def test_voice_over_waves_live_and_real():
    recorder = read("VoiceRecorder.swift")
    assert "recorder.isMeteringEnabled = true" in recorder and "averagePower(forChannel: 0)" in recorder
    assert "@Published private(set) var levels" in recorder
    panels = read("EditorV4Panels.swift")
    assert "LiveWave(levels: recorder.levels)" in panels
    assert "Waveform.samples(url: url" in panels and "PlayedWave(samples: samples, progress: progress)" in panels
    assert "VoiceEnhancer.enhance(url)" in panels
    v4 = read("EditorV4View.swift")
    assert '"Allow microphone access"' in v4 and "VoiceRecorder.requestPermission()" in v4


def test_preview_has_sound_and_separate_volumes():
    tools = read("AudioTools.swift")
    assert "setCategory(.playback, mode: .moviePlayback" in tools
    playback = read("DraftPlaybackView.swift")
    assert "EditorAudio.preparePlayback()" in playback
    assert "VoiceOverStore.url(for: placement.assetID)" in playback
    assert "model.voiceOverVolume" in playback
    vm = read("DraftEditorViewModel.swift")
    assert "func setVideoTrackVolume(_ volume: Double) async" in vm
    assert "VoiceOverStore.save(fileURL, assetID: asset.assetID)" in vm


def test_mirror_and_rotate_preview_and_saved_on_the_draft():
    v4 = read("EditorV4View.swift")
    assert '.scaleEffect(x: model.isMirrored ? -1 : 1, y: 1)' in v4
    assert 'model.setDraftValue("mirror", .bool(mirrored))' in v4
    assert 'model.setDraftValue("rotation", .number(Double(next)))' in v4


def test_edits_show_at_once_and_run_in_one_queue():
    vm = read("DraftEditorViewModel.swift")
    assert "var displayDraft: JSONValue? { optimisticDraft ?? snapshot?.draft }" in vm
    assert "func perform(predict:" in vm
    assert "captionSettingsChain" not in vm  # one queue for every edit
    for fn in ("func remove(clipID: String) async {\n        await perform(",
               "func split(clipID: String, at sourceTime: Double) async {\n        await perform(",
               "func setAudio(clipID: String, muted: Bool? = nil, volume: Double? = nil) async {\n        await perform("):
        assert fn in vm, fn


def test_preview_and_upload_speedups():
    playback = read("DraftPlaybackView.swift")
    assert "withTaskGroup(of: (String, LoadedSource?).self)" in playback and "sourceCache[id]" in playback
    upload = read("MultipartUploadManager.swift")
    assert "let maxInFlight = 3" in upload and "withThrowingTaskGroup" in upload
    frames = read("ClipFrames.swift")
    assert "generators[url.path]" in frames


def test_design_screenshots_cover_every_v4_screen():
    shots = read("DesignScreenshots.swift")
    for name in ("editor_v4_base", "editor_v4_caption_edit", "editor_v4_vo_record", "editor_v4_vo_review",
                 "editor_v4_cap_highlight", "editor_v4_cap_box", "editor_v4_cap_fonts", "editor_v4_overlay_gallery",
                 "editor_v4_overlay_selected", "editor_v4_mic", "editor_v4_style"):
        assert f'"{name}"' in shots, name


def test_readable_and_renamable_title():
    v4 = read("EditorV4View.swift")
    assert '"/v1/projects/\\(project.projectID)", method: "PATCH"' in v4
    assert "CutNaming.readable(project: project)" in v4


def test_recorder_can_pause():
    recorder = read("VoiceRecorder.swift")
    assert "func pause()" in recorder and "func resume()" in recorder
