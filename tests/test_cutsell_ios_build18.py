"""Build 18 (errors on Swanny's real iPhone with build 17): sounds silent in the preview, per-clip
mute and the video speaker, keyframes with the fingers, overlays on the timeline, the gallery and a
hidden Diagnostics screen. Static checks of the iPhone sources; the Simulator probe with real
touches, built OPTIMIZED like TestFlight, is .github/workflows/cutsell-ios-build18-probe.yml."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def test_sound_files_stay_alive_in_the_preview_mix():
    """Root cause: the sound's AVURLAsset was a temporary; an optimized build frees it before the
    insert, try? hid the failure -> a silent track. The assets are now kept by the controller."""
    playback = read("DraftPlaybackView.swift")
    assert "AVURLAsset(url: placement.url).loadTracks" not in playback
    assert "private var audioAssetCache: [String: AVURLAsset]" in playback
    assert "let asset = audioAsset(for: url, used: &used)" in playback
    # Cut to what the file really holds (an AAC file can be a few ms shorter).
    assert "let to = min(sourceOut, fileEnd)" in playback
    # A failed insert is never hidden.
    assert 'CutSellDiagnostics.log("preview_audio_insert_failed"' in playback
    assert "track.segments.filter { !$0.isEmpty }" in playback


def test_newer_rebuild_wins():
    playback = read("DraftPlaybackView.swift")
    assert "func superseded() -> Bool { generation != buildGeneration }" in playback
    assert playback.count("if superseded() { return }") >= 4


def test_audio_session_playback_movie_every_play():
    tools = read("AudioTools.swift")
    assert "session.category == .playback, session.mode == .moviePlayback" in tools
    assert "try session.setCategory(.playback, mode: .moviePlayback, options: [])" in tools
    assert "EditorAudio.preparePlayback()" in read("DraftPlaybackView.swift")


def test_sound_downloads_are_checked_and_retried():
    model = read("EditorV5Model.swift")
    assert "nonisolated static func playableSeconds(_ url: URL) async -> Double?" in model
    assert 'CutSellDiagnostics.log("sfx_file_unplayable"' in model
    assert "func retry()" in model and "let length = SfxLibrary.shared.length(of: sfxID)" in model
    v4 = read("EditorV4View.swift")
    assert '"A sound didn\'t download"' in v4 and "sounds.retry()" in v4 and '"Getting sounds…"' in v4


def test_whole_video_mute_is_the_video_sound_not_the_sounds():
    b17 = read("EditorB17Model.swift")
    assert "case .whole: return videoVolume <= 0.001" in b17
    assert "func adoptLegacyMasterVolume()" in b17 and "adoptLegacyMasterVolume()" in read("DraftEditorViewModel.swift")
    # The speaker brings back every clip of the video.
    assert 'if clip["audio_muted"]?.boolValue == true { clip["audio_muted"] = .bool(false) }' in b17
    assert "func toggleVideoSound() -> Bool" in b17
    assert ".accessibilityValue(muted ? \"off\" : \"on\")" in read("EditorV4Timeline.swift")


def test_keyframes_use_uikit_pinch_rotation_pan():
    layer = read("FramingGestureLayer.swift")
    for name in ("UIPinchGestureRecognizer", "UIRotationGestureRecognizer", "UIPanGestureRecognizer",
                 "shouldRecognizeSimultaneouslyWith", "maximumNumberOfTouches = 2"):
        assert name in layer, name
    v4 = read("EditorV4View.swift")
    assert "if framingEnabled, !framingOverlay { framingLayer(size: size) }" in v4
    assert "if framingEnabled, framingOverlay { framingLayer(size: size) }" in v4
    assert "videoFramingGesture" not in v4
    assert "case nil, .video, .zoom, .overlay(_, isText: false): return true" in v4


def test_selected_items_move_with_a_plain_drag():
    timeline = read("EditorV4Timeline.swift")
    assert "DragGesture(minimumDistance: 6, coordinateSpace: .global)" in timeline
    for kind in ("voice", "sound", "overlay"):
        assert f".gesture(itemMove(kind: .{kind}," in timeline and "selected: isSelected))" in timeline
    v4 = read("EditorV4View.swift")
    assert '"Wait until it finishes uploading, then split it."' in v4


def test_gallery_uses_the_system_picker_and_says_when_access_is_limited():
    panels = read("EditorV4Panels.swift")
    assert "PHPickerViewController" in panels and "configuration.preferredAssetRepresentationMode = .current" in panels
    assert "fetchLimit = 300" not in panels
    assert '"Getting it from iCloud… \\(Int((value * 100).rounded()))%"' in panels
    gallery = read("GalleryPickerView.swift")
    assert '"Select more photos"' in gallery and '"Change permission"' in gallery
    assert "if preview == nil, library.status == .limited { limitedBanner }" in gallery


def test_diagnostics_screen_report_and_privacy():
    center = read("DiagnosticsCenter.swift")
    assert "static let limit = 50" in center
    for blocked in ('"user"', '"token"', '"email"', '"url"', '"path"', '"project"', '"key"'):
        assert blocked in center, blocked
    assert "ShareLink(item: reportURL)" in center and '"Send report"' in center
    assert "AVAssetReaderAudioMixOutput" in center
    assert "DiagnosticsEventLog.shared.add(event, fields)" in read("CutSellDiagnostics.swift")
    you = read("YouView.swift")
    assert "versionTaps.count >= 5" in you and "DiagnosticsView()" in you


def test_probe_is_built_like_testflight():
    workflow = (ROOT / ".github" / "workflows" / "cutsell-ios-build18-probe.yml").read_text()
    assert "SWIFT_OPTIMIZATION_LEVEL=-O" in workflow and "-only-testing:CutSellUITests/Build18UITests" in workflow
    assert "*.mp4" in workflow and "-delete" in workflow   # never the recording in the repo
    tests = (ROOT / "mobile" / "ios" / "CutSellUITests" / "Build18UITests.swift").read_text()
    for step in ('"CUTSELL_PROBE_TEMP_ASSET"', "typing in the mix at 2 s with sound", "speaker tap", ".rotate(",
                 "overlay moves with a plain drag", "Diagnostics opens with five taps"):
        assert step in tests, step
