"""Build 17 (iPhone): sounds editable like the voice and never silent, Volume / Mute per clip or for
the whole video, Clean up voice with the server's clean copy, Auto Zoom AI and manual keyframes
(CapCut-like). Static checks of the iPhone sources; the server side and real renders are covered
by tests/test_cutsell_build17_end_to_end.py (server repo) and the Simulator probe with real touches
(.github/workflows/cutsell-ios-build17-probe.yml)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def test_sounds_are_never_added_silent_and_the_preview_follows_their_download():
    model = read("EditorV5Model.swift")
    assert "guard SfxLibrary.localURL(sfxID) != nil else { return nil }" in model
    assert "func ensure() async" in model and "for delay in [0.0, 3, 10, 30]" in model
    panels = read("EditorV5Panels.swift")
    assert '"Getting sound…"' in panels and ".disabled(!ready)" in panels and "soundDownloading" in panels
    v4 = read("EditorV4View.swift")
    assert "@ObservedObject private var sounds = SfxLibrary.shared" in v4 and "\\(sounds.ready.count)" in v4
    assert "selection = .sound(clipID)" in v4          # a new sound comes selected
    vm = read("DraftEditorViewModel.swift")
    assert "Task { await SfxLibrary.shared.ensure() }" in vm


def test_sound_blocks_move_split_trim_and_are_easy_to_touch():
    timeline = read("EditorV4Timeline.swift")
    assert ".highPriorityGesture(directMoveGesture(kind: .sound, id: clip.id" in timeline
    assert "audioTrimHandle(.end, clip: clip)" in timeline
    assert ".frame(width: max(width, 32), alignment: .leading)" in timeline
    v4 = read("EditorV4View.swift")
    assert "case .voiceOver(let id), .sound(let id):\n            if model.splitAudioClip(id: id, at: time)" in v4


def test_volume_panel_selected_clip_or_whole_video_with_mute():
    panels = read("EditorV5Panels.swift")
    assert 'subtitle: target == .whole ? "Whole video" : "Selected clip"' in panels
    assert "model.setMuted(target, !muted)" in panels
    b17 = read("EditorB17Model.swift")
    for case in ("case .video(let id): return .video(id)", "case .voiceOver(let id), .sound(let id): return .audio(id)"):
        assert case in b17
    # Build 18: "Whole video" is the video's own sound (sounds and voices keep playing).
    assert "case .whole: return videoVolume <= 0.001" in b17 and "setVideoVolume(0)" in b17
    assert 'object["muted"] = .bool(true)' in read("EditorV5Model.swift")
    timeline = read("EditorV4Timeline.swift")
    assert "struct MuteBadge" in timeline and timeline.count("MuteBadge()") >= 3
    v4 = read("EditorV4View.swift")
    assert "onToggleVideoSound: { toggleVideoSound() }" in v4 and "model.toggleVideoSound()" in v4
    assert "if card == .volume { return }" in v4      # the panel follows the selection


def test_preview_mixes_mutes_master_and_gains_like_the_export():
    playback = read("DraftPlaybackView.swift")
    assert "let master = model.masterMuted ? 0 : model.masterVolume" in playback
    assert "AudioGainTap.make(gain: tapGain)" in playback
    vm = read("DraftEditorViewModel.swift")
    assert "volume: clip.muted ? 0 : clip.volume" in vm
    b17 = read("EditorB17Model.swift")
    assert "MTAudioProcessingTapCreate" in b17


def test_clean_up_uses_the_servers_clean_copy_with_progress():
    panels = read("EditorV5Panels.swift")
    assert 'Text("Removes background noise from the voice")' in panels
    assert 'Text("Cleaning… \\(progress)%")' in panels
    b17 = read("EditorB17Model.swift")
    assert '"/v1/projects/\\(project.projectID)/clean-audio"' in b17
    assert "CleanAudioCenter.shared.request(key, project: project)" in b17
    playback = read("DraftPlaybackView.swift")
    assert "CleanAudioCenter.shared.localURL(.source(sourceID), project: model.project)" in playback
    vm = read("DraftEditorViewModel.swift")
    assert "CleanAudioCenter.shared.localURL(.asset(asset), project: project)" in vm


def test_keyframes_like_capcut():
    b17 = read("EditorB17Model.swift")
    # The server's interpolation (smoothstep, held outside) and AI push-ins as four keyframes.
    assert "v * v * (3 - 2 * v)" in b17
    assert "let steps: [(Double, Double)] = [(t, 1), (t + r, s), (t + d - r, s), (t + d, 1)]" in b17
    assert "func adoptLegacyZoomKeyframes()" in b17 and "adoptLegacyZoomKeyframes()" in read("DraftEditorViewModel.swift")
    assert "return transformKeyframes.filter { !$0.isAI || auto }" in b17
    v4 = read("EditorV4View.swift")
    # Build 18: UIKit pinch / rotation / pan (FramingGestureLayer) instead of SwiftUI gestures.
    assert "FramingGestureLayer(" in v4 and "UIRotationGestureRecognizer" in read("FramingGestureLayer.swift")
    assert "ZoomDiamond(isSelected: true).scaleEffect(1.5)" in v4          # filled ◇ on a keyframe
    assert "model.deleteKeyframe(id: mark.id)" in v4
    # Preview order = export order: zoom, turn clockwise, then move.
    assert ".scaleEffect(key.scale)\n                .rotationEffect(.degrees(key.rotation))\n                .offset(x: box.size.width * key.x" in v4
    timeline = read("EditorV4Timeline.swift")
    assert "model.moveKeyframe(id: mark.id" in timeline and "model.overlayKeyframeMarks(overlayID: item.id)" in timeline


def test_ui_tests_exist_for_the_probe():
    tests = (ROOT / "mobile" / "ios" / "CutSellUITests" / "Build17UITests.swift").read_text()
    for step in ('tool("split")', 'tool("delete")', "press(forDuration: 0.6, thenDragTo:", '"audioTrimEnd"',
                 'app.switches["cleanUpToggle"]', 'app.switches["editorAutoZoomToggle"]', "pinch(withScale:",
                 'tap("editorExport")'):
        assert step in tests, step
