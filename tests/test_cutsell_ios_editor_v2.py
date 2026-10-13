"""Editor v2 (Figma 04_Editor 242:1532 / 950:1165 and its states) replaces the old list editor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_editor_screen_is_v2():
    editor = read("DraftEditorView.swift")
    body = editor.split("var body: some View {", 1)[1].split("@ViewBuilder", 1)[0]
    assert "EditorV2View(model: model, project: project)" in body
    assert "editor(model)" not in body  # old sections no longer shown


def test_layout_pieces_from_figma():
    v2 = read("EditorV2View.swift")
    for text in ('"Export"', '"Mute clip"', '"Voiceover"', '"Overlay"', '"Captions"', '"Split"', '"Delete"',
                 '"Voice\\nOver"', '"Video clip selected"', '"Voice-over selected"', '"Overlay selected"',
                 '"Video split at \\(at)"', '"Caption deleted"'):
        assert text in v2, text
    for image in ("EdExpand", "EdKeyframe", "EdUndo", "EdRedo", "EdToolDelete", "EdToolOverlay", "EdToolVoice"):
        assert f'"{image}"' in v2, image
    assert (APP / "Assets.xcassets" / "Figma" / "EdUndo.imageset").exists()


def test_actions_use_the_existing_model_routes():
    v2 = read("EditorV2View.swift")
    for call in ("model.split(clipID:", "model.safeSplitTime(clipID:", "model.remove(clipID:", "model.removeVoiceOverPlacement(id:",
                 "model.removeTextOverlay(id:", "model.removeMediaOverlay(id:", "model.editCaption(clipID:", "model.undo()",
                 "model.redo()", "model.setAudio(clipID:"):
        assert call in v2, call
    sheets = read("EditorV2Sheets.swift")
    assert "model.addRecordedVoiceOver(fileURL: url, at: start)" in sheets
    assert "model.setCaptionSettings(enabled: auto)" in sheets
    assert "await model.export()" in sheets


def test_readable_and_renamable_title():
    v2 = read("EditorV2View.swift")
    assert 'lower.hasPrefix("media-")' in v2 and 'return "Cut \\(formatter.string(from: date))"' in v2
    assert '"/v1/projects/\\(project.projectID)", method: "PATCH"' in v2
    assert "CutNaming.dated()" in read("CreateFlowView.swift")
    assert "CutNaming.readable(project: project)" in read("ProjectsView.swift")


def test_recorder_can_pause():
    recorder = read("VoiceRecorder.swift")
    assert "func pause()" in recorder and "func resume()" in recorder


def test_broll_shows_on_the_video_track_and_can_be_deleted_or_split():
    v2 = read("EditorV2View.swift")
    assert "model.timelineComposition?.brollPlacements" in v2
    assert 'accessibilityIdentifier("brollBlock")' in v2
    assert "case broll(String)" in v2 and 'return "B-roll selected"' in v2
    assert "await model.removeBrollPlacement(id: id)" in v2
    assert "await model.splitBrollPlacement(id: id, at: time)" in v2
    assert '"editor_sel_broll"' in read("DesignScreenshots.swift")


def test_editor_probe_places_voice_over_and_broll_with_the_apps_calls():
    probe = (ROOT / ".github/workflows/cutsell-ios-editor-probe.yml").read_text()
    for needle in ('"PRIMARY_SOURCE"', "/timeline-uploads", "/timeline-assets", '"PUT", f"/v1/projects/{pid}/timeline"',
                   '"VOICE_OVER", "AUDIO"', '"SUPPLEMENTAL_BROLL", "VIDEO"'):
        assert needle in probe, needle
