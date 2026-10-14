"""iPhone Voice Over recording (phase 1): microphone permission, AAC .m4a take, Stop / Record
again / Use, and "Use" goes through the same import path and is placed at the cursor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_recorder_writes_aac_m4a_after_asking_for_the_microphone():
    rec = read("VoiceRecorder.swift")
    assert "AVAudioApplication.requestRecordPermission" in rec
    assert "kAudioFormatMPEG4AAC" in rec and '.m4a"' in rec
    assert "AVAudioRecorder(url: url, settings: settings)" in rec


def test_server_accepts_m4a_voice_over_uploads():
    uploads = (ROOT / "cutsell_worker" / "uploads.py").read_text()
    assert 'ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS = {".m4a"}' in uploads
    assert '"audio/mp4"' in uploads
    # The app labels .m4a as audio/mp4, one of the accepted types.
    assert 'case "m4a": return "audio/mp4"' in read("VoiceOverImportManager.swift")


def test_screen_has_record_stop_again_use_and_no_unavailable_label():
    view = read("VoiceOverView.swift")
    for ident in ("voiceover.recordButton", "voiceover.stopButton", "voiceover.recordAgainButton",
                  "voiceover.useTakeButton", "voiceover.recordingTime", "voiceover.micDenied"):
        assert ident in view
    assert "Record (not available)" not in view


def test_use_uploads_by_the_import_path_and_places_at_the_cursor():
    model = read("DraftEditorViewModel.swift")
    body = model.split("func addRecordedVoiceOver(fileURL: URL, at start: Double)", 1)[1].split("\n    }\n", 1)[0]
    # Saved on the phone and shown at once; the upload (same import path) runs in the background.
    assert "uploadPendingVoiceOver(id: id)" in body and "freeStart(start, duration: duration" in body
    finish = model.split("private func finishPendingVoiceOver(", 1)[1].split("\n    }\n", 1)[0]
    assert "VoiceOverImportManager.shared.importAudio(" in finish
    assert "addVoiceOverPlacement(" in finish and "start: at, end: at + length" in finish
    assert "playheadTime: playheadTime" in read("TimelineEditorView.swift")
    assert "model.addRecordedVoiceOver(fileURL: url, at: playheadTime)" in read("VoiceOverView.swift")


def test_microphone_usage_text_is_set():
    assert "INFOPLIST_KEY_NSMicrophoneUsageDescription: CutSell uses the microphone to record your voice-over" in (
        ROOT / "mobile" / "ios" / "project.yml").read_text()
