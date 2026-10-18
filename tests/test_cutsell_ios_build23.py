"""Build 23: audio first -- the cut starts from the voice (AAC mono ~64 kb/s, ~0.5 MB per minute) while
the full video uploads in the background; only with a server that says it can (GET /v1/capabilities).
Static checks of the contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_voice_is_small_aac_mono():
    audio = read("AudioExtractor.swift")
    assert "kAudioFormatMPEG4AAC" in audio and "AVNumberOfChannelsKey: 1" in audio
    assert "static let bitRate = 64_000" in audio and '"audio_extracted"' in audio


def test_only_with_a_server_that_can():
    first = read("AudioFirst.swift")
    assert '"/v1/capabilities"' in first and "answer.audio_first == true" in first
    submitter = read("CutSubmission.swift")
    assert "await ServerCapabilities.audioFirst()" in submitter and "resumeProjectID == nil" in submitter


def test_cut_starts_from_the_voice_and_the_video_follows_in_the_background():
    submitter = read("CutSubmission.swift")
    voice = submitter[submitter.index("func startFromVoice"):]
    order = [voice.index(x) for x in ("AudioExtractor.extract", "MultipartUploadManager.shared.begin",
                                      "SourceVideoCache.shared.store", "VoiceUpload.upload", "Self.startJob",
                                      "PendingCutStore.shared.markStarted", "BackgroundVideoUploads.shared.start")]
    assert order == sorted(order)
    assert 'metadata: ["audio_uri": voiceURI]' in voice and "uri: starts[index].sourceURI" in voice
    first = read("AudioFirst.swift")
    assert '"/v1/projects/\\(projectID)/video-ready"' in first and '"video_background_done"' in first
    assert "Uploading video in background" in first


def test_export_waits_for_the_video_and_finisher_resumes_it():
    editor = read("DraftEditorViewModel.swift")
    assert "while BackgroundVideoUploads.shared.isUploading(project.projectID)" in editor
    finisher = read("PendingUploadFinisher.swift")
    assert "if record.jobID != nil {" in finisher and "BackgroundVideoUploads.shared.start(record: record, session: current)" in finisher
    assert "$0.jobID == nil" in read("ProjectsView.swift")


def test_progress_is_honest():
    tracker = read("CutTracker.swift")
    assert "Cutting your video…" in tracker
    screen = read("ProcessingScreen.swift")
    assert '"backgroundVideoUpload"' in screen
