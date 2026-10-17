"""Build 19: a new cut must never stay at "Uploading 0 %" (Swanny's iPhone, builds 16-18). Render's
logs showed the iPhone never even created the project: the AI notice was a SwiftUI sheet on the app
root, hidden under the upload screen (a full-screen cover), so the cut waited for an "Agree" that
never appeared. Static checks; the Simulator probe (.github/workflows/cutsell-ios-build19-probe.yml)
makes real cuts through the Create flow on the live server."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def test_ai_notice_is_presented_on_top_of_everything_and_asked_before_the_upload_screen():
    consent = read("AIConsent.swift")
    assert "UIHostingController(rootView: AIConsentSheet())" in consent
    assert "while let next = top?.presentedViewController" in consent
    app = read("CutSellApp.swift")
    assert ".sheet(isPresented: $consent.asking)" not in app
    flow = read("CreateFlowView.swift")
    assert "guard await AIConsentCenter.shared.ensure() else { return }\n            start(newJob)" in flow


def test_every_step_shows_its_real_percent():
    flow = read("CreateFlowView.swift")
    assert '"Downloading from iCloud"' in flow
    assert "options.progressHandler = { value in progress(value) }" in read("GalleryPickerView.swift")
    prep = read("VideoPreparation.swift")
    assert "progress(Double(exporter.progress))" in prep
    submitter = read("CutSubmission.swift")
    assert 'stage("Creating your project")' in submitter and "self?.stage(uploadTitle, partProgress)" in submitter
    uploader = read("BackgroundPartUploader.swift")
    assert "didSendBodyData" in uploader and "callback?(totalBytesSent)" in uploader
    assert "ByteTally(total: size" in read("MultipartUploadManager.swift")
    assert 'Text(stage.percent >= 0 ? "\\(stage.percent)%" : "…")' in read("ProcessingScreen.swift")


def test_three_tries_and_a_30_second_stall_watch_with_the_real_error():
    reliability = read("UploadReliability.swift")
    assert "static func run<T>(_ what: String, attempts: Int = 3" in reliability
    assert "limit: TimeInterval = 30" in reliability and "if AIConsentCenter.shared.asking { submitter.touch(); continue }" in reliability
    manager = read("MultipartUploadManager.swift")
    for what in ('Retry.run("start"', 'Retry.run("part \\(partNumber)"', 'Retry.run("complete"', 'Retry.run("status"'):
        assert what in manager, what
    assert 'Retry.run("project"' in read("CutSubmission.swift")
    flow = read("CreateFlowView.swift")
    assert "try await StallWatch.run(submitter) { try await work(current) }" in flow
    assert "resumeProjectID: projectID" in flow and 'primaryTitle: "Try again"' in flow


def test_r2_multipart_contract_unchanged():
    manager = read("MultipartUploadManager.swift")
    # Same part size for every part but the last (the server's 16 MiB), every ETag kept, parts sorted.
    assert "let readCount = min(start.partSize, remaining)" in manager
    assert ".sorted { $0.partNumber < $1.partNumber }" in manager
    assert "guard parts.count == start.partCount else { throw UploadError.incompleteUpload }" in manager


def test_probe_makes_cuts_through_the_create_flow():
    workflow = (ROOT / ".github" / "workflows" / "cutsell-ios-build19-probe.yml").read_text()
    assert "-only-testing:CutSellUITests/Build19UITests" in workflow and "SWIFT_OPTIMIZATION_LEVEL=-O" in workflow
    assert "3840x2160" in workflow and "simctl addmedia" in workflow and "grant photos" in workflow
    tests = (ROOT / "mobile" / "ios" / "CutSellUITests" / "Build19UITests.swift").read_text()
    for step in ('"hubUpload"', '"galleryTile"', '"styleCreateButton"', '"aiConsentAgree"', '"readyReviewButton"', '"timelineV4"'):
        assert step in tests, step


def test_every_step_is_an_event_in_diagnostics():
    events = {
        "AIConsent.swift": ['"consent_shown"', '"consent_accepted"'],
        "CreateFlowView.swift": ['"asset_export_started"', '"asset_export_progress"', '"asset_export_done"'],
        "GalleryPickerView.swift": ['"asset_export_failed"'],
        "CutSubmission.swift": ['"project_created"', '"upload_started"', '"upload_progress"', '"upload_failed"', '"prepare_progress"'],
        "UploadReliability.swift": ['"upload_failed"', '"upload_retry"'],
    }
    for name, needles in events.items():
        source = read(name)
        for needle in needles:
            assert needle in source, (name, needle)
    # A video already on the iPhone reports no %: the bytes written keep the copy alive on screen.
    assert "copied?(size)" in read("GalleryPickerView.swift")


def test_audio_is_playback_from_launch_without_stopping_other_apps():
    tools = read("AudioTools.swift")
    assert "static func setUpAtLaunch()" in tools and "options: [.mixWithOthers]" in tools
    assert "EditorAudio.setUpAtLaunch()" in read("CutSellApp.swift")
