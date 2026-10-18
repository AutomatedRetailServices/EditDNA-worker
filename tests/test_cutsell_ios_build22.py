"""Build 22: Swanny's report of 18 Oct (build 21). R2 accepted all three uploads ("complete" answered 200
at 09:28:56, 09:32:32 and 10:03:04 in the Render log), but the cut never started: iOS suspended CutSell
right after "complete" was sent, a repeated "complete" got 404 (the server had closed the session) and
"Try again" uploaded everything from 0. Static checks of the new contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_part_etags_are_saved_to_disk_even_from_the_background_session():
    uploader = read("BackgroundPartUploader.swift")
    assert "task.taskDescription = Self.tag(uploadID: uploadID, part: part)" in uploader
    done = uploader[uploader.index("didCompleteWithError error: Error?) {"):]
    # The ETag goes to disk BEFORE looking for a waiting continuation (none after a relaunch).
    assert done.index("UploadLedger.shared.recordPart(") < done.index("guard let item else { return }")
    assert '"upload_part_done_background"' in uploader
    ledger = read("UploadLedger.swift")
    assert "applicationSupportDirectory" in ledger and ".atomic" in ledger
    assert "completeFileProtectionUntilFirstUserAuthentication" in ledger
    manager = read("MultipartUploadManager.swift")
    assert "UploadLedger.shared.recordPart(uploadID: uploadID, part: partNumber, etag: etag)" in manager


def test_final_step_is_visible_with_real_codes():
    manager = read("MultipartUploadManager.swift")
    for event in ('"upload_complete_started"', '"upload_complete_done"', '"upload_complete_failed"'):
        assert event in manager
    assert 'CutSellDiagnostics.log("upload_complete_failed", tagged(Retry.fields(error)))' in manager
    submitter = read("CutSubmission.swift")
    assert '"processing_started"' in submitter and '"job_submitted"' in submitter and '"job_submit_failed"' in submitter
    reliability = read("UploadReliability.swift")
    fields = reliability[reliability.index("static func fields"):reliability.index("static func short")]
    assert 'out["code"]' in fields and 'out["message"]' in fields and "no HTTP answer" in fields
    # Never a bare "APIError": the failure log carries code + message.
    assert 'CutSellDiagnostics.log("upload_failed", Retry.fields(error,' in submitter
    assert "throw APIError.invalidResponse }" not in submitter and "UploadFlowError.notSignedIn" in submitter


def test_a_404_after_complete_was_sent_means_done_never_upload_again():
    manager = read("MultipartUploadManager.swift")
    assert "code == 404 && (sentBefore || sends.value > 1)" in manager
    assert "UploadLedger.shared.markCompleteSent(start.uploadID)" in manager
    resume = manager[manager.index("catch APIError.http(let code, let message) where code == 404 {"):]
    resume = resume[:resume.index("await resumeStore.remove(uploadID: saved.start.uploadID)")]
    assert "completeSentAt != nil" in resume and "return Self.completed(saved.start" in resume
    assert '"upload_already_complete"' in manager
    # The resumable record stays until the cut started on the server.
    complete = manager[manager.index('"upload_complete_started"'):manager.index("static func completed(")]
    assert "resumeStore.remove" not in complete
    assert "MultipartUploadManager.shared.finished(fileURL: video.url" in read("CutSubmission.swift")


def test_resume_uses_r2_list_of_parts_and_sends_only_the_missing_ones():
    manager = read("MultipartUploadManager.swift")
    assert '"/v1/uploads/multipart/\\(saved.start.uploadID)"' in manager
    assert "for part in status.uploadedParts" in manager
    assert "let missing = (1...start.partCount).filter { completedParts[$0] == nil }" in manager


def test_finishes_on_reopen_and_background_wake():
    app = read("CutSellApp.swift")
    assert 'PendingUploadFinisher.run(session: appState.session, reason: "launch"' in app
    assert 'PendingUploadFinisher.run(session: appState.session, reason: "reopen"' in app
    assert "PendingUploadFinisher.wokeForUploads()" in app
    assert "PendingUploadFinisher.uploadEventsDelivered()" in read("BackgroundPartUploader.swift")
    finisher = read("PendingUploadFinisher.swift")
    assert "!ActiveUploads.shared.contains(record.projectID)" in finisher
    assert "state != .allPartsUploaded && state != .completed" in finisher
    assert "CutSubmitter.startJob(" in finisher and 'BackgroundTime("cutsell-auto-finish")' in finisher
    submitter = read("CutSubmission.swift")
    assert 'BackgroundTime("cutsell-finish")' in submitter
    assert "onPartsUploaded: { Task { @MainActor in finishing.begin() } }" in submitter


def test_original_video_from_photos_and_no_reencode():
    gallery = read("GalleryPickerView.swift")
    assert "let original = resources.first(where: { $0.type == .video })" in gallery
    assert "guard let resource = original ?? edited" in gallery
    prep = read("VideoPreparation.swift")
    assert "if fineCodec && under4K" in prep


def test_eight_mib_parts_requested_and_saved_paths_survive_updates():
    manager = read("MultipartUploadManager.swift")
    assert "static let requestedPartSize = 8 * 1024 * 1024" in manager
    assert "ContainerPath.resolve(filePath)" in read("PendingCutStore.swift")
    assert "ContainerPath.resolve(" in read("UploadResumeStore.swift")


def test_each_clip_is_its_own_upload_and_a_failing_clip_does_not_stop_the_others():
    submitter = read("CutSubmission.swift")
    assert "clipOrder: index," in submitter and "clipAttempt >= 3" in submitter
    assert '"clip_upload_retry"' in submitter and '"clip_upload_failed"' in submitter
    assert '"Clip \\(index + 1) of \\(uploadVideos.count) · Uploading"' in submitter
    manager = read("MultipartUploadManager.swift")
    assert 'clipOrder.map { ["source_order": String($0)] }' in manager
    flow = read("CreateFlowView.swift")
    assert '"clip_cut_failed"' in flow and '"multiple_finished"' in flow and "firstFailure = firstFailure ?? error" in flow
