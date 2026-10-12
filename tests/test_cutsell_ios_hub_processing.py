"""iPhone, Figma part 1 (b): Create opens on the Home Hub (1068:22); the upload happens inside
Processing (670:409) and the progress comes from the cut's REAL state on the server until Ready
(672:405) -> "Review & edit" opens the editor. "Continue in background" keeps following the cut."""
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_create_tab_opens_on_home_hub():
    flow = read("CreateFlowView.swift")
    assert "HomeHubView(" in flow
    assert "CreateFlowView(selectTab: select)" in read("MainTabView.swift")
    hub = read("HomeHubView.swift")
    for text in ('"Upload videos"', '"Record with CutSell"', '"Voice Over project"', '"Recent cuts"', '"See all in Edits"',
                 '"Turn your raw video into a ready-to-sell cut."'):
        assert text in hub, text
    # No invented plan numbers.
    assert "Text(\"7 / 30" not in hub and "cuts ·" not in hub.split("struct HomeHubView", 1)[1]


def test_tracker_polls_the_real_server_state():
    tracker = read("CutTracker.swift")
    assert '"/v1/projects/\\(projectID)"' in tracker and '"/v1/jobs/\\(jobID)"' in tracker
    assert "cut.jobProgress = status.progress" in tracker
    assert 'projectState == "draft_ready" || jobState == "finished"' in tracker
    # Ready while elsewhere: banner in the app, notification when CutSell is in the background.
    assert "readyBanner = cut.id" in tracker and "UNNotificationRequest" in tracker


def test_upload_is_step_one_with_its_real_percent():
    tracker = read("CutTracker.swift")
    assert 'Stage(step: 1, title: "Uploading your video", percent: Int((min(1, max(0, progress)) * 19).rounded()))' in tracker
    assert "20 + Int(Double(job ?? fallback) * 0.8)" in tracker
    screen = read("ProcessingScreen.swift")
    assert "CutTracker.uploadStage(progress: submitter.progress" in screen
    assert "CutTracker.serverStage(cut)" in screen


def test_processing_stays_open_and_follows_the_cut():
    flow = read("CreateFlowView.swift")
    assert "CutProgressView(" in flow and "submitter: submitter" in flow
    assert "startTracking(projectID: projectID" in flow
    assert "onStarted()" not in flow
    screen = read("ProcessingScreen.swift")
    for text in ('"Continue in background"', '"We’ll let you know in CutSell when your cut is ready."',
                 '"Processing continues if you leave this screen."', '"Your cut is ready"', '"Review & edit"'):
        assert text in screen, text
    assert "DraftEditorView(project: project)" in screen


def test_edits_route_uses_the_new_processing_screen():
    detail = read("ProjectsView.swift").split("struct ProjectDetailView", 1)[1]
    assert "CutProgressView(" in detail and "CutTracker.shared.follow(project: project)" in detail


def test_ready_numbers_come_from_the_draft():
    screen = read("ProcessingScreen.swift")
    assert '"/v1/projects/\\(project.projectID)/draft"' in screen
    assert '"YOU JUST SAVED"' in screen and "TimeSavedLedger.saved(forFootage: footage)" in screen


def test_time_saved_is_counted_per_account_once_per_cut():
    ledger = read("TimeSavedLedger.swift")
    assert "static let editingMinutesPerFootageMinute = 2.0" in ledger
    assert '"cutsell.timeSaved.\\(userID)"' in ledger
    assert "if perCut[projectID] == nil, footageSeconds > 0" in ledger
    assert "TimeSavedLedger.record(userID: session.userID, projectID: project.projectID" in read("ProcessingScreen.swift")


def test_editor_load_is_not_cancelled_by_its_spinner():
    editor = read("DraftEditorView.swift")
    spinner = editor.split('ProgressView("Opening draft…")', 1)[1].split("}", 1)[0]
    assert ".task" not in spinner
    assert ".task(id: project.projectID)" in editor and "await model.load()" in editor


def test_no_leaving_during_upload_and_one_cut_at_a_time():
    screen = read("ProcessingScreen.swift")
    assert "private var uploading: Bool { submitter?.isSubmitting == true || (projectID == nil && submitter != nil) }" in screen
    assert '"Keep CutSell open while your video uploads."' in screen
    assert screen.count(".disabled(uploading)") >= 3
    flow = read("CreateFlowView.swift")
    assert "guard !submitter.isSubmitting else { return }" in flow and "submitter.reset()" in flow


def test_polling_stops_on_cancel_or_missing_project():
    tracker = read("CutTracker.swift")
    assert 'jobState == "canceled"' in tracker
    assert "if cut.clientErrors >= 3 { cut.gone = true" in tracker


def test_notification_copy_is_honest():
    screen = read("ProcessingScreen.swift")
    assert "We’ll notify you when your cut is ready." not in screen
    assert "We’ll let you know in CutSell when your cut is ready." in screen
