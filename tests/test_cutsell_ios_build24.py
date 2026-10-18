"""Build 24: background processing + Edits · Processing (Figma 1134:4635, 1134:2683, 1134:2885).
Static checks of the contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_continue_in_background_is_active_at_every_step():
    screen = read("ProcessingScreen.swift")
    button = screen[screen.index('Text("Continue in background")'):screen.index('"continueInBackground"')]
    assert ".disabled(uploading)" not in button and "opacity(uploading" not in button
    assert ".disabled(uploading)" not in screen
    tabs = read("MainTabView.swift")
    assert "CreateFlowView(selectTab: select)\n                .opacity(tab == .create ? 1 : 0)" in tabs
    flow = read("CreateFlowView.swift")
    assert "onClose: { closeProcessing() }" in flow and '"continue_in_background"' in flow


def test_bulk_sheet_go_to_edits_and_stay_here():
    queue = read("ProcessingQueue.swift")
    sheet = queue[queue.index("struct BulkStartedSheet"):]
    for text in ("Your \\(count) videos are being edited", "You can close this screen or the app. Editing keeps going, one video at a time.",
                 "In Edits, in order", "Each one, as soon as it’s done", "We’ll notify you as each cut is ready.",
                 "Go to Edits", "Stay here"):
        assert text in sheet
    flow = read("CreateFlowView.swift")
    assert "bulkSheetCount = assets.count" in flow and "openEditsProcessing()" in flow
    assert 'ProcessingQueue.shared.requestedEditsFilter = "processing"' in flow


def test_edits_processing_filter_and_rows():
    edits = read("ProjectsView.swift")
    assert 'case processing = "Processing", ready = "Ready", drafts = "Drafts", all = "All"' in edits
    assert "processingRows.isEmpty ? [.all, .ready, .drafts] : EditsTab.allCases" in edits
    assert "Cuts being edited now. You can leave the app." in edits
    assert "Each one moves to Ready when it’s done" in edits and "In order" in edits
    assert "Editing keeps going on our servers even if you close the app. We’ll notify you as each cut is ready." in edits
    queue = read("ProcessingQueue.swift")
    for status in ('"QUEUED"', '"UPLOADING \\(p)%"', '"PROCESSING \\(p)%"', '"FAILED"', 'Text("Retry")'):
        assert status in queue
    assert "QueueIconProcessing" in queue and "QueueIconQueued" in queue
    for name in ("QueueIconProcessing", "QueueIconQueued", "QueueIconCompleted"):
        assert (APP / "Assets.xcassets" / "Figma" / f"{name}.imageset" / f"{name}.svg").exists()


def test_ready_moves_at_once_with_the_notice():
    tracker = read("CutTracker.swift")
    assert "ProcessingQueue.shared.ready(projectID: cut.id, title: cut.title)" in tracker
    assert "AppStateBridge.markReady?(cut.id)" in tracker
    tabs = read("MainTabView.swift")
    assert "ReadyToast(name: toast.name)" in tabs
    queue = read("ProcessingQueue.swift")
    assert 'Text("✓  \\(Self.short(name)) is ready")' in queue and 'Button("View"' in queue


def test_work_goes_on_after_leaving_or_closing():
    app = read("CutSellApp.swift")
    assert "BackgroundFollow.enteredBackground()" in app and "BackgroundFollow.register()" in app
    assert "ResumeRunner.resume(record, appState: appState)" in app
    plist = (ROOT / "mobile" / "ios" / "Config" / "Info.plist").read_text()
    assert "<string>fetch</string>" in plist and "ai.cutsell.app.follow-cuts" in plist
    finisher = read("PendingUploadFinisher.swift")
    assert '"upload_auto_resumed"' in finisher


def test_bulk_chip_and_processing_chip_are_findable():
    ios = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"
    hub = (ios / "HomeHubView.swift").read_text()
    # The card's own Single tap is only on its upper part; chips are separate buttons.
    assert "chipTappedAt" not in hub and ".onTapGesture { onUpload(.single) }" in hub
    gallery = (ios / "GalleryPickerView.swift").read_text()
    assert ".onChange(of: initialMode)" in gallery
    projects = (ios / "ProjectsView.swift").read_text()
    assert projects.count('.accessibilityElement(children: .contain)\n                            .accessibilityIdentifier("editsTabPicker")') == 2


def test_bulk_sheet_buttons_keep_their_ids():
    ios = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"
    queue = (ios / "ProcessingQueue.swift").read_text()
    assert '.accessibilityElement(children: .contain)\n        .accessibilityIdentifier("bulkStartedSheet")' in queue


def test_paused_uploads_follow_the_queue():
    ios = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"
    projects = (ios / "ProjectsView.swift").read_text()
    assert "!ActiveUploads.shared.contains(pending.projectID)" in projects
    assert ".onReceive(queue.$items.dropFirst())" in projects
