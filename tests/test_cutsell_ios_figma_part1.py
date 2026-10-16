"""iPhone, Figma part 1: design system, tab bar (Create · Edits · Plan · You, opens on Create),
Create (camera, gallery Single/Bulk/Combine, style sheet, permissions, problems), Edits grid,
the bell inside You -- and the upload / cut pipeline unchanged."""
import json
from pathlib import Path

IOS = Path(__file__).resolve().parents[1] / "mobile" / "ios"
APP = IOS / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_tabs_order_and_app_opens_on_create():
    tabs = read("MainTabView.swift")
    assert "case create, edits, plan, you" in tabs
    assert "init(initialTab: MainTab = .create)" in tabs
    assert "MainTabView()" in read("CutSellApp.swift")
    assert "ProjectsView()" not in read("CutSellApp.swift")


def test_bell_moved_into_you():
    assert "NotificationInboxView()" in read("YouView.swift")
    assert 'Image(systemName: "bell")' in read("YouView.swift")
    edits = read("ProjectsView.swift")
    assert "showingNotifications" not in edits and '"bell"' not in edits


def test_one_upload_pipeline_shared_by_create_and_resume():
    submitter = read("CutSubmission.swift")
    assert submitter.count('"/v1/flow-b/jobs"') == 1
    assert "MultipartUploadManager.shared.upload(" in submitter
    assert "PendingCutStore.shared.save(pending)" in submitter
    assert "VideoPreparation.shared.prepare(" in submitter
    # The two callers use it; neither keeps its own copy of the job request.
    for name in ("NewCutView.swift", "CreateFlowView.swift"):
        source = read(name)
        assert "submitter.submit(" in source
        assert '"/v1/flow-b/jobs"' not in source
    # Resumable uploads keep the mode names already saved on phones.
    assert 'static let singleMode = "Single clip"' in submitter
    assert 'static let multipleMode = "Multiple clips"' in submitter
    assert 'case single = "Single clip"' in read("NewCutView.swift")


def test_gallery_modes_and_texts_match_figma():
    gallery = read("GalleryPickerView.swift")
    for text in (
        '"One video → one edit."', '"Each video becomes a separate edit."',
        '"One edit in the order you selected."', '"Recents"', 'return "Combine \\(max(count, 0)) clips"',
    ):
        assert text in gallery, text
    assert "PHAssetResourceManager.default().writeData" in gallery
    assert "options.isNetworkAccessAllowed = true" in gallery
    flow = read("CreateFlowView.swift")
    # Combine = one cut with several sources; Bulk = one cut per video.
    assert "if current.mode == .combine" in flow and "CutSubmitter.multipleMode" in flow


def test_style_sheet_overlap_and_auto_zoom_are_sent_with_the_cut():
    sheet = read("StyleSheetView.swift")
    assert '"Choose your style"' in sheet and '"Use this style for my next videos"' in sheet
    assert '"Overlap audio"' in sheet and '"Auto Zoom AI"' in sheet
    assert "Pace" not in sheet  # Pace was removed (FICHA_03)
    assert "var overlapAudio: Bool = true" in sheet and "var autoZoom: Bool = true" in sheet  # both ON by default
    assert "UserDefaults.standard" in sheet
    submit = read("CutSubmission.swift")
    assert "dialogue_overlap_enabled: audioOverlap" in submit and "audio_overlap: audioOverlap" in submit
    assert "auto_zoom_enabled: autoZoom" in submit
    flow = read("CreateFlowView.swift")
    assert flow.count("audioOverlap: current.style.overlapAudio, autoZoom: current.style.autoZoom") == 3


def test_limits_and_problem_screens_follow_figma():
    alerts = read("AlertScreens.swift")
    assert "maxDurationSeconds: Double = 10 * 60" in alerts
    assert "maxFileBytes: Int64 = 1_000_000_000" in alerts
    assert "maxClips = 10" in alerts
    assert '["mp4", "mov", "m4v"]' in alerts
    for text in ('"Unsupported media"', '"Upload limit exceeded"', '"This video is too long"',
                 '"10 clips maximum"', '"Something went wrong"', '"Formats and limits"',
                 '"Allow camera access"', '"Allow microphone access"', '"Select More Photos…"'):
        assert text in alerts, text


def test_camera_asks_permission_with_figma_screens_before_starting():
    camera = read("CreateCameraView.swift")
    assert "AVCaptureDevice.authorizationStatus(for: .video)" in camera
    assert "AVCaptureDevice.authorizationStatus(for: .audio)" in camera
    assert "PermissionScreen(kind: .camera" in camera and "PermissionScreen(kind: .microphone" in camera
    assert "camera.record {" in camera
    assert "func setTorch(_ on: Bool)" in read("CameraController.swift")


def test_screenshot_mode_is_debug_only():
    shots = read("DesignScreenshots.swift")
    assert "#if DEBUG" in shots and "struct DesignScreenshotRoot" in shots
    app = read("CutSellApp.swift")
    assert "#if DEBUG" in app and "DesignScreenshotRoot(screen: screen)" in app


def test_figma_assets_and_inter_fonts_are_bundled():
    figma = APP / "Assets.xcassets" / "Figma"
    for name in ("BrandLogo", "TabCreate", "TabEdits", "TabPlan", "TabYou", "IconFlip", "IconFlash",
                 "IconGrid", "IconUpload", "IconBack", "IconSearch", "IconCheck", "EmptyPlay"):
        contents = json.loads((figma / f"{name}.imageset" / "Contents.json").read_text())
        files = [image["filename"] for image in contents["images"]]
        assert files and all((figma / f"{name}.imageset" / f).exists() for f in files), name
    for weight in ("Regular", "Medium", "SemiBold"):
        assert (APP / "Fonts" / f"Inter-{weight}.otf").exists()
    assert (APP / "Fonts" / "LICENSE-Inter-UI.txt").exists()


def test_no_secrets_or_qa_media_in_new_files():
    for name in ("Theme.swift", "MainTabView.swift", "CreateCameraView.swift", "CreateFlowView.swift",
                 "GalleryPickerView.swift", "StyleSheetView.swift", "AlertScreens.swift", "YouView.swift",
                 "ProjectsView.swift", "DesignScreenshots.swift", "CutSubmission.swift"):
        source = read(name)
        for term in ("BEGIN PRIVATE KEY", "sk-", "VIDEO-2026-07-30", "5E01F214", "D40F1D43"):
            assert term not in source, (name, term)


def test_failed_start_offers_retry_unless_saved_for_resume():
    submitter = read("CutSubmission.swift")
    assert "throw CutSubmitError(underlying: error, savedForResume: pendingSaved, projectID:" in submitter
    flow = read("CreateFlowView.swift")
    # Build 19: the real error is shown, and "Try again" resumes a saved upload.
    assert "failure = submitError.savedForResume ? .savedForResume(projectID: submitError.projectID, detail: detail) : .notSaved(detail: detail)" in flow
    assert 'primaryTitle: "Try again"' in flow
    # The upload screen opens only after the gallery finished closing (no timer race).
    assert "fullScreenCover(isPresented: $showingGallery, onDismiss:" in flow
    assert "asyncAfter" not in flow


def test_camera_pauses_while_covered_and_permission_screen_is_review_safe():
    camera = read("CreateCameraView.swift")
    assert ".onChange(of: isActive)" in camera and "camera.stopSession()" in camera
    assert "if camera.isRecording { startTimer(limit:" in camera
    alerts = read("AlertScreens.swift")
    assert 'deniedBefore ? "Open Settings" : "Continue"' in alerts
    assert "if let onClose, deniedBefore {" in alerts


def test_size_and_length_limits_wait_for_product_decision():
    assert "static let sizeAndLengthLimitsOn = false" in read("AlertScreens.swift")
