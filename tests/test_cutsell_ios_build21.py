"""Build 21: the upload Swanny's iPhone cancelled by itself (build 19 report: parts still sending,
the 30 s watch fired, NSURLError -999). Static checks of the new contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (APP / name).read_text()


def test_watch_is_90_seconds_on_bytes_and_never_cancels():
    reliability = read("UploadReliability.swift")
    watch = reliability[reliability.index("enum StallWatch"):reliability.index("final class Once")]
    assert "static let limit: TimeInterval = 90" in watch
    assert ".cancel()" not in watch.replace("watcher.cancel()", "")
    assert "submitter.markStalled(" in watch and '"upload_stalled"' in watch
    # Back from the background: not a stall.
    assert "Date().timeIntervalSince(lastTick) > 10" in watch
    # Repeated byte counts (the 2 s poll) are not activity.
    assert "guard bytes > (inFlight[part] ?? 0)" in reliability


def test_try_again_restarts_only_the_parts_in_flight():
    submitter = read("CutSubmission.swift")
    assert "MultipartUploadManager.restartStuckParts()" in submitter
    assert "if stalled != nil {" in submitter and "self?.touch()" in submitter
    screen = read("ProcessingScreen.swift")
    assert '"uploadStalled"' in screen and '"uploadTryAgain"' in screen
    assert "onTryAgain: { submitter?.restartStuckParts() }" in screen


def test_parts_go_to_the_background_session_each_retried_on_its_own():
    manager = read("MultipartUploadManager.swift")
    assert "BackgroundPartUploader.shared.upload(fileURL: file" in manager
    assert "error.code == NSURLErrorUnknown" in manager and "backgroundBroken.set()" in manager
    assert 'Retry.run("part \\(partNumber)"' in manager
    assert "configuration.timeoutIntervalForRequest = 90" in manager
    assert "maxInFlight" not in manager
    uploader = read("BackgroundPartUploader.swift")
    assert "configuration.httpMaximumConnectionsPerHost = 3" in uploader
    assert "task.countOfBytesSent" in uploader and "repeating: 2" in uploader
    assert "func restartStuckParts()" in uploader
    reliability = read("UploadReliability.swift")
    assert 'init(_ name: String = "cutsell-upload")' in reliability and "beginBackgroundTask(withName: name)" in reliability


def test_eight_mib_parts_requested():
    manager = read("MultipartUploadManager.swift")
    assert "static let requestedPartSize = 8 * 1024 * 1024" in manager
    assert "part_size: Self.requestedPartSize" in manager
    # The app always cuts the file with the size the server answers.
    assert "Int64(start.partSize)" in manager


def test_preparation_never_makes_the_file_heavier():
    prep = read("VideoPreparation.swift")
    assert "codec == kCMVideoCodecType_H264 || codec == kCMVideoCodecType_HEVC" in prep
    assert "if fineCodec && under4K" in prep and '"prepare_skipped"' in prep
    assert "guard preparedSize > 0, preparedSize < size else" in prep
    assert '"mb_before"' in prep and '"mb_after"' in prep
    assert "AVAssetExportPresetHEVC1920x1080" in prep and "fileLengthLimit" in prep
    assert "targetBitsPerSecond: Double = 10_000_000" in prep
