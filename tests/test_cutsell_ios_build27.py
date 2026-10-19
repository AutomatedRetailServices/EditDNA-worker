"""Build 27: Export says what it is doing at every step (Diagnostics), never stays silent, plain words in the app."""
from pathlib import Path

IOS = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"


def read(name):
    return (IOS / name).read_text()


def test_every_export_step_is_logged():
    model = read("DraftEditorViewModel.swift")
    for event in ("export_tapped", "export_waiting", "export_blocked", "export_requested", "export_accepted", "export_failed"):
        assert f'"{event}"' in model
    tracker = read("ExportTracker.swift")
    for event in ("export_progress", "export_render_done", "export_failed"):
        assert f'"{event}"' in tracker
    actions = read("FinishedExportActionsView.swift")
    assert '"export_downloaded"' in actions and '"export_saved_to_photos"' in actions and '"Retry"' in actions


def test_an_export_that_cannot_start_says_so_with_try_again():
    model = read("DraftEditorViewModel.swift")
    assert "exportError = " in model and 'CutSellDiagnostics.log("export_blocked", ["reason": "draft_not_loaded"])' in model
    sheet = read("EditorV2Sheets.swift")
    assert "model.exportError" in sheet and "Waiting for a free editor" not in sheet and "Preparing your export…" in sheet


def test_inter_already_registered_is_not_reported_as_a_failure():
    assert "CTFontManagerError.alreadyRegistered" in read("CaptionOverlayView.swift")
