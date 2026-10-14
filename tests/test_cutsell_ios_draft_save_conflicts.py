"""Build 11 report: "draft revision conflict: expected 12, current 13 (409)" on any edit.

Every draft change now goes through ONE save path: built from the server's latest draft, saved
with one PUT whose answer becomes the local draft, one at a time; a 409 reloads the draft, makes
the change again and retries once with no alert; only a second failure is shown, in plain words.
Choosing what is already chosen saves nothing. File-like cut names show as "Cut 13 Oct"."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "mobile" / "ios" / "CutSell"
VM = (APP / "DraftEditorViewModel.swift").read_text()


def _func(name: str) -> str:
    start = VM.index(name)
    return VM[start:VM.index("\n    }\n", start)]


def test_no_second_save_path_left():
    assert "func autosave(" not in VM and "func edit(path:" not in VM
    assert VM.count('method: "PUT"') == 2  # the save path + the Debug-only probe helper
    assert "probeMakeLocalRevisionStale" in VM.split("#if DEBUG")[1]


def test_save_takes_the_server_answer_and_retries_a_conflict_once():
    put = _func("private func put(_ draft: JSONValue, over base: DraftSnapshot)")
    assert '"expected_revision": .number(Double(base.revision))' in put and "snapshot = saved" in put
    commit = _func("private func commit(")
    assert "for attempt in 0..<2" in commit
    assert "built = try await build(base.draft)" in commit           # rebuilt on the reloaded draft
    assert "if attempt == 0, await reloadDraft() { continue }" in commit
    assert "if edited == base.draft { return true }" in commit         # nothing changed: no save
    assert "errorMessage = Self.friendly(APIError.http(409" in VM
    friendly = _func("static func friendly(")
    assert "409" not in friendly and "This cut was changed somewhere else" in friendly


def test_every_public_edit_runs_in_the_queue():
    # Public functions call the save path only through `perform` (one at a time).
    for match in re.finditer(r"\n    func (\w+)[^\n]*\{\n(.*?)\n    }\n", VM, re.S):
        name, body = match.group(1), match.group(2)
        for line in body.splitlines():
            if "editAndSave(" in line or "saveChange(" in line:
                assert "perform(" in line, (name, line)


def test_choosing_the_same_caption_option_saves_nothing():
    apply = _func("func applyCaptionSettings(")
    assert "changes = changes.filter" in apply and "return current != value" in apply
    assert 'changed["caption_preset"]?.stringValue' in apply


def test_file_names_are_not_shown_as_the_cut_title():
    naming = (APP / "EditorV2View.swift").read_text()
    block = naming[naming.index("static func isInternal"):naming.index("static func dated")]
    for prefix in ('"copy_"', '"img_"', '"media-"'):
        assert prefix in block
    assert '"[0-9a-f]{8,}"' in block


# --- Editor speed + preview = the cut (build 11.1) ---

PLAYBACK = (APP / "DraftPlaybackView.swift").read_text()


def test_preview_is_the_cut_and_stops_at_its_end():
    assert "item.forwardPlaybackEndTime = cutEnd" in PLAYBACK
    assert "composition.removeTimeRange(CMTimeRange(start: cutEnd, end: composition.duration))" in PLAYBACK
    assert "AVPlayerItem.didPlayToEndTimeNotification" in PLAYBACK and "self.stopAtEnd()" in PLAYBACK
    assert "min(seconds, self.duration)" in PLAYBACK  # the time (and timeline) never passes the end


def test_source_video_kept_on_the_phone():
    caches = (APP / "LocalCaches.swift").read_text()
    assert "actor SourceVideoCache" in caches and "static let maxBytes" in caches
    assert "SourceVideoCache.localURL(forURI: uri)" in PLAYBACK and "SourceVideoCache.shared.prefetch" in PLAYBACK
    assert "SourceVideoCache.shared.store(fileURL: video.url, uri: result.sourceURI)" in (APP / "CutSubmission.swift").read_text()


def test_editor_and_edits_list_open_from_what_the_phone_kept():
    load = _func("    func load() async {")
    assert "DraftSnapshotCache.load(projectID: project.projectID)" in load
    app = (APP / "AppState.swift").read_text()
    assert "ProjectListCache.load(userID: current.userID)" in app and "ProjectListCache.save(" in app
    v4 = (APP / "EditorV4View.swift").read_text()
    assert "ThumbnailStore.shared.image(for: project.projectID)" in v4 and "model.loadGeneration" in v4
