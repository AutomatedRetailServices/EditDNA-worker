"""Voice Over: on the mic row at once when Save is tapped, uploaded in the background, kept on the
phone and retried when the upload fails; "Video sound" panel under the timeline; label not cut."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def body_of(source: str, signature: str) -> str:
    return source.split(signature, 1)[1].split("\n    }\n", 1)[0]


def test_save_shows_the_voice_at_once_and_uploads_in_the_background():
    vm = read("DraftEditorViewModel.swift")
    add = body_of(vm, "func addRecordedVoiceOver(fileURL: URL, at start: Double)")
    # No network call before the voice is on the row: only reading its length and keeping the file.
    assert "importAudio" not in add and "await self" not in add
    assert "PendingVoiceOverStore.duration(of: fileURL)" in add
    assert "PendingVoiceOverStore.keep(fileURL, id: id)" in add
    assert add.index("pendingVoiceOvers.append(item)") < add.index("uploadPendingVoiceOver(id: id)")
    assert "return .placed" in add
    # The upload runs in its own task.
    upload = body_of(vm, "private func uploadPendingVoiceOver(id: String)")
    assert "Task { @MainActor" in upload


def test_failed_upload_keeps_the_recording_and_retries():
    store = read("PendingVoiceOvers.swift")
    assert ".applicationSupportDirectory" in store and "PendingVoiceOvers" in store
    assert "temporaryDirectory" not in store
    assert "static let attemptsPerRound = 4" in store
    vm = read("DraftEditorViewModel.swift")
    upload = body_of(vm, "private func uploadPendingVoiceOver(id: String)")
    assert "for attempt in 0..<PendingVoiceOverStore.attemptsPerRound" in upload
    assert "retryDelay(afterAttempt: attempt)" in upload
    assert "$0.failed = true" in upload
    finish = vm.split("private func finishPendingVoiceOver(", 1)[1].split("\n    }\n", 1)[0]
    # The file is removed only once the voice is on the server.
    assert finish.index("importAudio") < finish.index("PendingVoiceOverStore.discardFile(item)")
    assert "return false" in finish
    # Tried again when the editor opens, and when the block is tapped.
    load = body_of(vm, "func load() async")
    assert "resumePendingVoiceOvers()" in load
    timeline = read("EditorV4Timeline.swift")
    assert "model.retryPendingVoiceOver(id: item.id)" in timeline


def test_pending_voice_is_on_the_mic_row_and_in_the_preview():
    timeline = read("EditorV4Timeline.swift")
    assert "let pending = model.pendingVoiceOvers" in timeline
    assert "status: item.failed ? .failed : .uploading" in timeline
    playback = read("DraftPlaybackView.swift")
    assert "for placement in model.previewVoices" in playback
    vm = read("DraftEditorViewModel.swift")
    assert 'key: "pending_\\(item.id)"' in vm


def test_retry_delays_grow():
    store = read("PendingVoiceOvers.swift")
    assert "case 1: return 2" in store and "case 2: return 6" in store and "default: return 15" in store


def test_video_sound_panel_opens_only_from_the_speaker_or_a_voice_clip():
    v4 = read("EditorV4View.swift")
    # Not drawn over the timeline any more.
    assert ".overlay(alignment: .bottom) { volumeStrip }" not in v4
    assert "soundPanelView(soundPanel)" in v4
    toggle = body_of(v4, "private func toggleVideoSound()")
    assert "soundPanel = soundPanel == .video ? nil : .video" in toggle
    changed = body_of(v4, "private func selectionChanged(_ newValue: EditorSelection?)")
    assert "if case .voiceOver = newValue" in changed and "soundPanel = .voice" in changed
    assert "case .video" not in changed
    # The space under the video makes room for it, so no row is covered.
    assert "(soundPanel == nil ? 0 : SoundPanelV4.height)" in v4


def test_slider_dot_matches_the_percentage():
    v4 = read("EditorV4View.swift")
    assert 'SoundPanelV4(title: "Video sound", icon: "speaker.wave.2.fill", value: $videoVolume, maximum: 2' in v4
    assert 'value: $model.voiceOverVolume, maximum: 1' in v4
    assert "static func knobCenter(value: Double, width: CGFloat, maximum: Double) -> CGFloat" in v4
    assert "knob / 2 + usable * CGFloat(min(max(0, value / maximum), 1))" in v4
    assert '"\\(Int((value * 100).rounded()))%"' in v4


def slider_center(value, width, maximum, knob=18.0):
    usable = max(1.0, width - knob)
    return knob / 2 + usable * min(max(0.0, value / maximum), 1.0)


def slider_value(x, width, maximum, knob=18.0):
    usable = max(1.0, width - knob)
    fraction = min(max(0.0, (x - knob / 2) / usable), 1.0)
    result = round(fraction * maximum * 100) / 100
    return 1.0 if abs(result - 1) < 0.03 else result


def test_slider_math_round_trips():
    # Same formulas as SoundSlider: 0% at the start, 200% at the end, 100% exactly in the middle.
    width = 200.0
    assert slider_center(0, width, 2) == 9
    assert slider_center(2, width, 2) == width - 9
    assert slider_center(1, width, 2) == width / 2
    assert slider_center(1, width, 1) == width - 9  # voice-over: 100% at the end
    for pct in range(0, 201, 5):
        value = pct / 100
        assert abs(slider_value(slider_center(value, width, 2), width, 2) - value) < 0.011


def label_positions(blocks):
    """Python mirror of TimelineV4.voiceLabels: (left, width, title) -> (x, label_width, inside)."""
    out, previous_end = [], float("-inf")
    for left, width, title in sorted(blocks):
        label_width = len(title) * 6.2 + 12
        x = max(left + 7, previous_end + 6)
        previous_end = x + label_width
        out.append((x, label_width, x + label_width <= left + width - 4))
    return out


def test_voiceover_label_is_never_cut():
    timeline = read("EditorV4Timeline.swift")
    # Labels on their own layer above the blocks, laid out so none covers another.
    assert "ForEach(labels, id: \\.key)" in timeline
    assert "let x = max(input.left + 7, previousEnd + 6)" in timeline
    label = timeline.split("struct VoiceLabelV4: View {", 1)[1].split("\n}\n", 1)[0]
    assert ".fixedSize()" in label and ".lineLimit(1)" in label and "truncationMode" not in label
    assert "Capsule().fill(Color(hex: 0x1769FF))" in label
    # A 1.5 s voice right before another (the probe's case): both labels whole, side by side.
    first, second = label_positions([(100, 45, "Voiceover 1"), (150, 180, "Voiceover 2")])
    assert first[2] is False                                   # on its own tab past the short block
    assert second[0] >= first[0] + first[1] + 6                # starts after the first label



def free_start(start, duration, taken):
    at, moved = max(0.0, start), True
    ranges = sorted(taken)
    while moved:
        moved = False
        for r_start, r_end in ranges:
            if at < r_end - 0.001 and at + duration > r_start + 0.001:
                at, moved = round(r_end + 0.01, 3), True
    return at


def test_a_new_voice_never_overlaps_the_voices_on_the_row():
    # The server refuses overlapping voice-overs (TIMELINE_INVALID 422): place it right after.
    assert free_start(1.0, 6.0, [(1.0, 2.5)]) == 2.51
    assert free_start(0.0, 1.0, [(1.0, 2.5)]) == 0.0           # fits before
    assert free_start(0.5, 1.0, [(1.0, 2.5), (2.6, 4.0)]) == 4.01
    store = read("PendingVoiceOvers.swift")
    assert "static func freeStart(_ start: Double, duration: Double, taken:" in store
    assert "at = ((range.end + 0.01) * 1000).rounded() / 1000" in store
    vm = read("DraftEditorViewModel.swift")
    add = body_of(vm, "func addRecordedVoiceOver(fileURL: URL, at start: Double)")
    assert "PendingVoiceOverStore.freeStart(start, duration: duration, taken: takenVoiceRanges(except: nil))" in add
    finish = vm.split("private func finishPendingVoiceOver(", 1)[1].split("\n    }\n", 1)[0]
    assert "takenVoiceRanges(except: item.id)" in finish
    # A background upload never pops an alert.
    assert "errorMessage = shownError" in finish


def test_plus_never_covers_the_voiceover_label():
    timeline = read("EditorV4Timeline.swift")
    assert "let px = max(Self.cursorX + 4, x(lastEnd) + 6, labelEnd + 8)" in timeline
