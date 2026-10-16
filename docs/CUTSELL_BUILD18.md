# CutSell build 18: what failed on a real iPhone with build 17, and the fixes

Branches:
- App: `feat/figma-part1`.
- Server: no change. Build 18 needs no deploy.

No merge to `main`. No audio, video or keys in the repository.

Build 17 passed its Simulator probe, but the probe ran a **Debug** build and only read the editor's own state, never what the mix carried. Build 18's probe (`.github/workflows/cutsell-ios-build18-probe.yml`) builds the app **optimized** (`-O`, whole module, like TestFlight). It checks the preview's mix with `AVAssetReader`: the seconds each sound really has in the composition and its level in dB.

## 1. Sounds silent in the timeline preview (priority 1)

**Root cause.** Each sound or voice track was loaded from a temporary `AVURLAsset(url:)` that nothing kept alive:

```swift
guard let voice = try? await AVURLAsset(url: placement.url).loadTracks(withMediaType: .audio).first …
try? track.insertTimeRange(…, of: voice, …)
```

A track whose asset is gone cannot be inserted. The video tracks already documented this ("a track whose asset is gone cannot be inserted (black preview)"). In an optimized build ARC frees the temporary asset at once, and the `try?` hid the failure, so the composition got an empty track and stayed silent. In the Debug Simulator build the asset lived longer, so the sound played there.

**Fixes** (`DraftPlaybackView.swift`):
- **Assets kept alive.** Every audio asset the composition uses stays in `audioAssetCache`, one entry per file version, until a newer mix no longer uses it.
- **Range cut to the file.** The inserted range is limited to what the file really holds (its track `timeRange`). An AAC file can be a few ms shorter than the length the draft gives it.
- **No hidden failures.** A failed insert is logged (`preview_audio_insert_failed`). The report reads what the composition really holds (`track.segments`).
- **Newest rebuild wins.** A rebuild superseded by a newer one stops before it replaces the player (`buildGeneration`). Before, an older rebuild still waiting on the network could finish last and install a mix without the sound just added.
- **Audio session.** `.playback` / `.moviePlayback` is checked at every play, not only once, so a voice-over recording (`.playAndRecord`) cannot leave it changed. The sound plays with the ring/silent switch on silent too.
- **Sound downloads** (`SfxLibrary`). A file counts as ready only once it is on the iPhone and AVFoundation can play it. An unplayable file is deleted and downloaded again. The editor shows "Getting sounds…", or "A sound didn't download · Try again".
- **Sound length.** A new sound gets the file's measured length.
- The probe runs the old way too (`CUTSELL_PROBE_TEMP_ASSET=1`, Debug probe only) on the same optimized build, to show the difference.

## 2. Audio per clip and the video row's speaker

- **"Whole video" and the speaker are one setting:** the video's own sound (`video_volume`, 0 = muted). Sounds, voices and music are never silenced by it.
  - Build 17 used `master_*`, which silenced the whole mix, sounds included.
  - A draft from build 17 with `master_*` set becomes the same setting on `video_volume` once, when the editor opens (`adoptLegacyMasterVolume`).
- **Speaker:**
  - Off silences every clip.
  - On brings every clip back: its level, and any clip silenced on its own.
  - The icon's accessibility value is on / off.
- **Selected clip:** Mute or volume works on that clip only (`audio_muted` / `audio_volume`), as before. The server keeps both fields through saves, split and reload.
- **"Sometimes yes, sometimes no"** came from the rebuild race above. The newest mix now always wins.
- **Preview = Export:** the export already mixes `video_volume`, the per-clip fields and each sound at its own volume.

## 3. Manual keyframes

- **UIKit gestures** (`FramingGestureLayer.swift`):
  - `UIPinchGestureRecognizer`, `UIRotationGestureRecognizer` and `UIPanGestureRecognizer` (up to two fingers) recognize together. They replace build 17's stacked SwiftUI gestures, which did nothing on the iPhone.
  - With a video clip (or nothing) selected, the layer sits under the overlays: an overlay can still be touched.
  - With a Float overlay selected, it sits over them: fingers anywhere on the video move, size and turn that overlay.
  - The overlay case applies to its own place, size and turn, or to its keyframe at the cursor once it has keyframes.
- **Twist:** turning starts past 6° (a pinch always twists a little) and snaps to straight within 5°.
- **◇ keeps the framing.** Playback interpolates smoothly between ◇ (smoothstep, as before). Auto Zoom AI ◇ zoom the preview, and the probe measures the zoom in the picture itself.
- **No conflict with the timeline:** the gestures live only on the video, never on the timeline.

## 4. Overlays on the timeline (photos and videos)

The probe found three causes, and each one would also break the iPhone:
- **Hidden row.** Selecting an overlay opens its panel, and the timeline then shows only two rows, so the overlay's own row was hidden below. The selected item's row now scrolls into view by itself.
- **Selection lost after upload.** A new overlay is selected under its upload id (`pov_…`), and the upload gives it a new id (`ov_…`). The selection now follows the new id (`placedOverlayRename`).
- **Drag read as a tap.** On a selected block the drag now has priority over the tap (`highPriorityGesture` for selected items), so a moving finger never deselects it.


- **Select, move, trim, delete:**
  - Tap selects.
  - A selected block moves with a plain drag. An unselected one still needs hold-and-drag, so a plain swipe keeps scrolling the timeline. The same applies to sounds and voices.
  - The white handles trim either edge, longer or shorter.
  - Delete removes the selected overlay.
- **Split:**
  - A placed overlay splits as before.
  - One still uploading answers "Wait until it finishes uploading, then split it." (it used to say the cursor was outside).

## 5. Gallery

- **Photo opened videos.** The Overlay panel's Photo / Video choice was a separate `@State`, read stale by the sheet on its first presentation, so Photo could open a videos-only gallery. The choice now travels with the sheet (`EditorSheet.overlayPhoto`).

- **Editor overlay and B-roll gallery:** now the system picker (`PHPickerViewController`). It shows the whole library (photos and videos, albums, search) without full photo access.
  - Build 17's own grid read at most 300 items (`fetchLimit = 300`), and with "Limited" access only the photos picked earlier.
  - A picked item not on the iPhone downloads from iCloud with "Getting it from iCloud… %".
- **Create gallery (videos for a new cut):** keeps its design. With Limited access it always shows a banner with "Select more photos" (the system's limited picker) and "Change permission" (Settings).

## 6. Diagnostics (hidden)

- **How to open it:** You → five quick taps on "CutSell 0.1.0 (18)".
- **What it shows:**
  - The build.
  - The audio session: category, mode, output, volume.
  - The Photos permission.
  - Each sound or voice of the last preview: file on the iPhone, length, in the mix, volume, level in dB.
  - Keyframe counts.
  - The last 50 events: sounds, mute, keyframes, framing gestures, uploads, preview.
- **Send report** shares a text file.
- **No personal data:** field names with user, token, email, url, path, name, title, text, caption, project, account, session or key are dropped, and so are values with a link, a path or "@".

## 7. Tests

**Honest note about the sounds.**
- **Optimized Simulator:** the old way (temporary asset) and the new way both put Typing in the mix (in = 3.00 s, -17.4 dB). The Simulator therefore did NOT reproduce the iPhone's silence, so the device cause is not proven.
- **Mix measured:** what the player really plays is measured too (tap): -16.6 dB heard.
- **On the iPhone:** if a sound is still silent there, Diagnostics shows whether its file is on the phone, whether it is in the mix, and how loud it really played.

**Last probe run** (b64031c, run 38962745793, optimized build, live server, real cut): every check PASS.
- **Sounds:**
  - Typing at 0:02 is in the mix and heard while playing.
  - The audio session is Playback / MoviePlayback.
  - Move (plain drag), Split, trim and Delete all rebuild the mix.
- **Volume:**
  - Selected clip mute silences that clip only, and it survives a reload.
  - The speaker worked 6 / 6 times, and the sounds keep playing.
- **Keyframes:**
  - Pinch zooms 1.65× (the white square goes from 0.278 to 0.470 of the width on screen).
  - A two-finger twist turns 30°; ◇ keeps both framings.
  - Between them the framing is smooth (1.65/13° → 1.23/3° → 1.02/0° → 1.00/13° → 1.00/30°).
  - A drag on the video never starts playback.
  - Auto Zoom AI alone (previous run): 1.04 zoom on screen.
- **Overlays, photo and video:**
  - Plain drag, both trim handles, two fingers (size and turn), Split and Delete.
  - A video overlay's start can only be shortened: it can't start before its own first frame.
- **System picker:** a photo picked becomes an overlay.
- **Export:** OK in 62 s.
- **Diagnostics:** opens with five taps, lists the editor's events, and Send report opens the share sheet.

**TestFlight:** build 18 VALID (run 38966527399, commit 04cfff4).


- **Static:** `tests/test_cutsell_ios_build18.py`; older iOS static tests updated (309 pass).
- **Simulator probe:** `.github/workflows/cutsell-ios-build18-probe.yml`, XCUITest `mobile/ios/CutSellUITests/Build18UITests.swift`. Optimized build, live server, real cut. Frames and logs go to `design-screenshots/b18/`; the screen recording stays on the test machine.
