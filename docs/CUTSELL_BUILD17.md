# CutSell build 17: sounds, Volume / Mute per clip, Clean up voice, keyframes

Branches:
- App: `feat/figma-part1`.
- Server: `feat/editor-v4-mirror-export` (commit 39d0b46 and later). Deployed only with the Product Owner's permission.

No merge to `main`. No audio, video or keys in the repository.

## What changed for the creator

| Item | iPhone | Export (server) |
|---|---|---|
| CutSell sounds | Select, hold-and-move, Split, trim by the edges, Delete, like a voice-over. The touch area is at least 32 pt wide. A sound is added only once its file is on the iPhone: until then its button shows a spinner and "Add at playhead" says "Getting sound…". Downloads retry by themselves (3 s, 10 s, 30 s). The preview rebuilds when a file arrives. A new sound comes selected. | Mixed at its time and volume (unchanged). |
| Volume / Mute | With a clip selected (video, voice, sound or video overlay) the panel says "Selected clip" and works on that clip only. With nothing selected it says "Whole video" and works on the whole sound. A silent clip shows a red mute icon on the timeline. The video row's speaker silences the video's own sound. Above 100 % the preview adds the extra gain with an audio tap (AVAudioMix stops at 1.0). | `audio_clips[].muted`; `master_volume` / `master_muted` on the finished mix; the QC treats a Whole video mute as the creator's choice. |
| Clean up voice | A clear On/Off switch: "Removes background noise from the voice". On asks the server for the clean copy and shows "Cleaning… %". When it is ready the preview plays it; Off plays the original. | One stored clean copy per source video or per voice / music recording (`audio_clean.py`). The export plays that same file. If the export starts before the copy exists, it makes the copy first, so it never plays the uncleaned sound. |
| Auto Zoom AI | The cut engine's push-ins become normal keyframes once, when the editor opens: four ◇ per zoom. They show in the preview and on the clips. The Style toggle shows or hides them. | Drawn only while `auto_zoom_enabled` is on. |
| Manual keyframes | Select a clip or a Float overlay, put the cursor on a moment, then pinch, drag or twist the video and tap ◇. The twist only counts past 10° and snaps to straight. On a ◇ the button is filled and tapping it deletes that ◇. A ◇ can be dragged to another moment. Undo and Redo work. | Video: one perspective warp per frame (zoom about the centre, clockwise turn, move). Float overlays: zoompan on a fixed canvas, then rotate, then a moving overlay. |

## Draft fields (app ↔ API ↔ export)

| Field | Values |
|---|---|
| `keyframes[]` | `keyframe_id`, `target` (video \| overlay), `source_asset_id` + `source_time` (video), `overlay_id` + `time` (overlay, seconds after its start), `scale` (video 1–5, overlay 0.2–5), `x`, `y` (−1…1 of the frame), `rotation` (°, clockwise), `origin` (ai \| user), `kind`, `group` |
| `audio_clips[].muted` | bool |
| `selected[].clean_voice` | bool (this clip plays the clean copy of its source) |
| `video_clean_voice` | bool (every clip plays the clean copy) |
| `master_volume` | 0–2 (absent = 1) |
| `master_muted` | bool |

Interpolation (app and server): inside one video clip (or one overlay) keyframes are sorted by time. The value is held before the first and after the last. Between two keyframes it follows smoothstep `a + (b − a)·u²(3 − 2u)`. A video pan is limited to `(scale − 1) / 2`.

An AI push-in (`zoom_keyframes`: t, d, s) is exactly four keyframes: 1 at t, s at t+r, s at t+d−r, and 1 at t+d, with r = min(0.35, d/3). This is the curve the export always drew; a test checks it.

## API

- `POST /v1/projects/{id}/clean-audio` `{user_id, target: source|asset, id}`. Returns `ready` (with a 1-hour link) or `cleaning` (the worker job is started only once).
- `GET  /v1/projects/{id}/clean-audio?user_id&target&id`. Returns `ready` / `cleaning` (with `progress`) / `failed` / `missing`.
- Clean copies are stored at `{upload prefix}/{user16}/{project16}/clean/v1/{target}-{id16}.m4a`, inside the project's own folder, so they are deleted with the project or the account.

## Fixed along the way

- **Auto Zoom:** the export's push-in went towards the top-left corner. Scale-then-crop kept the crop at x = 0, because crop does not see the per-frame size. It is now one perspective warp about the centre.
- **Accessibility:** panels and timeline rows replaced their children's identifiers with their own, so VoiceOver and UI tests could not find Mute, Clean up, sound blocks or ◇. They now use `.accessibilityElement(children: .contain)`.

## Tests

- **Server:**
  - `tests/test_cutsell_build17_end_to_end.py`, real renders measured:
    - Gap noise is more than 12 dB lower after Clean up. The clean copy is made once and reused, and the export's sound equals the stored copy.
    - A muted sound is silent while the others play. Whole video at 50 % is about −6 dB, and Mute silences everything.
    - Two keyframes zoom from 100 % to 130 % smoothly.
    - Rotation and pan keyframes work.
    - AI keyframes follow the Auto Zoom toggle.
    - An overlay moves and doubles in size between its keyframes.
    - The clean-audio API works, and every new field survives the API round trip.
  - The full suite has the same 63 failures as the live code before build 17 (environment), and none are new.
- **App:**
  - `tests/test_cutsell_ios_build17.py`.
  - The Simulator probe with real touches: `.github/workflows/cutsell-ios-build17-probe.yml` (XCUITest `mobile/ios/CutSellUITests/Build17UITests.swift`). Frames and logs go to `design-screenshots/b17/`; the screen recording stays on the test machine.
