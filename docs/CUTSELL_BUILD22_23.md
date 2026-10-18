# CutSell iOS builds 22 and 23 (18 Oct 2026)

## Build 22: uploads that finish

**What happened on 18 Oct.** Swanny's Diagnostics (build 21) and the Render log showed the following.

- R2 accepted all three uploads: `complete` answered 200 at 09:28:56, 09:32:32 and 10:03:04 (local time).
- The cut never started, because iOS suspended CutSell right after it sent `complete`.
- A repeated `complete` then got a 404, because the server had already closed the session. Diagnostics logged it as a bare `APIError`.
- "Try again" then uploaded everything again from 0.

**Fixes.**

- `UploadLedger`: part ETags and the complete-sent and complete-done state are kept on disk. ETags are saved even when the part finished in the background session with CutSell suspended or relaunched; `taskDescription` carries the upload id and the part number.
- A 404 after `complete` was sent counts as done. The resumable record stays until the cut has started.
- `PendingUploadFinisher`: an upload whose parts are all on R2 is completed and its cut started. This runs on launch, on reopen, and when iOS wakes CutSell for finished parts, with fresh background time from iOS.
- Diagnostics events for the final step: `upload_complete_started`, `upload_complete_done` and `upload_complete_failed`, then `processing_started`, `job_submitted` and `job_submit_failed`. Each carries the real code and message.
- Photos: the original resource is copied, not the edited render.
- Several clips:
  - each clip is its own upload, retried on its own;
  - in Multiple mode, a clip that fails no longer stops the others;
  - every event carries `source_order`.

**Probe (simulator).** 113 MB video, network limited to 0.5 MB/s, CutSell left for 60 s mid-upload and for 60 s near the end. Every check passed and no "Try again" was needed. It reached the editor 302 s after Create.

## Build 23: audio first, iCloud light, 3 parts at a time

**Audio first.**

1. The iPhone extracts the voice: AAC mono at 64 kb/s, about 0.4 MB per minute.
2. The voice uploads with a single PUT.
3. The cut starts from the voice; `sources[i].metadata.audio_uri` carries it.
4. The video's multipart session is created first, so the draft points at the video URI.
5. The editor plays the iPhone's own copy.
6. The video uploads in the background (`BackgroundVideoUploads`), then the app calls `POST /v1/projects/{id}/video-ready`. That runs a follow-up worker job, which registers the primary video.
7. Export waits for the video.

The flow is used only when `GET /v1/capabilities` answers `audio_first: true`. Setting the UserDefaults key `cutsell.audioFirst.off` switches it off on a device.

**iCloud light.** For a video that is only in iCloud, under 4K, with a known size:

- A 960x540 fast copy comes down first and is used for the voice and the preview.
- The original downloads byte for byte in the background, then uploads.

This is not testable in the simulator.

**Parts.** The server now honours 8 MiB parts. The foreground session sends at most 3 parts at a time; queued all at once, waiting parts had been timing out at 90 s.

**Progress screen.** "Preparing… / Uploading audio… / Cutting your video… / Uploading video in background · % · time left".

**Diagnostics timings.**

- `audio_extracted`, `audio_uploaded`, `stage_timing`
- `server_step` (seconds since the cut was sent)
- `editor_opened`
- `video_background_started`, `video_background_done`, `video_background_failed`
- `combine_started`, `combine_done`, `combine_failed`
- `icloud_light`, `icloud_original_done`

**Measured.** Simulator, outgoing traffic limited to about 1 MB/s, timed from tapping Create to the cut open in the editor:

| Video | Before (whole video first) | Audio first |
|---|---|---|
| 1 min, 81 MB | 131 s | 35 s |
| 3 min, 217 MB | about 305 s to ready | 73–81 s |

For the 3-minute video with audio first:

| Stage | Time |
|---|---|
| Voice extraction | 16–25 s (simulator) |
| Voice upload | 2 s |
| Server, submitted to ready | about 31–36 s |
| Editor open | about 4 s |
| Full video in the background | about 248 s, with no retries |

## Server, deployed 18 Oct with the Product Owner's permission

Commit 429b73f on `feat/simple-engine-v1`, deployed to Render (API) and Modal (worker).

- **Multipart complete is idempotent.** The session is kept for 24 h as completed. A repeated complete or a status request gets the same answer.
- **R2 refusals** are answered as 502 with R2's own code and message.
- **Optional parameters:** `part_size` (5–64 MiB) and `expires_in` (60–3600 s).
- **Audio first:**
  - the worker cuts from `metadata.audio_uri`, with no video format gate and no shot snap for that first cut;
  - `run_video_followup_job` and `POST /v1/projects/{id}/video-ready` handle the video when it arrives;
  - `GET /v1/capabilities` tells the app what the server supports.

**Rollback.** Render: redeploy db3f61d. Modal: deploy cb8163f.
