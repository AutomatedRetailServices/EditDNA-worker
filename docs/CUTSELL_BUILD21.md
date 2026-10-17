# CutSell iOS — build 21 (2026-10-16): uploads that finish

## Why (build 19 report from Swanny's iPhone 15 Pro, iOS 26.6)
- The upload ran at ~1 MB/s (7 parts of 16 MiB). iOS's background upload service reports bytes in
  bursts, so the 30 s watch saw "no progress", marked the upload failed and CANCELLED parts that
  were still sending (NSURLError -999 right after progress 30 % and 40 % arrived).
- Preparation re-encoded a 1180x2556 HEVC screen recording to 1080p: 72 MB became 103 MB.

## What changed (app only)
| Area | Build 21 |
|---|---|
| Progress | Bytes per part (didSendBodyData) + `countOfBytesSent` polled every 2 s; repeated counts are not activity |
| Watch | 90 s with no byte sent; never cancels anything; shows "No data sent for 90 seconds…" + **Try again** (restarts only the parts in flight); disappears by itself when bytes move |
| Retries | Each part on its own, up to 3 times (fresh link each time); finished parts never re-sent |
| Parts | App asks for 8 MiB; current server ignores it and answers 16 MiB (the app always uses the server's size) |
| Background | All parts queued in the background URLSession (3 at a time), extra background time while handing them over; continues with the app closed / screen locked. Falls back to a normal session when iOS answers NSURLErrorUnknown (Simulator) |
| Preparation | H.264/HEVC under 4K → uploaded as is (no re-encode). Otherwise HEVC 1080p capped near 10 Mb/s, used only if smaller than the original. Logs `mb_before` / `mb_after` |

## Proof (Simulator, optimized build, live server + R2) — probe run 38991695590
- Network limited to ~1 MB/s (dummynet), 1080p H.264 113 MB, 75 s.
- `prepare_skipped codec=avc1` (original uploaded), 7 parts, upload 123 s (~0.9 MB/s).
- Left CutSell at 15 % for 25 s → back at 40 %+; no stall notice, % never went back, no error.
- Processed, "Review & edit" opened the editor. Frames/logs: `design-screenshots` branch, `b21/`.
- Simulator limitation: iOS's background service fails there (-1), so the fallback session was
  used; the real background service is only provable on the iPhone.

## Server change prepared, NOT deployed (needs owner permission)
Branch `feat/multipart-part-size-8mib` (commit 53b282d): optional `part_size` (5–64 MiB) on
`/v1/uploads/multipart/start` and optional `expires_in` (60–3600 s) on part presign. Absent =
exactly today's behavior. Rollback = redeploy the previous Render commit (db3f61d).
