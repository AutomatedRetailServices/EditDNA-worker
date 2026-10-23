# 22 Oct 2026 — uploads, export on the iPhone, prompt cache, storage

Branches: engine/server `feat/oct22-cache-storage` (on top of the live `fd2b3e5`), iOS `feat/ios-build24`.
Nothing here is deployed until the Product Owner says so. Rollback of server and engine: `fd2b3e5` (live on 22 Oct).

## What failed on 22 Oct (her Bulk of 5)
- IMG_9663 (6:39) and IMG_9665 (5:44): iCloud originals uploaded byte for byte; 1080p60 at ~25 Mb/s weighs over 1 GB,
  the server refused them at `multipart/start` (422, 1 GiB limit).
- IMG_9664 (5:22): the phone asked for all ~80 part links at once; the server signed them one by one (~0.65 s each:
  a new S3 client and a new Redis connection per link) and the last ones passed the phone's 20 s limit (499/502).
- IMG_9661 (0:03) and IMG_9664's voice: nobody talking to the camera (6 frames: mouth closed; a quiet voice in the
  background). Deepgram's auto-detect heard "Japanese" / "Slovak".

## Fixes
- iOS: heavy videos (> ~14 Mb/s) and heavy iCloud originals get a light copy before upload (reader/writer, HEVC,
  ≤ 1080p, 10 Mb/s; past ~12.5 min the bitrate drops so the copy stays under 950 MB). All modes, Bulk one by one, Mix too.
  Part links at most 4 at a time.
- Server: one S3 client and one Redis connection per process for part links (80 links: ~50 s → 0.03 s).
  The 4 GiB limit exists on `fix/upload-big-files` only (PO: not deployed).
- Engine: language detected other than es/en → heard again in es and en, more words wins (Gold: es/en, unchanged).
  `no_speech` when: no words; < 15 words/min in a clip over 20 s; or mean volume under −25 dB with < 15 words
  (background voice). The Gold (min 113 words a video) can never meet these rules (test pins it).
  The app shows "We couldn't hear anyone talking in this video" · NO VOICE · Use in Mix.

## Export on the iPhone (default)
- Settings sheet = Figma 950:8730 / 950:8790. 4K Ultra HD by default when the original is 4K; never upscaled (an option
  above the original is greyed out: "Your video was recorded in 1080p").
- Reads the ORIGINAL from Photos (`OriginalLinks`: Photos id per uploaded file / storage URI), downloading from iCloud
  when needed; original deleted → the light copy, and it says so.
- Same timeline, mix (volumes, ducking, Clean up, voices, sounds, overlay sound), framing (fit/fill, mirror, rotation,
  keyframes / Auto Zoom) and the editor's own overlay and caption views (SwiftUI `ImageRenderer`) as the preview;
  custom AVVideoCompositing; HEVC, BT.709.
- Simulator check (same cut, iPhone vs server export): same captions, framing and moments. The server additionally trims
  silent clip tails (0.37 s on the test cut) that the preview shows; the iPhone export follows the preview.
- Server export stays: "Export on our server (1080p)" in ⋯, after a failure, or when there's no space
  ("Not enough space on your iPhone. Free up X GB" + Open Settings). Plan my week: the night export (charging + WiFi)
  and Post to TikTok export on the iPhone; the server when a voice / sound / overlay file isn't on the phone.

## Prompt cache (engine)
- Every video (Single, Combine, Bulk, Mix; all users) caches each step's fixed instructions. TTL 1 h while fewer than
  12 videos were cut in the last hour (Redis count), else 5 min. `prompt_cache` per source records tokens, cost with and
  without the cache, saved USD. Prices: Sonnet 5.5 $2 in / $10 out / $2.50 5-min write / $4 1-h write / $0.10 read per MTok.
- The cut's decisions and prompts are unchanged (the Gold guard pins engine.py + prompts; transport/billing may change).

## Storage
- Old Amazon bucket: test inputs (15.74 GB) and her uploads (2.44 GB, 294 files) copied to R2 under the same keys and
  verified (sizes); 823 MP4s of old test runs (31.46 GB) deleted, their .json kept. S3: 49.9 → 18.5 GB.
- Reads of an old-bucket address go to R2 first (same key), Amazon only when R2 lacks it (30-day backup).
- R2 rule (PO): the light copy and the voice are removed 3 days after the last use; opening a cut resets the count.
  `storage_lifecycle.candidates` only reports; deletion needs CUTSELL_STORAGE_LIFECYCLE_DELETE=1 and confirm=True.
  A removed copy is rebuilt from the Photos original and uploaded to the same key ("Restoring video…").

## Tests
- 19 failing tests on `fd2b3e5`: 2 needed sqlalchemy locally (pass); 10 need mediapipe 0.10.21's legacy API (skip
  with the reason when it's missing); 7 legacy-stack tests failing since the 28 Sep restore (xfail with the reason;
  users are cut by the simple engine). None deleted.

## 23 Oct — Bulk and Mix through the Batch API
- Bulk videos and Mix send their Claude calls through the Message Batches API (50 % of every price, the cache prices
  too; same request body). Each video waits at most 30 min in all; past it the pending batch is cancelled and the
  normal API answers at once. Single and Combine unchanged. `CUTSELL_ENGINE_BATCH=0` turns it off (no redeploy).
- Test (QA bloopers): 15 videos by batch, 0 fallbacks; each call ~1.5–3.5 min in the batch, a video 7–17 min in all
  (normal: 17–39 s). Cost 0.46 USD vs 0.92 USD at normal prices (−50 %).
- Same cut: the request is identical; run-to-run variation of the model is the same by batch as by the normal API
  (two normal runs of the same 5 videos differed as much as normal vs batch).
- App: a video processing for over 8 min says "Taking a little longer, we'll notify you".

## 23 Oct (after the deploy) — Bulk vs the "2 cuts at once" limit
- Each user may have 2 cuts processing at once (`CUTSELL_MAX_CONCURRENT_JOBS_PER_USER`, unchanged since before). With the
  Batch API a Bulk video now holds its slot 10–17 min, so videos 3–5 of a Bulk would be refused (the app retries 3 times
  in seconds). Found by the production check (an unhandled 500); no real user was affected.
- Fix (`feat/bulk-batch-api`, not deployed yet): a Bulk send gets one slot per video (`bulk_size`, max 10); Single and
  Combine keep the limit of 2. The limit answers 429 `processing_concurrency_limit` (Retry-After 60) instead of a 500.

## 23 Oct — why the same video gets different cuts (diagnosis, no code change)
- Where the runs split: the pass step. The engine goes MULTI-PASS (keeps only one "backbone" pass) when the model marks
  2+ passes as complete. In the bloopers each "pass" is a different product, so one wobbly yes/no (is the 2nd pitch
  complete?) flips the cut between "everything cleaned" (56–92 s) and "one pitch" (11–43 s).
- Same words, 5 runs each: Bloppers5 58–71 s ×4 and 11 s ×1; Bloppers2 88 s ×1 and 43 s ×4; Bloppers3 35 s ×5.
  Pass step at full thinking (instead of medium): Bloppers5 56–62 s ×5 (stable); Bloppers2 still 43 s ×4, 92 s ×1.
- The two "complete passes" of Bloppers2 share only 15–20 % of their words (lip gloss vs perfume): different pitches,
  not retakes.
- Proposed (needs a Gold check before turning on): (1) pass step back to full thinking; (2) MULTI-PASS only when 2+
  complete passes are retakes of the same pitch (word overlap above a threshold measured on the Gold), else SINGLE PASS.
