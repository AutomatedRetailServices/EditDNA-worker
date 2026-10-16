# CutSell state, 2026-10-16

Supersedes CutSell_estado_2026-10-08.md for storage, the AI services in use and build 17. The two documents the Product Owner meant to attach never arrived, so this file was written from the repository and the live checks.

## Live now

| Part | Version |
|---|---|
| Modal worker | server commit db3f61d (branch `feat/simple-engine-v1`) |
| Render API `srv-d9rdlsvavr4c73912uo0` | db3f61d, deploy `dep-db965nek1f9s73dn2iq0`, live 17:47 UTC |
| iPhone app | build 17 on TestFlight (VALID; app commit 64e107b) |
| App branch | `feat/figma-part1` (probe fix 88da209, which affects the DEBUG probe only) |

There is no merge to `main`.

## Storage: Cloudflare R2

- New files go to R2, bucket `cutsell-videos` (`AWS_ENDPOINT_URL_S3`, `AWS_REGION=auto`).
- The old Amazon S3 bucket (`FALLBACK_*`) is read-only, so old cuts still open.
- The only exception: Delete Account also deletes that user's files in the old bucket (commit 76f1859).
- db3f61d: the storage client signs links with V4 in the configured region. Without a region, boto3 signed GET links with V2, and R2 answered 401. This showed up live on the Clean up link.

## AI services really called

- Deepgram (speech to text, on the audio).
- Anthropic (the text of the transcript).

Gemini and OpenAI are off. The API calls no AI service. The AI notice, the Privacy Policy and the Terms name only these two, plus Cloudflare R2 for storage. The contact address is arsgroup01@gmail.com.

## Live test on R2 (2026-10-16, server db3f61d)

| Step | Time | Result |
|---|---|---|
| Upload | 8.2 s | to r2.cloudflarestorage.com |
| Process | 36.8 s | 2 AI zooms |
| Edit | 9.7 s | voice-over, photo overlay, Good + Typing |
| Clean up voice | 16.6 s | progress seen: 0 / 10 / 35 %. Pause noise −41.7 → −45.1 dB (a quiet test video). The voice level is kept. Offline, on a noisy clip, noise drops by more than 12 dB. |
| Export | 32.3 s | link on R2 |
| Download | 0.8 s | 1080x1920, 14.6 s, with audio |

All 6 CutSell sounds were downloaded from R2.

**Still pending:** opening a cut made before the move to R2 (S3 fallback). Only the Product Owner has old accounts, so this check is theirs.

## Build 17 on the Simulator, against the live server

Probe run 38942885541, frames in `design-screenshots/b17/`. Real touches checked:

- **Sound (Typing):**
  - Added at the cursor, and it comes selected.
  - Split at 1 s makes two pieces; one piece deleted.
  - Held and moved from 1.00 to 3.27 s.
  - End trimmed from 2.00 to 1.31 s.
- **Volume / Mute:**
  - With a clip selected, the panel says "Selected clip". Mute silences that clip only and shows the mute icon.
  - With nothing selected, it says "Whole video".
- **Clean up voice:**
  - On shows "Cleaning… %", then ready, with the clean copy playing.
  - Off goes back to the original.
- **Keyframes:**
  - Auto Zoom AI shows 8 ◇.
  - Manual ◇ at 1 s and 2 s (zoom 2.55). The button is filled on a ◇, and tapping it deletes that ◇.
- **Export:** OK in 40 s.

Human Watch + Listen on a real iPhone is still pending (Product Owner).

Details: `docs/CUTSELL_BUILD17.md`.
