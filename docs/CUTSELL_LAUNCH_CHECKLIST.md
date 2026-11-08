# CutSell — before the paid launch (Product Owner decisions, kept here so nothing is lost)

Last updated 8 Nov 2026.

## Decided by the Product Owner
- **Free trial: 3 days** (PO 5 Nov 2026). After the trial the person pays to keep cutting.
- **Sign in with Apple for everyone, starting in testing** (PO 6 Nov 2026, replaces the 5 Nov "guest in testing"
  decision). Build 74: first open shows "Continue with Apple"; saved guest sessions keep working and move their
  videos into the Apple account on sign-in. Server: CUTSELL_APPLE_CLIENT_ID=ai.cutsell.app is set on Render.
  7 Nov: CUTSELL_GUEST_SIGNUP=0 is set (new guest accounts refused, 403). Build 76: a phone still on a guest
  account must sign in with Apple once after updating; its videos move to the Apple account. Older builds keep
  their guest session until they update. iPhone probes use /v1/auth/probe-session (GitHub OIDC).
- **When CutSell is paid:** paying needs an account (Sign in with Apple), so every subscription belongs to one person.
- **Account deletion = everything, nothing recoverable** (PO 6 Nov 2026).
- **Subscription ends = everything deleted, nothing recoverable** (PO 8 Nov 2026). Following Apple, a canceled
  subscription keeps working until the end of the period already paid (or the end of the 3-day trial); on that date
  the full account deletion runs. Before it, the app shows the dated warning (Create + You). Paying again before the
  date cancels it. Built in `cutsell_worker/subscription_lifecycle.py` (daily run inside the Modal storage_cleanup);
  dormant until payments exist: the StoreKit / App Store Server Notifications handler only calls `record(...)`.
- **Product AI:** when the creator picks brand + product in the app, the AI is never asked; when she picks nothing,
  the AI guesses from the words only — no photos (PO 6 Nov 2026).

## To build before the paid launch
1. **Payments (StoreKit 2)** with the 3-day free trial as an App Store introductory offer, and **server-side
   validation** (App Store Server API + App Store Server Notifications V2). The server gives processing minutes
   only to a valid trial or subscription — never because the phone says so.
2. **Guest accounts:** decide what a guest can do once payments exist (e.g. the trial only after signing in with
   Apple, or a very small free try). Today a guest has the same 300 min/month as an Apple account and guests can
   be created with little friction (security review 4 Nov, point 2).
3. **Sign in with Apple revoke:** the code is written (6 Nov) and waits for a Sign in with Apple key: the PO creates
   it in the Apple Developer account and sets CUTSELL_APPLE_TEAM_ID, CUTSELL_APPLE_KEY_ID, CUTSELL_APPLE_PRIVATE_KEY
   on Render. Until then deletion skips only that step ("not_configured").
4. **"Nothing recoverable" checks only the PO can do:** old Amazon S3 bucket has Versioning OFF; Render database
   backups expire within 30 days (the privacy policy says so).
5. Security review 4 Nov 2026: the remaining 🔴/🟠 items (secrets in GitHub Environments, Redis and AWS keys
   per component, repo private, 2FA, provider spend caps).

## Already done
- 7 Nov 2026 (live at a674c64): security 8 (oversized multipart upload deleted and refused), 9 (monthly minutes
  meter fails closed; counted in Redis because Render has NO database), 13 (API + worker packages pinned to the
  versions in production), 14 (Modal deploy tools/actions pinned; App Store key only in its 2 steps — ships with
  the next TestFlight build from build/apple-signup); probe door /v1/auth/probe-session (GitHub OIDC of this repo,
  3-hour non-sliding session) so CUTSELL_GUEST_SIGNUP=0 no longer breaks the iPhone probes.
  Pending PO: delete the 20 Video00 test-video artifacts + 3 Yaskira frame artifacts on GitHub (public repo).
- 6 Nov 2026 (server LIVE at 6e2a2b9; app = build 73 on TestFlight, also soft Record/Upload): product AI skipped when
  the product is picked, photo step off; complete account deletion (every project, every record, every storage
  drawer in both buckets, AI answers; "partial" never reported as deleted); AI answers remembered 30 days per
  creator (same video, same cut, no repeat cost); privacy policy and AI notice say words only; soft Create bubbles.
- 5 Nov 2026: spend limits on every door that queues paid work (real video length checked before Deepgram /
  Claude; hourly ceilings; Bulk counted by the server) — commit 01db5be, live.


## Security review (4 Nov 2026) — PAUSED by the PO on 8 Nov 2026; resume only when she says so
Done (12 of 27): 1 spend limits · 2 guest accounts closed (Sign in with Apple) · 3 product AI words only ·
4 full account deletion (Apple revoke waits for her Sign in with Apple key) · 8 oversized uploads refused ·
9 monthly meter fails closed (Redis; Render has no database) · 11 daily cleanup of iPhone copies after 24 h
(REPORT ONLY until CUTSELL_STORAGE_LIFECYCLE_DELETE=1 in the Modal secret, after her OK) · 13 packages pinned ·
14 App Store key only in its steps, deploy tools pinned · 15 test tokens masked, probe sessions 3 h.
Remaining, in the order proposed to the PO:
- 🔴 5 GitHub secrets into Environments (guide for the PO, ~15 min).
- 🟠 6 + 7 one key per component for Redis and AWS/R2, then rotate (code by Claude, new keys by the PO).
- 🟠 10 move timeline prepare/export off the API server · 12 payments (StoreKit 2, at paid launch).
- 🟡 16 kill switch · 17 tell the creator when a job dies · 18 health alerts · 19 never process/charge the same
  video twice · 20 rate limits on other routes · 21 check uploads are real videos · 22 + 27 remove unused keys and
  the old EditDNA service · 23 + 24 pin ffmpeg and the TikTok SDK · 25 privacy policy (remove DRAFT, add TikTok
  and Sentry, professional contact) · 26 push notifications route missing on the server.
- PO accounts: 2FA everywhere · provider spend caps · Sign in with Apple key · R2 not public / S3 versioning off.
- PO decisions: reinstall / new iPhone loses the "which Photos video" link (fix with a server copy, or accept) ·
  turn on deletion of the 24 h copies after the report and a real "Restoring video…" test · old GitHub artifacts
  (20 Video00 test videos + 3 Yaskira frame sets) left to expire on their own.
- Proposed next batch (no cost, nothing visible): 10, 16, 17, 19, 21, 26.
- App changes waiting for the next build (branch build/apple-signup, after build 76): dated deletion warning for
  a canceled subscription, privacy text on subscription end, Week Plan reminders removed when the account is gone,
  probe via GitHub identity, App Store key only in its steps, TestFlight check tools in their own environment.
- Known (not security): the Monday-Sunday Week Plan mode can come out empty on a Sunday; check whether the app
  still uses that mode.
