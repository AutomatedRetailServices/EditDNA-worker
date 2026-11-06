# CutSell — before the paid launch (Product Owner decisions, kept here so nothing is lost)

Last updated 6 Nov 2026.

## Decided by the Product Owner
- **Free trial: 3 days** (PO 5 Nov 2026). After the trial the person pays to keep cutting.
- **Testing phase now:** the app does NOT ask to sign in; it starts a guest account by itself. That stays while
  CutSell is in testing (PO 5 Nov 2026).
- **When CutSell is paid:** paying needs an account (Sign in with Apple), so every subscription belongs to one person.
- **Account deletion = everything, nothing recoverable** (PO 6 Nov 2026).
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
- 6 Nov 2026 (server LIVE at 6e2a2b9; app part on build/privacy-create = build 73, pending): product AI skipped when
  the product is picked, photo step off; complete account deletion (every project, every record, every storage
  drawer in both buckets, AI answers; "partial" never reported as deleted); AI answers remembered 30 days per
  creator (same video, same cut, no repeat cost); privacy policy and AI notice say words only; soft Create bubbles.
- 5 Nov 2026: spend limits on every door that queues paid work (real video length checked before Deepgram /
  Claude; hourly ceilings; Bulk counted by the server) — commit 01db5be, live.
