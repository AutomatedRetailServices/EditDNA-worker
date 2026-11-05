# CutSell — before the paid launch (Product Owner decisions, kept here so nothing is lost)

Last updated 5 Nov 2026.

## Decided by the Product Owner
- **Free trial: 3 days** (PO 5 Nov 2026). After the trial the person pays to keep cutting.
- **Testing phase now:** the app does NOT ask to sign in; it starts a guest account by itself. That stays while
  CutSell is in testing (PO 5 Nov 2026).
- **When CutSell is paid:** paying needs an account (Sign in with Apple), so every subscription belongs to one person.

## To build before the paid launch
1. **Payments (StoreKit 2)** with the 3-day free trial as an App Store introductory offer, and **server-side
   validation** (App Store Server API + App Store Server Notifications V2). The server gives processing minutes
   only to a valid trial or subscription — never because the phone says so.
2. **Guest accounts:** decide what a guest can do once payments exist (e.g. the trial only after signing in with
   Apple, or a very small free try). Today a guest has the same 300 min/month as an Apple account and guests can
   be created with little friction (security review 4 Nov, point 2).
3. **Account deletion:** delete everything (no 100-project cap; plan, catalog, Mix, renders, name, review videos)
   and **revoke Sign in with Apple** (Apple requirement; needs a Sign in with Apple .p8 key).
4. **Privacy text:** the policy and the AI consent screen must say that 1–2 video frames go to Claude for the
   product name (or that step is switched off).
5. Security review 4 Nov 2026: the remaining 🔴/🟠 items (secrets in GitHub Environments, Redis and AWS keys
   per component, repo private, 2FA, provider spend caps).

## Already done
- 5 Nov 2026: spend limits on every door that queues paid work (real video length checked before Deepgram /
  Claude; hourly ceilings; Bulk counted by the server) — commit 01db5be, live.
