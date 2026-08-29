# CLAUDE.md — CutSell.ai Operating Contract

You are continuing an existing production-oriented app. Do NOT redesign CutSell and do NOT start from scratch.

## Mandatory startup
Before editing, inspect live Git state:

```bash
git status
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

Expected handoff context unless newer intentional commits exist:
- repo: `AutomatedRetailServices/EditDNA-worker`
- branch: `cutsell/mobile-v1-clean`
- PR #25: OPEN / DRAFT / UNMERGED
- main base SHA: `2fb13e5aa228e8e525b942a9b49182032b797e61`
- handoff head before Claude docs: `9fad0788e120b2af07576f2d145fc4c179b24adb`

If HEAD is newer, inspect intervening commits and reconcile state. Never reset blindly.

## Read order
1. `docs/claude-handoff/CUTSELL_COMPLETE_HANDOFF.md`
2. `docs/claude-handoff/SECURITY_CONSTITUTION.md`
3. existing `AGENTS.md`
4. existing `docs/CUTSELL_DECISIONS.md`
5. existing `docs/CUTSELL_BRAIN_DOCTRINE.md`
6. existing `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md`

## Source precedence
When sources conflict:
1. live code/live Git state;
2. current Claude handoff;
3. current scope/decision contract;
4. canonical repo doctrine;
5. historical checkpoint;
6. old conversations.

## Current mission
**Flow B → Universal Clean Cut → Unified Whole-Video Selection.**

Preserve:
- one final semantic Selection authority;
- Selection Freeze;
- Boundary-only physical timing after freeze;
- SELECT/SWAP/DISCARD semantics;
- Human Gold QA;
- unseen generalization.

## Editorial rules
- WHEN UNCERTAIN, KEEP.
- Preserve unique audience-facing information.
- Preserve story/personality.
- Remove real failed/retry/BTS material.
- Never invent speech.
- Never hardcode Video00 timestamps, phrases or clip IDs.
- Do not force rigid sales-funnel logic during Clean Cut.
- Human performance errors matter even when transcript is complete.

## Engineering rules
- Diagnose from latest run/artifact before editing.
- Prefer structural root-cause fixes.
- Add targeted tests for behavior changes.
- Run tests before CI.
- CI success is not editorial success.
- Do not run overlapping paid RAW benchmarks.
- Always verify RunPod teardown.
- Preserve observability; never accept silent provider fallback.

## Security rules
`docs/claude-handoff/SECURITY_CONSTITUTION.md` is binding.
Security runs in parallel with editorial QA; do not postpone it to launch.

## Repository protection
Without explicit user approval:
- do not merge PR #25;
- do not write to `main`;
- do not close PR #25;
- do not deploy production;
- do not perform TestFlight/App Store release;
- do not make destructive repository/archive changes;
- do not expose/move secrets;
- do not create materially new recurring paid infrastructure.

## Current first task
Read the `CURRENT LIVE BLOCKER` and `EXACT NEXT ACTION` sections of the complete handoff.
Short version: fix Unified Selection benchmark/serverless observability before any new semantic Video00 rule.

## QA loop
`diagnose → fix → targeted tests → CI → one RAW → JSON+MP4 → architecture check → Selection analysis → Human Gold → Watch+Listen → unseen/regression`

## Status vocabulary
Report exact state instead of generic "done":
- CODE FIXED
- TESTS PASS
- CI GREEN
- RAW COMPLETE
- ARCHITECTURE PASS/FAIL
- GOLD CANDIDATE
- HUMAN WATCH+LISTEN PASS
- REGRESSION PASS
- SECURITY REVIEWED / SECURITY SCAN PASS when applicable

## Documentation duty
After material checkpoints, update durable current-state/decision docs. Never make the user reconstruct CutSell from chat again.
