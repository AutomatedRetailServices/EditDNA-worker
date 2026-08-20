# CutSell.ai — Current State

Product generation: **CutSell.ai 7**

This file is the operational checkpoint. Update it whenever the active benchmark, release gate, branch state or major implementation focus changes.

## Repository

- Repository: `AutomatedRetailServices/EditDNA-worker`
- Active branch: `cutsell/mobile-v1-clean`
- Active PR: `#25` (Draft, open, unmerged)
- Base: `main`
- Base SHA remains `2fb13e5aa228e8e525b942a9b49182032b797e61`
- PR #24 remains historical/reference backup and must stay untouched.

## Current focus

**Flow B / Clean Cut editorial-quality hardening.**

Human review is the quality gate. Workflow success alone is not an editorial pass.
Sales-funnel/storytelling work remains intentionally separate until Clean Cut reaches reliable real-video quality.

## Benchmark history that matters

### Benchmark #39 — bad human baseline

- 16/16 processed technically.
- Human review: editorial failure.
- Core failure class: repeated failed attempts surviving, fragmentary edits, over-cutting and weak Best Take behavior.

### Benchmark #48 — first large editorial improvement

- 16/16 processed.
- 0 execution failures / 0 provider failures.
- Hybrid availability: 21/22 = 95.45%.
- Human review:
  - 00 still wrong: repeated sonography + hereditary-cancer delivery.
  - 02 major improvement, almost ready to deliver.
  - 03 cut ending too early.
  - 04 correct.
  - 05 correct.

### Benchmark #49 — technically clean, editorially failed

Exact worker:
- source `499d1c59e11ea1abe550678ae334cd46573312d1`
- image digest `sha256:01dc103a7742054b525b4ff71e720d4af220ff8a12b1a0938ed858b0bff250c5`
- workflow run `32318014133` (visible run #51)
- Pod `o5flfm9ioegdt6`, cleanup PASS

Technical:
- 16/16
- 0 execution failures
- 0 provider failures
- Hybrid 21/22 = 95.45%

Human review:
- 00 still kept competing versions instead of one winner.
- 02 regressed and reintroduced malformed failed speech.
- 03 ended at `te protegen` and cut the complete idea.
- 04 stayed correct.
- 05 stayed correct.

### Benchmark #50 — meaningful recovery, but 00 not fully solved

Exact worker:
- source `e596002dfc550c9da6d8d10a9eb47b3363272f69`
- image digest `sha256:ba14a4344e000e3a2200a1c74da32569ff53f823dd74688182f25947f7cb3431`
- benchmark run `32324942712` (visible run #52)
- Pod `uoz7exmx8tdf11`, cleanup PASS

Technical:
- 16/16
- 0 execution failures
- 0 provider failures
- Hybrid 21/22 = 95.45%

Observed editorial result:
- 00 improved strongly in sonography retries, but hereditary-cancer close still duplicated.
- 02 recovered the #48-quality path and removed `I people It was very funny`.
- 03 preserved the complete ending through `te reparan la barrera`.
- 04 remained correct/clean.
- 05 remained correct/stable.

### Benchmark #51 — architecture experiment exposed a production integration bug

Exact worker under validation:
- source `65fe16519bbf72562805d5dac0333b993c43c179`
- image digest `sha256:4593f402deb56e9128b1d058efd24515c3b91ae0712eac84aff172bc4536d99f`
- benchmark run `32345888694` (visible run #53)
- Pod `goi51dpki0m153`
- Pod cleanup step PASS

Technical result:
- 16/16 processed
- 0 execution failures
- 0 provider failures
- report and preview artifacts uploaded

Key editorial finding:
- 02/03/04/05 preserved the expected good behavior.
- 00 still selected the complete hereditary-cancer delivery **plus** the split retry (`prefix` + `continuation`).

Production diagnostics proved why:
- the new final sibling reconciliation logic itself worked in unit/regression tests;
- however `safe_group_takes_by_sessions` partitions the video into mini-sessions first and calls grouping independently inside each session;
- in Video 00 the complete hereditary delivery, split retry prefix and continuation ended in three distinct session-scoped groups;
- therefore the sibling layer never saw the three candidates together in production.

This means Benchmark #51 did **not** disprove the sibling-family architecture. It exposed that the reconciliation was installed one level too low.

## Current architecture decision

Recover the useful earlier EditDNA invariant without rolling back the modern Clean Cut architecture:

**same audience-facing idea + competing deliveries -> one sibling/retry family -> Best Take chooses exactly one selected winner**

Important:
- losers are not destroyed;
- they remain available as alternates / Swap Take;
- Composer should receive only the selected winner from a retry family;
- Story/funnel logic is not part of this decision.

The newer Clean Cut capabilities remain:
- whole-video understanding;
- attempt reconstruction;
- session boundaries;
- Hybrid/Gemini semantic reasoning;
- story-preservation guards;
- temporal completion/boundary repair;
- immutable source identity.

## Fix implemented after Benchmark #51

A new **global post-session sibling bridge** now wraps the final output of `safe_group_takes_by_sessions`:

1. session-scoped grouping still runs normally to protect true compilation boundaries;
2. all final groups are then presented to conservative sibling reconciliation at the whole-source level;
3. only strong competing-delivery evidence can merge groups;
4. no take is deleted by this bridge;
5. the existing TakeJudge/Hybrid Best Take authority chooses the one selected winner.

A production-path regression now forces the three Video 00 hereditary takes into three separate mini-sessions and requires the final grouping output to collapse them into one sibling family.

## Current validated code checkpoint

Current code head:

`140bec542dd9f0b429093810e4b85b541e35f589`

Validation:
- **CutSell Clean Worker CI #1474 — PASS**
- **CutSell iOS CI #1255 — PASS**
- new cross-session production-path sibling regression — PASS

PR #25 remains Draft and unmerged. `main` remains unchanged.

## Next gate

Code-level validation is green, but the new post-session sibling fix has **not yet been proven on real RAW video**.

Before the next paid validation:
1. keep current head green;
2. build an immutable worker image from the exact validated source;
3. stop for explicit user approval before creating a new paid RunPod Pod / benchmark;
4. after approval, run exactly one controlled 16-video RAW benchmark;
5. verify deletion of the exact Pod;
6. present 00/02/03/04/05 for human review.

Success criteria:
- **00:** hereditary-cancer competing deliveries collapse to one winner; no duplicate sonography regression.
- **02:** keeps the #48/#50 improvement and no malformed retry fragment.
- **03:** retains the complete natural ending.
- **04:** stays correct.
- **05:** stays correct.

## Sales Funnel sequencing

Do **not** reintroduce rigid funnel structure while Clean Cut is still being validated.

After Clean Cut is reliable, resume the commercial layer as a flexible narrative-understanding stage that can recognize different selling structures (for example story -> discovery -> experience -> recommendation, pain -> demo -> proof, hook -> solution -> CTA) rather than forcing every video into one fixed funnel template.

## Current execution rule

Continue automatically after each non-paid fix/status block. Stop only for:
- explicit approval before new paid infrastructure/benchmark spend; or
- human review when new preview videos are ready.

## Update rule

Whenever work advances to a later benchmark or major Brain checkpoint, update this file in the same development cycle instead of reconstructing state from chat history.
