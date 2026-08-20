# CutSell.ai — Current State

Product generation: **CutSell.ai 7**

This file is the operational checkpoint. Update it whenever the active benchmark, release gate, branch state or major implementation focus changes.

## Repository

- Repository: `AutomatedRetailServices/EditDNA-worker`
- Active branch: `cutsell/mobile-v1-clean`
- Active PR: `#25` (Draft)
- Base: `main`
- PR #24: historical/reference backup; do not modify as part of the clean release path.

## Current focus

**Brain editorial-quality hardening for Flow B / Clean Cut using the latest human-reviewed real-video evidence from Benchmark #49.**

Human review is the quality gate. Workflow success alone is not an editorial pass.

## Current Brain checkpoint

### Benchmark #39 — bad human baseline

- 16/16 processed technically.
- 0 execution/provider failures.
- Human review: editorial failure.
- Failure pattern: repeated failed attempts surviving, fragmentary edits, over-cutting, and cases where too little of the original delivery remained to judge naturally.
- Treat #39 as the original bad baseline that later benchmarks must visibly beat.

### Benchmark #48 — partial editorial improvement, not a pass

- 16/16 processed.
- 0 execution failures.
- 0 provider failures.
- Hybrid availability: **21/22 = 95.45%**.

Human review:

- **Video 00:** still wrong; repeated sonography lines and repeated/hereditary-cancer close survived.
- **Video 02:** major improvement, almost ready to deliver.
- **Video 03:** cut too early at the ending; idea/words incomplete.
- **Video 04:** correct.
- **Video 05:** correct.

### Benchmark #49 — technically clean, editorially failed

Exact controlled worker under validation:

- Source SHA: `499d1c59e11ea1abe550678ae334cd46573312d1`
- Immutable image digest: `sha256:01dc103a7742054b525b4ff71e720d4af220ff8a12b1a0938ed858b0bff250c5`
- Benchmark workflow run id: `32318014133` (visible workflow run #51)
- Exact staging Pod: `o5flfm9ioegdt6`
- Pod cleanup step completed successfully.

Technical result:

- 16/16 processed.
- 0 execution failures.
- 0 provider failures.
- Hybrid availability: **21/22 = 95.45%**.
- JSON report and preview-video artifacts uploaded successfully.

Human review: **NOT an editorial pass.**

- **Video 00:** still wrong. The Brain still keeps multiple versions of the same repeated idea instead of selecting the winning delivery. Core sonography repetition and hereditary-cancer repetition remain.
- **Video 02:** regressed from Benchmark #48. #49 reintroduced malformed failed speech (`I people It was very funny`) that #48 did not select.
- **Video 03:** regressed further; #49 ends at `te protegen`, cutting the complete delivery too early.
- **Video 04:** remained correct and improved slightly by removing the stranded `priceless.` fragment.
- **Video 05:** remained correct/stable; blooper debris stays removed while useful hidden-hunger story remains.

Do not use #49 technical success as evidence of editorial success.

## Root causes proven by #49 diagnostics

### Video 00

Two separate production-path failures were confirmed:

1. **Correct Hybrid deletion was later undone by story protection.** Hybrid correctly labelled some repeated/bad takes `failed` or `alternate`, but `hybrid_story_coverage_guard` restored them because lexical retry coverage did not prove a peer. This revived material the semantic judge had already rejected.
2. **Short semantic retries remained stranded across deterministic groups.** Short sonography retries were separately labelled `alternate`/`failed`, but because they were not in one deterministic retry group and local corroboration was insufficient, they remained selected.

Therefore the remaining problem is not simply Best Take ranking inside a retry group. The Brain needs a final cross-group semantic retry integrity pass after fail-open/story restoration.

### Video 02

The malformed fragment `I people It was very funny` was Hybrid-labelled `failed` at confidence **0.85**. The generic short-fragment guard required 0.86, so the bad fragment survived. Lowering the generic threshold to 0.85 proved unsafe because an existing regression requires ordinary 0.85 failed speech without structural corroboration to remain fail-open.

The final fix therefore uses additional structural evidence (a compact pronoun-collision restart) at 0.85 while preserving the generic 0.86 fail-open threshold.

### Video 03

The prior completion rollback logic could shorten a take already classified as `complete_idea=True`. That contradicts the editing doctrine. A completed delivery is now protected; a failed tail may be removed without amputating the complete preceding speech.

## Current code checkpoint

Current validated code head before this documentation-only update:

`8101891752545003d1461f59564ec4f0bc63848d`

Validation on that exact code head:

- **CutSell Clean Worker CI #1452 — PASS**
- **CutSell iOS CI #1233 — PASS**

PR #25 remains Draft and unmerged. `main` remains unchanged by this release path.

## Hardening now implemented after Benchmark #49

- final cross-group semantic retry integrity after Hybrid/story-coverage restoration;
- combined coverage from nearby authoritative winner/keep deliveries, including winner + immediate continuation;
- preservation of critical semantic tokens such as negations and numeric facts before removing a repeated delivery;
- cross-group collapse regressions for Video 00 sonography and hereditary-cancer repetitions;
- fail-open protections so Video 02 unique story and Video 05 hidden-hunger story are not deleted merely for topic similarity;
- targeted malformed failed-fragment cleanup for Video 02 at 0.85 when independent pronoun-collision structure proves a broken restart;
- preservation of the established generic 0.85 fail-open contract for ordinary speech;
- complete-idea protection for Video 03 so a failed tail cannot shorten already-complete speech;
- existing protections for Video 04 stranded alternates remain in place.

## Canonical document hierarchy

Read these as the current source of truth:

1. `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md`
2. `docs/CUTSELL_BRAIN_DOCTRINE.md`
3. `docs/CUTSELL_DECISIONS.md`
4. `docs/CUTSELL_STAGING_DEPLOYMENT_CONTRACT.md`
5. `docs/CUTSELL_STAGING_READINESS.md`
6. `docs/CUTSELL_CURRENT_STATE.md`
7. `AGENTS.md`

If two documents conflict, prefer the more recent explicit decision in `CUTSELL_DECISIONS.md`, then update the older document so the contradiction does not persist.

## What is already proven

The clean release path already has evidence for:

- CutSell API container build/boot and health;
- Clean Worker CI;
- iOS Simulator CI/build;
- real RunPod GPU smoke;
- immutable GPU worker image build/push;
- Render staging API;
- shared Redis/S3 staging topology;
- real staging E2E through public API: auth -> upload -> Flow B -> Draft recovery -> edit/autosave -> export -> download/validation;
- persisted Draft recovery;
- retry/cancel behavior;
- Render rollback;
- worker-image rollback and cleanup.

## Next gate

The current code-level defects exposed by #49 have targeted fixes and green CI, but **real-video editorial quality is not yet revalidated**.

Before another paid RunPod validation:

1. Keep the current code head/regressions green.
2. Build an immutable worker image from the exact validated source.
3. **Stop for explicit user approval before creating a new paid RunPod Pod / running another paid validation cycle.**
4. After approval, run exactly one controlled 16-video Gold RAW cycle.
5. Delete and verify deletion of the exact staging Pod.
6. Present Video 00/02/03/04/05 previews for human review.
7. Treat the cycle as successful only if:
   - 00 chooses one winning delivery rather than keeping repeated attempts;
   - 02 regains #48 quality and removes the malformed fragment without losing unique story;
   - 03 preserves a complete natural ending;
   - 04 remains correct;
   - 05 remains correct.

## Remaining path to closed TestFlight

1. Reach acceptable Brain quality on real-video review/regression.
2. Apple Developer signing / App ID / provisioning / TestFlight setup.
3. Physical iPhone validation: camera, Photos/iCloud import, background multipart resume, app kill/resume, Flow B, playback/timeline, Swap Take, trim/split, overlays, captions, recovery, export, Save to Photos and share.
4. Fix device-only blockers found by physical QA.
5. Build/upload closed TestFlight beta after approval.

## Current execution rule

Continue automatically after each non-paid fix/status block. Stop only for explicit approval before new spend/paid infrastructure, or when human review of new videos is required.

## Update rule

Whenever work advances to a later benchmark or major Brain checkpoint, update this file in the same development cycle instead of reconstructing state from chat history.
