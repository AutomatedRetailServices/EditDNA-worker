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

**Brain editorial-quality hardening for Flow B / Clean Cut using the latest human-reviewed real-video evidence from Benchmark #48.**

Human review is the quality gate. Workflow success alone is not an editorial pass.

## Current Brain checkpoint

### Benchmark #39 — bad human baseline

- 16/16 processed technically.
- 0 execution/provider failures.
- Preview artifacts were generated and the paid staging RunPod was deleted.
- Human review: editorial failure.
- Failure pattern: repeated failed attempts surviving, fragmentary edits, over-cutting, and cases where too little of the original delivery remained to judge naturally.
- Treat #39 as the bad baseline that later benchmarks must visibly beat.

### Benchmarks #42 / #44 / #45 — architecture/runtime diagnosis

- #42 exposed concrete defects including `SessionBoundary` compatibility failure, Hybrid payload-budget failures, zero-selected drafts and post-Best-Take clip identity loss.
- #44/#45 completed technically, but human feedback still judged the rendered videos effectively as bad as #39.
- These are NOT editorial passes.

### Benchmark #48 — latest human-reviewed benchmark

- 16/16 processed.
- 0 execution failures.
- 0 provider failures.
- Hybrid availability: **21/22 = 95.45%**.
- This is materially better technically than #42/#44/#45, but quality remains partially unresolved and is NOT yet an editorial pass.

Human review of #48 currently recorded:

- **Video 00:** still wrong. Must remove/resolve repeated sonography lines and the repeated/hereditary-cancer closing. Root issue: semantic Best Take can only choose among candidates already recognized as the same retry family; this case still escapes grouping/relationship recognition.
- **Video 02:** almost ready to deliver.
- **Video 03:** still wrong at the ending. It must preserve the completed idea/words and avoid cutting off the end of the delivery. Requires completion-preserving boundary refinement.
- **Video 04:** correct.
- **Video 05:** correct.

Do not generalize #48 as “fixed” from aggregate metrics. The remaining work is driven by the specific human-reviewed failures above.

### Current code checkpoint

- Current branch head before this documentation reconciliation: `7a7c8d26475e6dfed36dd4362504fc88321df00e`.
- PR #25 remains Draft and unmerged.
- `main` remains unchanged by this release path.
- Latest Clean Worker CI run #1440 and iOS CI run #1221 show `action_required` with zero jobs executed; this is not a test failure and must not be described as broken code.

### Hardening completed after bad-video reviews

The branch contains regression protection for the major failure classes discovered in real-video runs:

- complete delivery-attempt reconstruction before destructive editorial decisions;
- conservative attempt-boundary integrity;
- retry-family reconciliation for false starts, weak retries and reformulated retries;
- session-scoped grouping bound to the installed production retry reconciler;
- Best Take identity preservation after winner selection;
- prevention of weak/restart fragments leaking into selected output alongside a fuller delivery;
- Hybrid request payload compaction so large candidate windows are not rejected before Gemini;
- story-coverage guards that preserve long coherent delivery when short fragments would otherwise veto it;
- semantic Best Take integrity when Hybrid clearly marks the local winner as failed and one clear usable peer exists;
- Hybrid alternate integrity to suppress stranded alternate prefixes/suffixes superseded by a complete final winner while failing open on unique material;
- Benchmark #47/#48 regressions for failed-local-vs-clean-peer selection and stranded Hybrid alternate cleanup.

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

## Current hardening area

Prioritize only remaining evidence-backed editorial defects:

1. **Video 00 class:** repeated/reformulated delivery that is semantically the same attempt but not being grouped as one retry family, including sonography/hereditary-cancer repetition.
2. **Video 03 class:** completion-preserving final boundary refinement so completed words/ideas are not clipped at the end.
3. Preserve the gains already seen in Videos 02/04/05; do not make cleanup more aggressive globally to solve 00/03.
4. Continue to fail open on unique valid story material and preserve personality/meaningful pauses.

## Next gate

- Continue code/regression hardening without paid infrastructure.
- Do not launch another paid RunPod benchmark merely because CI/workflows complete.
- A new paid GPU/provider validation cycle requires explicit user approval.
- When a new benchmark is justified and approved, run exactly one controlled validation cycle, delete/verify deletion of the staging Pod, then present previews for human review and stop for user input before any further paid benchmark.

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
