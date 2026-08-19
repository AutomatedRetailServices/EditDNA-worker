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

**Brain editorial-quality hardening for Flow B / Clean Cut before the next paid real-video benchmark.**

Human review is the quality gate. Workflow success alone is not an editorial pass.

## Current Brain checkpoint

### Human baseline / failure reference

- Benchmark #39 remains the key human-reviewed bad baseline.
- User review found repeated failed attempts surviving, fragmentary edits, over-cutting, and cases where too little of the original delivery remained to judge naturally.
- Do not treat later technical workflow success as improvement unless the rendered videos are visibly better than this baseline.

### Subsequent benchmark evidence

- Benchmark #42 exposed concrete architecture/runtime defects including a `SessionBoundary` compatibility crash, Hybrid payload-budget failures, zero-selected drafts, and post-Best-Take clip identity loss.
- Benchmarks #44/#45 completed technically, but human feedback still judged the videos as effectively as bad as Benchmark #39. Therefore they are **NOT editorial passes**.
- No later benchmark may be declared a quality pass without human review of the rendered previews.

### Current code checkpoint

- Current branch head at this checkpoint: `1e56b3c984ed6b3fa9d18425415801c19b665b01`.
- Clean Worker CI: green at run #1432.
- iOS CI: green at run #1213.
- PR #25 remains Draft and unmerged.
- `main` remains unchanged by this release path.

### Hardening completed after the bad-video reviews

The branch now contains regression protection for the major failure classes discovered in the real-video runs:

- complete delivery-attempt reconstruction before destructive editorial decisions;
- conservative attempt-boundary integrity;
- retry-family reconciliation for false starts, weak retries and reformulated retries;
- session-scoped grouping bound to the installed production retry reconciler;
- Best Take identity preservation after winner selection;
- prevention of weak/restart fragments leaking into selected output alongside a fuller delivery;
- Hybrid request payload compaction so large candidate windows are not rejected before Gemini;
- story-coverage guards that preserve long coherent delivery when short fragments would otherwise veto it;
- semantic Best Take integrity when Hybrid clearly marks the local winner as failed and one clear usable peer exists;
- Hybrid alternate integrity to suppress stranded alternate prefixes/suffixes that are superseded by a complete final winner while failing open on unique material.

## Canonical document hierarchy

Read these as the current source of truth:

1. `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md` — product/V1 scope and delivery order.
2. `docs/CUTSELL_BRAIN_DOCTRINE.md` — authoritative editing behavior.
3. `docs/CUTSELL_DECISIONS.md` — canonical/superseded/rejected product decisions.
4. `docs/CUTSELL_STAGING_DEPLOYMENT_CONTRACT.md` — infrastructure/deployment contract and approval gates.
5. `docs/CUTSELL_STAGING_READINESS.md` — proven staging evidence and remaining beta gates.
6. `docs/CUTSELL_CURRENT_STATE.md` — this live operational checkpoint.
7. `AGENTS.md` — instructions for Codex/developer agents.

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

## Product/Brain status

### Implemented or substantially present

- Flow B architecture;
- source identity;
- ASR;
- Clean Cut;
- take segmentation/grouping;
- retry handling;
- Best Take / Take Judge paths;
- whole-video and visual analysis paths;
- hybrid editorial/composer paths;
- editable Draft storage/edits;
- projects;
- multipart/resumable upload infrastructure;
- notifications;
- captions/text/overlays/audio edit paths;
- render/export/versioning;
- iOS client/editor code;
- benchmark/evaluation workflows.

### Current hardening area

- postable editorial quality on real messy videos;
- human-performance bad-take recognition;
- retry grouping accuracy;
- Best Take correctness;
- precision trimming/boundaries;
- preserving personality and meaningful pauses;
- global story/sales continuity;
- avoiding redundant or locally-good-but-globally-wrong selections;
- preventing Hybrid alternates or failed local winners from surviving into the selected draft incorrectly.

## Next gate

Before another paid RunPod benchmark:

1. Keep Clean Worker and iOS CI green on the current hardening head.
2. Confirm no remaining known unit/regression path can leak failed/retry fragments into selected output.
3. Build an immutable worker image from the exact current head.
4. **Stop for explicit user approval before launching paid RunPod.**
5. After approval, run exactly one controlled 16-video validation cycle.
6. Delete and verify deletion of the staging Pod.
7. Present the new preview videos for human editorial review and stop for user input before any further paid benchmark.

## Remaining path to closed TestFlight

1. Reach acceptable Brain quality on real-video review/regression.
2. Apple Developer signing / App ID / provisioning / TestFlight setup.
3. Physical iPhone validation:
   - camera;
   - Photos/iCloud import;
   - background multipart resume;
   - app kill/resume;
   - Flow B;
   - playback/timeline;
   - Swap Take;
   - trim/split;
   - overlays;
   - captions;
   - recovery;
   - export;
   - Save to Photos;
   - share.
4. Fix device-only blockers found by physical QA.
5. Build/upload closed TestFlight beta after approval.

## Current execution rule

Do not start a new paid benchmark merely because the previous workflow completed. First inspect the current evidence and identify a justified root-cause change. New paid GPU/provider runs require explicit approval.

## Update rule

Whenever work advances to a later benchmark or major Brain checkpoint, update this file in the same development cycle instead of reconstructing state from chat history.
