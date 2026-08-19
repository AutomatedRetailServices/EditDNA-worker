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

**Brain editorial-quality hardening for Flow B / Clean Cut, then physical iPhone beta validation and TestFlight preparation.**

## Current Brain checkpoint

### Benchmark #38

- Workflow run: `32178648887`
- Branch: `cutsell/mobile-v1-clean`
- Execution status: **SUCCESS**
- Benchmark job: completed successfully.
- RunPod enqueue/wait: completed successfully.
- JSON report: generated/uploaded.
- Preview videos: generated/uploaded.
- Exact staging GPU worker cleanup: completed successfully.
- Editing-quality status: **NOT YET DECLARED PASS solely from workflow success**.
- Required next interpretation: rendered outputs must satisfy human/postable quality review under `CUTSELL_BRAIN_DOCTRINE.md`.

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

- CutSell API container build/boot and health.
- Clean Worker CI.
- iOS Simulator CI/build.
- real RunPod GPU smoke.
- immutable GPU worker image build/push.
- Render staging API.
- shared Redis/S3 staging topology.
- real staging E2E through public API: auth -> upload -> Flow B -> Draft recovery -> edit/autosave -> export -> download/validation.
- persisted Draft recovery.
- retry/cancel behavior.
- Render rollback.
- worker-image rollback and cleanup.

## Product/Brain status

### Implemented or substantially present

- Flow B architecture.
- source identity.
- ASR.
- Clean Cut.
- take segmentation/grouping.
- retry handling.
- Best Take / Take Judge paths.
- whole-video and visual analysis paths.
- hybrid editorial/composer paths.
- editable Draft storage/edits.
- projects.
- multipart/resumable upload infrastructure.
- notifications.
- captions/text/overlays/audio edit paths.
- render/export/versioning.
- iOS client/editor code.
- benchmark/evaluation workflows.

### Current hardening area

- postable editorial quality on real messy videos;
- human-performance bad-take recognition;
- retry grouping accuracy;
- Best Take correctness;
- precision trimming/boundaries;
- preserving personality and meaningful pauses;
- global story/sales continuity;
- avoiding redundant or locally-good-but-globally-wrong selections.

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

Whenever work advances from CutSell.ai 7 to a later project checkpoint or Benchmark #39/#40/etc., update this file in the same development cycle instead of reconstructing state from chat history.