# CutSell.ai — Master Pipeline QA / Release Gates

This document turns the approved Mobile V1 scope into system-level QA gates. It is intentionally end-to-end: fixes are not considered complete unless the relevant layer and the adjacent handoff both pass.

## Product contract

Mobile-first Flow B ships first. Required experience: Create/Import -> reliable upload -> background processing -> Watch+Listen Clean Cut -> Take Groups -> Best Take -> editable Draft Timeline -> manual edits/autosave/recovery -> export -> project persistence. Flow A remains Phase 2 on the same engine/editor infrastructure.

## Gate 0 — Repository / release safety

PASS requires:
- work remains on `cutsell/mobile-v1-clean` until human approval;
- PR #25 remains Draft/Open/Unmerged during validation;
- `main` remains untouched;
- Clean Worker CI green;
- iOS CI green;
- no release is called ready from unit tests alone.

## Gate 1 — iOS application shell

PASS requires:
- app boots without local-only configuration;
- production/staging API base URL is valid HTTPS;
- session recovery from Keychain works;
- Create, Projects/Cuts, Batch, Account/You are reachable;
- project list survives restart;
- errors have retry/recovery paths, not frozen states.

## Gate 2 — Auth / ownership / account lifecycle

PASS requires:
- anonymous closed-beta session OR Sign in with Apple according to build flag;
- bearer token enforced server-side;
- user cannot read/write another user's projects/media/jobs;
- logout/session clear works;
- project deletion cleans project state/media;
- account deletion revokes sessions and known account/project state.

## Gate 3 — Media intake

PASS requires:
- in-app camera presets 60s / 3m / 10m;
- single gallery import;
- multi-clip import up to 10 clips;
- source identity and user-selected ordering preserved;
- Photos/iCloud materialization exposes real progress;
- prepared items survive one slow/failed item;
- no unnecessary transcode when source is compatible.

## Gate 4 — Resumable/background upload

PASS requires:
- direct S3 presign/multipart upload;
- retry/resume interrupted parts;
- large imports can continue/handoff in background where iOS permits;
- project can reopen and resume upload without rebuilding;
- uploaded object ownership/prefix validation is enforced;
- upload states are truthful: retrieving -> preparing -> uploading -> uploaded -> failed/retry.

## Gate 5 — API / queue handoff

PASS requires:
- API health verifies queue and storage readiness;
- Flow B submission returns one durable job id;
- Redis/RQ queue receives job exactly once;
- concurrency slots are reserved/released correctly on success/failure;
- cancel/retry/status endpoints agree on job state;
- no orphaned paid worker or orphaned reserved slot.

## Gate 6 — GPU worker infrastructure

PASS requires:
- one known-good immutable worker image can be pulled and started;
- GPU runtime becomes ready on the selected provider;
- RQ worker registers and heartbeats;
- worker can reach Redis, S3 and approved semantic provider;
- automatic cleanup after benchmark/failure;
- provider failure has an explicit fallback strategy, not repeated blind provisioning.

This gate is currently a launch blocker until a stable GPU execution path is restored.

## Gate 7 — Flow B perception / Clean Cut

PASS requires:
- English and Spanish ASR with word timestamps;
- full-source Watch+Listen evidence available before destructive decisions;
- source/take boundaries preserved;
- no cross-source merges;
- no invented/Frankenstein speech;
- no cuts inside words;
- dead air, obvious false starts, fumbles, explicit retakes/BTS and incomplete production fragments are removed;
- uncertain valid speech is kept.

## Gate 8 — Take Groups / Best Take / commercial meaning

PASS requires:
- semantically equivalent retries are grouped;
- one selected take + ranked valid alternatives;
- valid alternatives are recoverable in editor;
- Best Take ranks but does not own deletion authority;
- commercial labels are descriptive only and cannot delete valid content;
- strategy is flexible, not a forced Hook->Problem->Benefit->Proof->CTA template.

## Gate 9 — Focused human benchmark

Before full-suite generalization, Video00 / Video02 / Video03 must each pass:
- automated invariants;
- physical MP4 artifact generated from current worker source;
- human Watch+Listen review;
- no known Gold violation.

Only after focused PASS may the 16-video benchmark run.

## Gate 10 — Draft contract / recovery

PASS requires:
- selected / alternatives / discarded are persisted;
- source ids and source timestamps are preserved;
- initial draft is saved server-side;
- signed playback/timeline asset URLs can be rehydrated;
- revision conflict handling works;
- autosave, undo and redo survive app restart;
- failed processing/render never destroys the last good draft.

## Gate 11 — Mobile timeline editor

PASS requires:
- filmstrip/thumbnails + waveform;
- playback/playhead sync;
- select/reorder/trim/split/delete/restore/swap take;
- captions on/off + text correction;
- mute/volume;
- text and minimum photo/video overlay controls;
- undo/redo;
- every edit autosaves;
- no duplicated word/audio bleed at cuts.

## Gate 12 — Export

PASS requires:
- server render job from persisted draft;
- 9:16 1080p H.264-compatible output;
- A/V sync validation;
- real render progress;
- retry/cancel where feasible;
- render version persisted and reopenable;
- final download/share path works on device.

## Gate 13 — Projects / batch / notifications / feedback

PASS requires:
- projects survive restart/logout/login policy;
- up to 10 independent batch items with per-item progress;
- one failed batch item does not block the rest;
- completion/failure notification path works;
- Good Edit / Bad Edit feedback is stored with model/source/selection metadata.

## Gate 14 — Commercial / observability foundation

PASS requires:
- measured-media usage accounting;
- per-user concurrency limits;
- operational telemetry for API, queue, worker, provider, render and failures;
- stuck-job detection/recovery;
- no secret values in logs/artifacts;
- durable production DB becomes source of truth before broad commercial launch; Redis must not be the only durable account/project store.

## Release classification

A change can be marked only as one of:
- CODE FIXED — implementation exists;
- CI GREEN — automated tests pass;
- COMPONENT PASS — isolated component runs correctly;
- E2E PASS — adjacent handoffs from mobile to final artifact pass;
- GOLD PASS — benchmark invariants pass;
- HUMAN WATCH+LISTEN PASS — user reviewed artifacts;
- V1 RELEASE READY — every launch-blocking gate above is green.

## Current highest-risk blockers

1. Stable GPU worker/provider startup.
2. Full device-to-API-to-worker-to-draft E2E on staging.
3. Focused human benchmark artifacts on the current brain source.
4. Durable recovery under interrupted upload/job/render conditions.
5. Final mobile editor/export E2E on a physical iPhone build.

No individual patch is allowed to imply product readiness unless its parent gate is also revalidated.