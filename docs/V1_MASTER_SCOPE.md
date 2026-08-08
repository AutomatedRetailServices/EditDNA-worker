# EditDNA.ai / CutSell.ai — V1 Master Scope

## Objective
Build a conversion-first mobile/web editor for TikTok Shop, UGC, affiliate and direct-response creators by combining:

- EditDNA Flow A: script-driven recording and editing.
- EditDNA Flow B: raw improvised footage and blooper cleanup.
- Cut-style automatic editing: remove mistakes, compare takes, preserve natural delivery and assemble a coherent video.
- A flexible commercial structure rather than a mandatory linear funnel.

The V1 target is English and Spanish commercial content. Broad multilingual/script-family support is explicitly deferred.

## Product contract

### Flow A — Script Provided
1. Create project.
2. Add product link or product information.
3. Generate a script or paste an existing script.
4. Split script into editable semantic cards.
5. Choose style: talking head, testimonial, skit, voiceover or faceless/lifestyle.
6. Record or upload one or more takes for each card.
7. Use teleprompter, optional TTS/voiceover and caption preview.
8. Run Clean Cut and Best Take.
9. Produce an editable draft with alternates.
10. Render one or more vertical variants.

### Flow B — Raw Improv
1. Create project.
2. Upload retries, bloopers and raw takes from one or multiple source videos.
3. Transcribe with word timestamps.
4. Detect candidate takes without crossing source/take boundaries.
5. Remove only real mistakes, false starts, slate/restart instructions and unusable silence.
6. Preserve valid repeated takes as alternatives.
7. Classify commercial function using Semantic V2 and visual context.
8. Compare takes using clarity, pacing, energy, conviction, naturalness, audio, visual quality and continuity.
9. Assemble a flexible draft preserving natural source order.
10. Allow swap, restore discarded takes, caption edits and re-render.

## Semantic model
Allowed functions:

- HOOK
- PROBLEM
- FEATURES
- BENEFITS
- PROOF
- STORY
- CTA
- OTHER

Rules:

- Slots are optional and repeatable.
- No mandatory HOOK → PROBLEM → BENEFITS → CTA sequence.
- Natural source order is preferred.
- CTA is not forced to the end.
- Slot classification must not decide whether speech is deleted.

## Required pipeline architecture

Raw footage
→ secure upload and source registration
→ ASR with word timestamps
→ candidate clip/take segmentation
→ Clean Cut
→ semantic + visual understanding
→ Best Take / alternates
→ flexible commercial composer
→ editable draft
→ captions / preview / thumbnails
→ render
→ secure project library

### Layer ownership

#### Clean Cut
Responsible only for:

- silence and boundary cleanup;
- false starts and explicit restart/slate instructions;
- impossible microfragments;
- accidental adjacent residual duplicates;
- preserving source identity;
- never merging across takes or sources.

Clean Cut must not use commercial slot classification as deletion authority.

#### Semantic V2
Responsible for commercial function and meaning. It is the primary semantic authority.

#### Visual understanding
Responsible for product presence, framing, face visibility, motion/continuity and visual suitability.

#### Take Judge V2
Responsible for ranking valid alternatives, not deleting content based only on slot.

#### Fallback
Must be small, conservative and fail-open. It may detect obvious production commands and prevent crashes, but must not become a phrase-by-phrase commercial brain.

## Current repository status

### Present

- FastAPI application.
- `/render` asynchronous submission.
- `/jobs/{job_id}` status retrieval.
- cancellation endpoint.
- Redis/RQ queue.
- progress stages.
- multi-URL render input.
- modes: clean, human and blooper.
- S3 helpers and rendering infrastructure.
- ASR modules.
- large existing pipeline.
- Semantic V2 module.
- Take Judge V2 module.
- semantic/visual pass.
- funnel/composer code.
- benchmark infrastructure.
- tests for API/RQ, cancellation and pipeline components.

### Present but incomplete or unsafe for V1

- Clean Cut is coupled to semantic/commercial decisions in parts of the pipeline.
- Semantic V2 and Take Judge may fall back instead of being the effective authority.
- visual scoring is not yet a proven production authority.
- pipeline.py is oversized and mixes orchestration, heuristics, scoring, boundaries and composition.
- current public API returns a final job result but not a complete editable project/draft contract.
- render retries are disabled by default because errors are not fully classified.
- no persistent user/project ownership model.
- no resumable direct mobile upload flow.
- no clip/alternate/discard/swap API.
- no incremental re-render contract.
- no versioned public API contract.
- no quota/billing enforcement.
- no closed-beta acceptance metrics.

### Not implemented as a complete product

- Flow A product ingestion and script generation.
- semantic recording cards.
- teleprompter workflow.
- TTS/voiceover workflow.
- project library and history.
- secure per-user media ownership.
- upload resume/retry for unstable mobile networks.
- editable draft UI contract.
- alternate take and discarded-take restoration workflow.
- mobile push/completion notification.
- analytics for time-to-first-draft, swap rate and render success.

## Structural recovery plan

### Phase 1 — Repository recovery

1. Keep PR #21 and PR #22 frozen as reference only.
2. Work from `repair/v1-structural-recovery` created from `main`.
3. Recover only validated changes:
   - source identity;
   - boundary repair using word timestamps;
   - microfragment rejection;
   - adjacent residual duplicate suppression;
   - sanitized diagnostics;
   - backward-compatible visual score API;
   - flexible source-order composer.
4. Do not port broad multilingual/CJK logic.
5. Do not port an oversized phrase-by-phrase commercial fallback.

### Phase 2 — Clean architecture boundary

1. Extract Clean Cut decisions from commercial slot rules.
2. Make deletion reasons explicit and enumerable.
3. Preserve all valid repeated takes as alternatives.
4. Reject cross-source and cross-take merges by identity, not phrase heuristics.
5. Keep uncertain content instead of deleting it.
6. Add contract tests proving slot changes cannot delete a valid clip.

### Phase 3 — Primary intelligence

1. Make Semantic V2 the primary slot classifier when enabled.
2. Make Take Judge V2 the primary alternative ranker when enabled.
3. Add visual/audio signals to ranking without hard deleting valid speech.
4. Record provider/fallback usage on every processed clip.
5. Define safe fallback reasons and fail-open behavior.

### Phase 4 — Flexible composer

1. Preserve natural source order by default.
2. Allow missing and repeated slots.
3. Avoid forced CTA placement.
4. Select a primary take per semantic idea while retaining alternatives.
5. Produce a draft timeline plus alternate groups, not only a final MP4.

### Phase 5 — Flow B validation

Validation sequence:

1. unit and contract tests;
2. one controlled session;
3. representative English and Spanish sessions;
4. all 61 historical sessions;
5. rendered-video review.

Required metrics:

- valid clips retained;
- true mistakes removed;
- cross-take merge count;
- impossible microfragment count;
- adjacent duplicate residual count;
- selected-vs-alternate agreement;
- semantic provider usage;
- Take Judge provider usage;
- fallback rate and reasons;
- processing time;
- estimated cost;
- render success rate.

### Phase 6 — Editable mobile backend

Required resources:

- User
- Project
- SourceAsset
- ProcessingJob
- CandidateClip
- TakeGroup
- DraftTimeline
- CaptionTrack
- RenderVersion

Required API capabilities:

- create/list/read projects;
- presigned multipart upload;
- resume and complete upload;
- submit processing job with idempotency key;
- poll job state and cancel safely;
- retrieve draft timeline;
- retrieve selected, alternate and discarded takes;
- swap/restore/remove clip;
- edit captions;
- request preview and final render;
- list render versions and history;
- enforce per-user ownership and secure URLs.

### Phase 7 — Flow A

1. product information ingestion;
2. script generation and editing;
3. semantic cards and recording plan;
4. teleprompter;
5. card-level take upload/recording;
6. optional TTS and voiceover;
7. Clean Cut and Best Take per card;
8. editable draft and variants;
9. same project/render infrastructure as Flow B.

### Phase 8 — Closed beta

Entry criteria:

- full test suite passes without ignored files;
- Flow B produces an editable draft;
- users can swap and restore takes;
- jobs survive normal mobile reconnect/poll behavior;
- output URLs are secure and owned by the correct user;
- one controlled and all historical benchmark runs complete;
- failure reasons and costs are observable;
- no known cross-source merges or impossible microfragments.

## ASAP delivery priority

### V1 launch blockers

1. structural recovery and full tests;
2. reliable Flow B Clean Cut;
3. primary Semantic V2 / Take Judge integration;
4. editable draft contract;
5. secure resumable uploads and project ownership;
6. swap, discarded takes and re-render;
7. minimum Flow A script/card/record workflow;
8. closed beta instrumentation.

### Can wait until after closed beta

- broad multilingual support;
- advanced agency collaboration;
- universal product-page scraping;
- large template marketplace;
- fully automatic multi-platform publishing;
- advanced generative B-roll;
- deep brand-kit automation.

## Definition of V1

V1 is not merely an automatic MP4 renderer. It is complete when a creator can:

1. start from a script or raw improvised footage;
2. upload reliably from mobile;
3. receive a coherent editable vertical-video draft;
4. inspect selected, alternate and discarded takes;
5. swap clips and edit captions;
6. re-render without rebuilding the entire project manually;
7. return later to the project and its versions;
8. trust that real speech was not deleted because of a commercial slot guess.
