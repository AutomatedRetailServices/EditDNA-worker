# CutSell.ai — Mobile V1 ASAP Scope

## 1. Product Positioning

CutSell.ai is a mobile-first AI video production system for TikTok Shop sellers, affiliates and UGC creators.

### Core promise
Turn messy product footage into a conversion-ready, editable TikTok Shop video — automatically.

### Key differentiator
CutSell understands what the creator is trying to sell, not only what they said.

### Product advantage
CutSell combines:
- automatic raw-footage cleanup;
- watch + listen understanding;
- best-take selection;
- sales-aware structure;
- editable mobile timeline;
- reliable drafts, background processing and export;
- later, guided recording from product/script through Flow A.

CutSell must compete on sales intelligence and creator throughput, not on becoming a full CapCut clone.

---

## 2. ASAP Delivery Strategy

The product is delivered in three layers.

### Layer A — AI Editor
Raw footage → Watch + Listen → Clean Cut → Take Groups → Best Take → Commercial Meaning → Auto Strategy → Flexible Composer → Editable Draft.

### Layer B — Human Editor
Timeline → trim → split → delete → reorder → swap take → restore → captions → text → overlays → audio → undo/redo → export.

### Layer C — Reliable Product
Projects → upload recovery → autosave → background jobs → progress → notifications → drafts → render history → batch → account/subscription.

Flow B ships first because it solves the most immediate creator pain.

Flow A is built on top of the same engine and project/editor infrastructure after the Flow B core is stable.

---

## 3. V1 Launch Promise

A creator must be able to:

1. Open CutSell on mobile.
2. Record or upload one video OR select multiple clips.
3. Arrange multiple clips before processing.
4. Send footage to CutSell.
5. Leave the app while processing continues.
6. Receive a real progress state and completion notification.
7. Open an AI-edited draft where:
   - dead air is removed;
   - false starts and obvious fumbles are removed;
   - repeated takes are grouped;
   - the strongest take is selected;
   - valid alternatives remain available;
   - natural delivery is preserved;
   - commercial meaning is understood without forcing a rigid funnel.
8. Manually fine-tune the result in a lightweight timeline editor.
9. Export a sharp 9:16 video.
10. Return later and recover the project/draft if the app closes, the upload fails or the render fails.

---

## 4. Mobile Entry Flow

### Home / Create
Primary action: Create.

Creator chooses:

#### Single clip
Quickly edit one raw video.

#### Multiple clips
Combine multiple source videos/takes into one edit.

For Multiple Clips:
- add clips;
- remove clips;
- drag to arrange;
- display total duration;
- preserve source identity for every clip;
- optional audio overlap setting;
- Continue / Combine & Edit.

The user-provided order is treated as strong intent and preferred by the composer unless there is a clear reason to change it.

---

## 5. AI Engine — Flow B Core

### 5.1 Intake
For every source asset, preserve:
- source_asset_id;
- original filename;
- source order;
- duration;
- video/audio metadata;
- upload status;
- user ownership.

Never merge identities between source assets.

### 5.2 ASR
Required:
- English and Spanish V1;
- word-level timestamps;
- confidence where available;
- punctuation normalization;
- source-bound transcript mapping.

### 5.3 Watch + Listen Analysis
The engine must use both speech/audio and visual evidence.

Signals may include:
- transcript and word timings;
- silence/dead air;
- speech restart patterns;
- delivery pace;
- vocal confidence/clarity;
- face visibility;
- eye contact / engagement signal;
- framing;
- product visibility;
- motion;
- continuity;
- obvious visual fumble/restart behavior;
- audio quality.

Transcript-only editing is not sufficient for the final target engine.

### 5.4 Take Segmentation
Detect candidate takes and speech units while preserving:
- source boundaries;
- take boundaries;
- sentence/idea completeness;
- restart boundaries.

Forbidden:
- cross-source merges;
- cross-take sentence stitching that creates speech the creator never delivered;
- impossible microfragments.

### 5.5 Clean Cut
Clean Cut is only responsible for production cleanup.

May remove:
- dead air;
- obvious false starts;
- obvious fumbles;
- explicit restart/slate directions;
- incomplete production fragments;
- accidental adjacent duplicate residuals;
- unusable silence;
- boundary artifacts.

Must keep uncertain valid speech.

Clean Cut must never delete content because it is not a Hook/CTA/Benefit/etc.

### 5.6 Take Groups
Group valid attempts of the same semantic idea.

Example:
The creator says the same hook five times.

The engine should create one Take Group containing five candidates, not five independent unrelated ideas.

### 5.7 Best Take
Rank valid alternatives using multimodal evidence.

Signals:
- clarity;
- completeness;
- naturalness;
- energy;
- confidence;
- pacing;
- eye contact / engagement;
- visible distraction;
- audio quality;
- visual quality;
- product presentation;
- sales effectiveness;
- continuity with neighboring clips.

Output:
- one selected take;
- ranked alternatives;
- confidence;
- reasons/status;
- fallback status when AI is unavailable.

Best Take ranks valid content. It is not a deletion authority.

### 5.8 Commercial Meaning
Canonical semantic functions:
- HOOK
- PROBLEM
- FEATURES
- BENEFITS
- PROOF
- STORY
- CTA
- OTHER

Rules:
- optional;
- repeatable;
- descriptive;
- may be absent;
- CTA does not have to be last;
- Story may dominate the video;
- semantic classification must not delete otherwise valid speech.

### 5.9 Auto Strategy
The engine should automatically identify the dominant editing style.

Initial strategies:
- Direct Sales
- Storytelling
- Testimonial
- Demo / Product-led
- Educational
- Faceless / Voiceover
- Mixed

The user does not need to select a preset for the default experience.

Strategy guides ranking and composition but never becomes a rigid template.

### 5.10 Flexible Composer
Default behavior:
- preserve natural source order;
- respect creator-arranged source order;
- prefer complete ideas;
- remove obvious warmup when a stronger opening exists;
- allow repeated or missing commercial functions;
- keep storytelling coherent;
- never force HOOK → PROBLEM → BENEFIT → PROOF → CTA;
- do not automatically force CTA to the end.

Output is an editable Draft Timeline, not merely a final MP4.

---

## 6. Editable Draft Contract

Draft contains:
- selected timeline clips;
- alternate takes grouped by idea;
- discarded production mistakes;
- source identity;
- source timestamps;
- semantic role;
- selection confidence;
- captions;
- visual/audio metadata needed by editor;
- edit history/version.

Creator actions:
- Swap Take;
- Restore;
- Remove;
- Reorder;
- Trim;
- Split;
- Caption Edit;
- Undo;
- Redo.

Every edit autosaves.

---

## 7. Mobile Timeline Editor — V1 Minimum

CutSell is not trying to replace CapCut in V1.

It must provide enough control to finish a TikTok Shop video without leaving CutSell.

### Timeline
- visual filmstrip/thumbnails;
- playhead;
- smooth horizontal scroll;
- zoom/pinch zoom when practical;
- clip selection;
- drag reorder;
- trim handles;
- split at playhead;
- delete;
- undo/redo;
- autosave.

### Audio
- waveform;
- mute/unmute;
- volume;
- audio sync preservation;
- no duplicated word bleed between cuts;
- optional natural audio overlap between adjacent clips.

### Captions
- automatic captions;
- captions on/off;
- text correction;
- timing preservation;
- basic visual preset(s).

### Text
- add text;
- edit text;
- place on timeline;
- move/resize on canvas.

### Overlays
V1 minimum:
- photo overlay;
- video overlay;
- move/resize;
- trim overlay;
- mute overlay video audio;
- overlay timeline lane.

Advanced keyframe animation is not a V1 launch blocker.

---

## 8. Reliable Mobile Experience

### Upload
Required:
- direct/resumable multipart upload;
- retry interrupted chunks;
- support background handoff where platform allows;
- client-side video preparation/compression when needed;
- preserve original metadata safely;
- clear visible upload state.

### Processing
Required states:
- preparing;
- uploading;
- uploaded;
- transcribing;
- analyzing;
- composing;
- draft_ready;
- rendering;
- finished;
- failed;
- canceled.

No fake progress bars.

### Background behavior
After upload is safely handed off:
- user can minimize/leave;
- server continues processing;
- user receives notification when draft/render is ready.

### Recovery
Required:
- draft is never lost on app termination;
- upload can resume;
- failed processing can be retried;
- failed render does not destroy draft;
- stuck processing state can self-heal or be safely retried;
- user can resend a failed source without rebuilding the project.

---

## 9. Export

Default:
- 9:16;
- 1080p;
- sharp H.264/compatible mobile output;
- audio/video sync guaranteed by validation;
- visible real progress;
- cancel where feasible;
- final saved render version.

Future/optional settings:
- 2K/4K;
- frame-rate controls up to 60fps;
- bitrate controls.

Higher export controls must not delay the first usable V1 if 1080p output is strong and reliable.

---

## 10. Projects and Drafts

Required entities:
- User
- Project
- SourceAsset
- ProcessingJob
- CandidateClip
- TakeGroup
- DraftTimeline
- DraftEdit
- CaptionTrack
- OverlayTrack
- RenderVersion

Every Project must survive logout/app restart.

Each user only accesses their own projects and media.

Signed/expiring media URLs required.

---

## 11. Batch / High-Output Creator Mode

Batch is highly relevant to TikTok Shop creators and should arrive early.

Initial batch:
- queue up to 10 videos/projects;
- submit one after another;
- show per-item progress;
- failures do not block remaining items;
- reopen finished cuts individually.

Batch combining multiple clips into one project uses the Multiple Clips flow, not the batch-processing queue.

---

## 12. Feedback Learning Loop

Every finished AI draft should support:
- thumbs up / Good Edit;
- thumbs down / Bad Edit;
- optional reason/category;
- optional clip/time marker;
- report problem.

Store feedback with:
- model version;
- strategy;
- selected clips;
- alternatives;
- semantic statuses;
- fallback statuses;
- source IDs;
- processing metrics.

This becomes the production evaluation dataset for improving CutSell.

Human evaluation of finished cuts is more important than isolated model scores.

---

## 13. Flow A — Phase 2 on Same Platform

Flow A is retained but must not delay Flow B launch.

### Flow A experience
1. Create project.
2. Add product link or product information.
3. Generate script OR paste existing script.
4. Edit script.
5. Split into recording cards.
6. Choose or auto-recommend style.
7. Record one/multiple takes per card.
8. Teleprompter.
9. Optional voiceover/TTS later if needed.
10. Run same Watch + Listen / Clean Cut / Best Take engine.
11. Build editable draft.
12. Finish in same timeline editor.
13. Export variants.

Flow A initial styles:
- Talking Head
- Testimonial
- Storytelling
- Demo
- Voiceover / Faceless

Complex skit automation may follow after initial Flow A.

---

## 14. Variants — Competitive Differentiator

Once a project contains enough valid material, CutSell can generate multiple sales edits without re-recording.

Variant dimensions:
- alternate hook;
- alternate take;
- alternate proof;
- alternate CTA;
- shorter cut;
- storytelling cut;
- direct-sales cut.

V1 can initially expose a small number of high-confidence variants.

Do not generate variants by creating fake speech or rearranging content into incoherent sequences.

---

## 15. What Is NOT a V1 Launch Blocker

Defer unless required by testing:
- full CapCut-level keyframe system;
- advanced animation engine;
- huge template marketplace;
- advanced generative B-roll;
- broad multilingual support beyond English/Spanish;
- agency/team collaboration;
- universal e-commerce scraping;
- fully automatic publishing to every platform;
- deep brand-kit automation;
- advanced TTS studio;
- complex scripted skit generation.

---

## 16. Technical Build Doctrine for the New Worker

Do not reproduce the oversized legacy pipeline.

The new engine should use small modules with explicit ownership:

1. intake
2. media_probe
3. asr
4. silence_analysis
5. take_segmentation
6. clean_cut
7. visual_analysis
8. semantic_analysis
9. take_grouping
10. take_judge
11. strategy
12. composer
13. draft_builder
14. captions
15. render_plan
16. render
17. observability

Rules:
- each module has one responsibility;
- typed/versioned contracts between stages;
- deterministic source identity;
- fail-open for uncertain deletion;
- provider failures observable;
- unit tests per module;
- golden real-video evaluation set;
- no hidden commercial deletion heuristics.

---

## 17. Build / Delivery Order — ASAP

### Milestone 0 — Foundation
Deliver:
- new CutSell worker architecture;
- API contracts;
- source identity;
- job state model;
- project/draft schema;
- test harness;
- CI.

### Milestone 1 — Flow B Brain
Deliver:
- upload one/multiple videos;
- ASR;
- Watch + Listen signals;
- take segmentation;
- Clean Cut;
- Take Groups;
- Best Take;
- semantic meaning;
- Auto Strategy;
- flexible composer;
- editable Draft JSON.

Exit condition:
real messy videos produce coherent drafts without cross-source merges or destructive commercial deletions.

### Milestone 2 — Mobile Draft Editor
Deliver:
- timeline;
- trim;
- split;
- delete;
- reorder;
- Swap Take;
- Restore;
- captions;
- waveform;
- text;
- photo/video overlay;
- undo/redo;
- autosave.

### Milestone 3 — Reliable Product
Deliver:
- resumable uploads;
- projects;
- draft recovery;
- background processing;
- push completion notifications;
- reliable render/re-render;
- secure media URLs;
- feedback 👍/👎;
- basic subscriptions/usage limits.

### Milestone 4 — Beta Launch
Deliver:
- 1080p reliable export;
- English/Spanish validation;
- real-video benchmark;
- crash/error instrumentation;
- closed TestFlight beta;
- creator feedback loop.

### Milestone 5 — Throughput
Deliver:
- batch queue;
- limited high-confidence variants;
- faster repeat workflows.

### Milestone 6 — Flow A
Deliver:
- product/script intake;
- script generation/editing;
- recording cards;
- teleprompter;
- card-level takes;
- same AI engine/editor/export path.

---

## 18. V1 Acceptance Criteria

The V1 beta can ship when:

- one and multiple clip intake work reliably;
- uploads resume after normal mobile interruption;
- AI processing continues in background;
- dead air and obvious fumbles are removed reliably;
- valid speech is not deleted because of commercial labels;
- repeated takes remain as alternatives;
- Best Take beats naive last-take selection on the evaluation set;
- no cross-source sentence fabrication;
- user can manually finish the video in CutSell;
- autosave and recovery work;
- captions can be edited;
- audio remains synchronized and no words bleed/duplicate across cuts;
- 1080p export is sharp and reliable;
- project can be reopened later;
- errors and AI fallbacks are observable;
- English and Spanish real-world sessions pass human review;
- thumbs up/down feedback is captured.

---

## 19. North-Star User Experience

The user should feel:

"I dump my ugly TikTok Shop footage into CutSell, it understands what I was trying to say, chooses the best delivery, cleans the mess, gives me a strong sales edit, and I can fix anything else in seconds without opening another editor."

That is the V1 product standard.
