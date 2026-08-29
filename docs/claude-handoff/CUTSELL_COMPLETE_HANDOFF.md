# CutSell.ai — Complete Claude Code Handoff

This is the durable continuation contract for CutSell.ai. It consolidates the useful engineering/product lineage from Carga Snapshot / EditDNA through CutSell.ai 1–11 and the current Unified Selection pivot. It is intentionally not a raw chat dump: historical experiments are separated from current truth so Claude Code does not resurrect superseded architecture.

## 1. SOURCE PRECEDENCE — NON-NEGOTIABLE
When information conflicts, use this order:

1. live GitHub code and current branch state;
2. this current handoff;
3. current scope/decision contracts;
4. existing canonical repo doctrine;
5. historical checkpoints;
6. old conversation text.

Never reset to an older SHA simply because an older doc/chat calls it canonical.

---

# CURRENT LIVE STATE

## Repository
- Repository: `AutomatedRetailServices/EditDNA-worker`
- Active branch: `cutsell/mobile-v1-clean`
- PR: `#25`
- PR state: OPEN / DRAFT / UNMERGED
- Base: `main`
- Protected base SHA: `2fb13e5aa228e8e525b942a9b49182032b797e61`
- Functional code head immediately before Claude handoff docs: `9fad0788e120b2af07576f2d145fc4c179b24adb`
- Functional head message: `Activate unified whole-video Selection RAW gate`

The branch will contain newer documentation-only commits for this Claude handoff. Do not confuse those documentation commits with a Brain change.

## Current CI on functional head
- CutSell Clean Worker CI run `33222855861`, run #2113: SUCCESS
- CutSell iOS CI run `33222855844`, run #1894: SUCCESS

Therefore:
- CODE/CI: GREEN
- EDITORIAL QUALITY: NOT YET PROVEN

## Current active phase
**Flow B → Universal Clean Cut → Unified Whole-Video Selection**

Do not drift into sales funnel, B-roll, URL→Ad, variants, advanced captions, production deploy or TestFlight while this blocker remains unresolved.

---

# PRODUCT LINEAGE

## EditDNA origin
EditDNA began as an AI editor for TikTok Shop / UGC footage: messy raw creator takes → clean sales-ready video.

Core early concepts included:
- Flow A: script/guided recording;
- Flow B: raw upload;
- Hook / Problem / Feature / Proof / CTA concepts;
- silence/filler removal;
- alternates;
- 1080×1920 export;
- teleprompter and B-roll presets;
- later semantic/visual plans such as duplicate filtering, OCR/product understanding and multimodal signals.

## Infrastructure proof
The project moved beyond concept and proved a working backend lineage:
- FastAPI / Render;
- Redis / RQ;
- RunPod GPU worker;
- S3;
- Postman E2E;
- first stitched outputs;
- S3 upload/download path;
- `/render → Redis → worker → S3` proven;
- full E2E execution.

Do not rebuild these foundations as if CutSell were a greenfield prototype.

## Rebrand to CutSell.ai
The selected brand is **CutSell.ai** singular.

Positioning sharpened toward:
- TikTok Shop sellers/affiliates/UGC creators;
- raw creator footage → clean/editable sellable draft;
- mobile-first creator workflow;
- later broader UGC Factory / URL→Ad possibilities.

## Mobile V1 clean architecture
The clean release path includes:
- `cutsell_worker/` — AI/media Brain;
- `cutsell_app/` — API;
- `mobile/ios/` — native SwiftUI;
- CutSell-only CI;
- isolated GPU worker/container workflows;
- staging/release contracts;
- editable draft/product reliability architecture.

PR #25 became the active clean branch snapshot. `main` remains protected.

---

# PRODUCT SCOPE — WHAT STILL MATTERS

## Flow B — CURRENT / FIRST
Creator already has footage.

Intended path:
raw/imported footage → ASR → Watch + Listen → Clean Cut → take/candidate evidence → semantic Selection → Boundary cleanup → editable Draft → render/export.

Flow B ships first.

## Flow A — LATER, SAME PLATFORM
Future guided recording path:
- product information/URL;
- script generation or pasted script;
- recording cards;
- teleprompter;
- multiple takes;
- same Watch + Listen / Clean Cut / Selection engine;
- editable draft;
- export/variants.

Flow A must not delay current Flow B Brain quality.

## Broader Mobile V1 retained
Still valid later/current product scope:
- in-app creator recording;
- gallery/Photos import;
- up to 10 clips in one multi-clip project;
- arrange/remove/add clips;
- resumable/background upload;
- real Photos/iCloud preparation states;
- editable mobile timeline;
- trim/split/delete/reorder;
- Swap Take / Restore;
- undo/redo;
- captions/basic text/overlay/audio;
- autosave;
- 9:16 1080p export;
- project recovery;
- render history;
- early Batch/high-output creator mode.

CutSell is NOT trying to become a full CapCut clone in V1.

---

# CURRENT EDITORIAL CONSTITUTION

## Preserve
- creator story;
- personality;
- humor;
- natural source logic;
- valid audience-facing information;
- numbers;
- names;
- negations;
- causal claims;
- genuinely new facts;
- necessary continuations;
- useful context.

## Remove
- false starts;
- abandoned takes;
- obvious verbal fumbles;
- failed delivery;
- BTS / recording-process material;
- explicit retry setup;
- unusable dead air;
- inferior duplicate delivery when it adds no unique audience-facing information;
- physical boundary debris.

## Core safety rule
**WHEN UNCERTAIN, KEEP.**

In the current model, preserving may mean SELECT or SWAP rather than destructive DISCARD.

## No Frankenstein speech
Never fabricate a sentence by stitching words/fragments across incompatible takes or sources.

A human-like Composite Best Take is allowed only from real delivered pieces/continuations that naturally form the intended delivery.

## Human performance matters
A transcript can be linguistically complete while the take is visually/performance-wise failed because of:
- frustration;
- lost-line behavior;
- eye/camera disengagement;
- body reset;
- camera adjustment;
- product-handling mistake;
- visible fumble;
- recording-process behavior.

Do not confuse authentic personality with a recording mistake.

---

# SELECTION VS BOUNDARY — BINDING ARCHITECTURE

## Selection owns semantics
Selection decides:
- which content lives;
- which take wins;
- retries;
- failed/BTS material;
- alternates;
- semantic story/content survival;
- continuation/composite membership.

## Selection Freeze
After final Selection and complete-idea recovery, semantic membership freezes.

## Boundary owns physical timing only
Boundary may:
- trim in/out;
- remove source-evidenced non-speech gaps;
- refine audio/visual cut timing;
- physically split/coalesce while preserving the ordered spoken stream.

Boundary may NOT:
- pick another semantic take;
- replace words;
- silently delete audience-facing content;
- reorder semantic spoken content.

Post-freeze Selection Contract verification must fail if Boundary mutates the spoken semantic stream.

---

# WHY THE ARCHITECTURE PIVOTED

The previous Brain became increasingly sophisticated with:
- whole-video context;
- attempt reconstruction;
- session boundaries;
- retry grouping;
- cross-session sibling reconciliation;
- Hybrid/Gemini semantic reasoning;
- complementary-delivery guards;
- final selection integrity layers;
- post-selection boundary protections.

That work was valuable and produced strong evidence, but Video00 repeatedly exposed a structural weakness: semantic decisions were often made in local/mini-session contexts, so competing takes could be partitioned and never evaluated together.

Over time the system accumulated too many semantic guards/thresholds. The user explicitly approved stopping the pattern of “one more Video00 guard / one more threshold” as the primary method.

Current direction:
> one whole-video final semantic Selection authority sees the entire candidate universe and returns one coherent plan before Selection Freeze.

Legacy/local modules may remain as upstream evidence and historical/fallback architecture, but they must not become a second competing final semantic brain when Unified Selection is enabled.

---

# UNIFIED WHOLE-VIDEO SELECTION — CURRENT

## Key files
- `cutsell_worker/brain_runtime.py`
- `cutsell_worker/unified_selection_reasoner.py`
- `cutsell_worker/unified_selection_google.py`
- `cutsell_worker/universal_clean_cut.py`
- `cutsell_worker/universal_clean_cut_validation.py`
- `cutsell_worker/serverless_handler.py`
- `.github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml`

The workflow filename contains legacy `v5`; this is only a historical filename, not the active editor version. Its display name was changed to `CutSell Video00 Unified Selection RAW`.

## Runtime gates
Benchmark environment uses:
- `CUTSELL_BRAIN_BACKEND=runpod_local`
- `CUTSELL_EDITORIAL_MODE=clean_cut`
- `CUTSELL_ASR_MODEL=medium`
- `CUTSELL_HYBRID_LLM_ENABLED=1`
- `CUTSELL_HYBRID_PROVIDER=google`
- `CUTSELL_UNIFIED_SELECTION_REASONER=1`

## Provider policy
Current approved primary model:
`gemini-3.5-flash-lite`

Optional escalation model exists:
`gemini-3.6-flash`

Do not casually change provider/model while diagnosing an observability contract issue.

## Candidate universe
Unified Selection sees all unique semantic candidates from:
- Selected;
- SWAP/Alternates;
- recoverable Discarded.

Evidence can include:
- source order;
- timestamps;
- text;
- take group id;
- current bucket;
- whole-video context;
- upstream/local Hybrid votes.

These are evidence, not final authority.

## Output actions
Every candidate gets exactly one:
- `select`
- `swap`
- `discard`

## Output relations
- `independent`
- `retry_winner`
- `retry_alternate`
- `composite_piece`
- `continuation`
- `failed`
- `bts`
- `uncertain`

## Safety application
- uncertain relation or confidence `< 0.70`:
  - current Selected stays Selected;
  - other content is preserved as SWAP;
- DISCARD below `0.80` confidence is demoted to SWAP;
- provider/validation error fails open and leaves existing draft membership intact.

## No two semantic brains
When Unified Selection reasoner exists:
- legacy bounded `editorial_judge` is intentionally OFF;
- two paid semantic brains must not fight over the edit.

---

# VIDEO00 — AUTHORITATIVE VALIDATION PAIR

## RAW
S3 key:
`Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4`

Approx duration:
`366.997s` / `6:06.997`

## Authoritative current Human Gold
Known filename:
`5E01F214-A364-4F4B-8F25-D39B1E2B21D2(1).MP4`

S3 naming:
`Editdna longform validation/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4`

Approx duration:
`141.667s` / `2:21.667`

Human Gold is the editorial oracle for Video00.

Do NOT substitute:
- old historical V5 approval;
- old generated previews;
- output duration similarity.

## Human Gold Decision Map
`human_gold_decision_map_v2.py` is QA/oracle tooling only.

Never turn Human Gold timestamps/phrases into production runtime rules.

## Why Video00 matters
It contains hard human editorial cases:
- retries with changed wording;
- complete delivery vs split retry;
- valid adjacent independent statements;
- continuations;
- false starts;
- fumbles;
- composite-best-take cases;
- boundaries.

Video00 is a teacher/eval anchor, not production-specific logic.

After Video00 passes, unseen videos and the broader regression suite must prove generalization.

---

# HISTORICAL CUTSELL.AI 11 CONTINUITY

A trusted CutSell.ai 11 recovery checkpoint is:
`8ab5ff45ff2ad3af7e90708be8f331f66669e52e`

PR #25 preserves that checkpoint in its description for continuity.

Important distinction:
- it is the canonical recovery point for what CutSell.ai 11 meant;
- it is NOT the current head;
- do not automatically reset to it;
- the current Unified Selection pivot was explicitly approved after that history.

A prior continuation mistake rolled back too early and caused regressions. Durable lesson:
**never reconstruct active CutSell state from fragmentary memory or stale chat. Verify live GitHub.**

---

# QA DISCIPLINE

## Binding loop
`diagnose → fix → targeted tests → CI → one controlled RAW → retrieve result JSON+MP4 → inspect → Human Gold → Watch+Listen → repeat`

Do not skip the artifact/result and immediately invent another semantic change.

## Quality states are separate
1. CODE FIXED
2. TESTS PASS
3. CI GREEN
4. RAW TECHNICALLY COMPLETED
5. ARCHITECTURE GATE PASS
6. SELECTION PARITY PASS
7. BOUNDARY/CONTINUITY PASS
8. GOLD CANDIDATE
9. HUMAN WATCH + LISTEN PASS
10. REGRESSION / UNSEEN PASS

Workflow success alone is not editorial quality.
Duration match alone is not Human Gold parity.

## Selection first
Selection parity comes before Boundary polish.
Do not use Boundary to hide a bad semantic Selection.

## Anti-overfitting
Never create runtime conditions based on:
- Video00 timestamp;
- clip id;
- literal Video00 phrase;
- a named symptom/topic from this benchmark;
- “if Video00”.

Historical intervals can be QA clues only.

---

# CURRENT LIVE BLOCKER — RAW #114

## Functional head
`9fad0788e120b2af07576f2d145fc4c179b24adb`

## GitHub RAW
- label: Unified Selection RAW #114
- GitHub run ID: `33222852152`
- benchmark ID: `video00-unified-selection-33222852152-1`
- RunPod job ID: `ff4dbbf3-6eb3-41d8-9c04-db2afb71c7a1-u2`
- immutable image digest: `sha256:b781809ba32259011dce7eead625c73ecb78e25228772caf5dcbc86199db26a8`

## #114 passed
- exact-head checkout ✅
- verify Unified Selection path ✅
- GHCR image build/push ✅
- base RunPod template load ✅
- temporary Unified Selection template ✅
- `CUTSELL_UNIFIED_SELECTION_REASONER=1` present ✅
- Gemini key presence guard ✅
- endpoint roll ✅
- CUDA health ✅
- original six-minute Video00 submit ✅
- RunPod job reached `COMPLETED` ✅
- teardown / workersMax=0 / temporary template removal ✅

## GitHub failed at
`Wait for unified Selection result`

The workflow required compact RunPod output:

```text
output.ok == true
output.speech_lock_ok == true
output.external_brain_calls_enabled == true
output.selection_reasoner_enabled == true
output.selection_reasoner_status == "applied"
```

and exited 1 after RunPod reached `COMPLETED`.

No diagnostic artifact was downloaded because the artifact steps came after this gate.

## Strong current root-cause hypothesis
This should currently be classified as:

**BENCHMARK OBSERVABILITY / SERVERLESS RETURN-CONTRACT MISMATCH**

—not yet as an editorial failure.

Why:

`run_single_universal_clean_cut_validation()` already returns fields including:
- `selection_reasoner_enabled`
- `selection_reasoner_status`
- `selection_reasoner_provider`
- `selection_reasoner_model`
- full diagnostics.

But `serverless_handler._focused()` writes the full result JSON to S3 and returns only a compact dict. The compact dict currently does NOT include:
- `selection_reasoner_enabled`
- `selection_reasoner_status`
- provider/model reasoner fields.

Therefore the GitHub jq architecture gate asks the compact RunPod output for keys the handler does not currently expose.

Also, when the `COMPLETED` gate failed, the workflow did not print the full `/tmp/status.json`, so the exact compact response was not preserved in GitHub logs.

Do NOT infer that Gemini/Unified Selection itself failed until the full result JSON is made observable and inspected.

---

# EXACT NEXT ACTION — CLAUDE MUST START HERE

## Objective
Repair Unified Selection benchmark observability so the actual reasoner output can be inspected before any new semantic change.

## A. Fix compact serverless return
In `cutsell_worker/serverless_handler.py`, `_focused()` should return at least:
- `selection_reasoner_enabled`
- `selection_reasoner_status`
- `selection_reasoner_provider`
- `selection_reasoner_model`
- `hybrid_requested_group_count`

Useful optional non-secret summary fields:
- `alternate_count`
- `selected_duration_sec`
- candidate/decision counts.

Do not expose secrets.

## B. Fix workflow evidence preservation
In `.github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml`:

When RunPod reaches `COMPLETED`:
1. preserve/print a sanitized architecture summary BEFORE a failing gate;
2. capture `preview_uri` / `result_uri` if present;
3. if `result_uri` exists, download full result JSON even if architecture validation later fails;
4. upload diagnostic artifact when possible;
5. run explicit architecture gate after evidence is preserved.

A failed gate must not destroy the evidence needed to diagnose it.

## C. Tests
Add targeted coverage proving:
- `_focused()` exposes reasoner state;
- compact state matches full validation result;
- disabled/provider-error state remains observable;
- secrets are not included.

## D. CI
Run targeted tests, then Clean Worker CI and iOS CI.
Do not launch the RAW until relevant CI is green.

## E. Run exactly one controlled Unified Video00 RAW
Required proof before editorial diagnosis:
- exact intended head;
- reasoner enabled;
- `selection_reasoner_status == applied`;
- legacy bounded Hybrid mini-session semantic requests = zero under Unified pivot;
- result JSON exists;
- MP4 exists;
- Selection contract/lock result known;
- teardown passes.

## F. Inspect reasoner decisions BEFORE semantic patches
Read:
`diagnostics.unified_selection_reasoner.decisions`

Analyze by:
- family_index;
- previous bucket;
- model action;
- effective action;
- relation;
- confidence;
- reason_code;
- safety override.

If a decision is wrong, classify the root cause before changing code:
- candidate missing?
- source context insufficient?
- prompt/schema problem?
- model relation/family reasoning problem?
- safety threshold interaction?
- downstream semantic mutation?
- Boundary problem?

Do NOT reflexively add a new guard.

## Completion criteria for this immediate task
The observability task is complete when:
1. compact return contract is fixed;
2. tests pass;
3. CI green;
4. one new RAW produces downloadable JSON + MP4;
5. Unified architecture status is known;
6. reasoner decision table is available for real diagnosis.

Video00 does NOT need to be Gold for this observability task to be complete.

---

# SECURITY CONTINUITY

Security is a parallel gate. Read `docs/claude-handoff/SECURITY_CONSTITUTION.md`.

High-level priorities include:
- auth/session correctness;
- per-user/project/source/draft ownership;
- secure uploads/media parsing;
- S3 signed/private access;
- secret hygiene;
- API validation/rate limits;
- least-privilege GitHub Actions/cloud access;
- dependency/container/supply-chain scanning;
- security regression tests.

Do not derail the immediate Selection observability fix, but do not introduce new insecure patterns while fixing it.

---

# REPOSITORY PROTECTION

Without explicit user approval:
- do not merge PR #25;
- do not modify `main`;
- do not close PR #25;
- do not deploy production;
- do not perform TestFlight/App Store release;
- do not expose/move secrets;
- do not make destructive repository/archive changes;
- do not silently introduce materially new recurring paid infrastructure.

Controlled continuation of the already-established Video00 benchmark loop should avoid duplicate/overlapping runs and must always tear down temporary RunPod resources.

---

# EXISTING REPO DOC RECONCILIATION

Existing `AGENTS.md`, `docs/CUTSELL_CURRENT_STATE.md`, `docs/CUTSELL_DECISIONS.md`, `docs/CUTSELL_BRAIN_DOCTRINE.md`, and `docs/CUTSELL_MOBILE_V1_ASAP_SCOPE.md` contain important doctrine/history.

Some operational headers are stale (for example older CutSell generation/benchmark/current-head labels). Use their durable doctrine, but this handoff supersedes stale operational state.

Especially retain from existing doctrine:
- Flow B first;
- Clean Cut first;
- valid speech cannot be deleted merely for weak commercial value;
- flexible—not rigid—sales/story logic;
- global context;
- no Frankenstein speech;
- immutable source identity;
- editable draft;
- preserve authentic personality;
- benchmark success != human quality.

After Claude establishes a successful Unified RAW checkpoint, synchronize the old current-state headers instead of leaving contradictory current-state documentation.

---

# FINAL OPERATING PRINCIPLE

Claude Code is not being asked to design CutSell from scratch.

Claude must:
- inspect live repo;
- respect current branch lineage;
- continue the exact current blocker;
- make general fixes from evidence;
- keep security and editorial QA explicit;
- update durable state as work advances;
- refuse scope drift.

Never make the user reconstruct CutSell from chat again.
