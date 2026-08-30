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

# RAW #114 OBSERVABILITY TASK — RESOLVED (RAW #116 / #117)

## What was fixed
Both halves of the RAW #114 "EXACT NEXT ACTION" were completed and pushed to
`cutsell/mobile-v1-clean`:

- §A (compact serverless return contract): already landed before this
  checkpoint as `0ae69d4` ("Expose Unified Selection reasoner state in
  serverless focused output").
- §B (workflow evidence preservation): landed as `7ad3a48` ("Preserve
  Video00 RAW evidence before the architecture gate can fail it"). The
  `Wait for unified Selection result` step no longer hard-gates on compact
  output keys before evidence is downloaded; artifact download/upload now
  run with `if: always()` and skip cleanly instead of failing when
  preview/result URIs are absent; the real architecture/Selection-lock gate
  now runs after the diagnostic artifact is uploaded.
- A follow-up, purely additive step landed as `1dbff15` ("Print unified
  Selection reasoner diagnostics in RAW CI logs"): prints
  `diagnostics.unified_selection_reasoner` (status/error or
  status/decisions) straight into CI logs, since this sandbox/session
  cannot reach the S3/blob hosts needed to pull the uploaded artifact
  directly.

Completion criteria 1-4 above are met (targeted tests pass, full suite
848 passed / 1 skipped, RAW #116 and #117 both produced a downloadable
result JSON + MP4 artifact — `cutsell-video00-unified-selection-human-review`
on runs `33268618850` and `33269880531`). Criteria 5-6 are also now met,
in the sense that Unified architecture status IS known — see below — even
though the news is not the "applied" status everyone was hoping to see.

## NEW CURRENT LIVE BLOCKER — Gemini 400 on the Unified Selection reasoner call

RAW #116 and RAW #117 (both on `cutsell/mobile-v1-clean`, head `7ad3a48`
and `1dbff15`) each reached RunPod `COMPLETED` and produced:

```
selection_reasoner_status: "provider_error_fail_open"
selection_reasoner_provider: null
selection_reasoner_model: null
```

`diagnostics.unified_selection_reasoner` (now printed directly in CI logs
by the `Print unified Selection reasoner diagnostics` step) reads:

```json
{
  "error": "HTTPError: 400 Client Error: Bad Request for url: https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
  "status": "provider_error_fail_open"
}
```

Per `apply_unified_selection_reasoner()` in
`cutsell_worker/unified_selection_reasoner.py`, this exception is caught
and the pipeline **fails open**: it returns the pre-reasoner draft
unchanged, i.e. Unified Selection never actually ran and the legacy
selection stood. That legacy selection is what `Verify frozen Selection
lock` then correctly flagged as a real regression against the frozen
Human Gold lock: 28 clips selected vs. 23 expected, several required
segments missing (sonography good-take parts, papillary cancer context,
pimples micro-takes, family context), two historically-bad takes
returned, and required orderings broken. **This is not evidence that
Unified Selection itself is editorially wrong — it has still never
successfully executed once against Video00.**

### Ranked hypotheses (none yet confirmed live)
1. **Most likely**: the Unified reasoner's `responseJsonSchema` in
   `unified_selection_google.py` enforces `minItems == maxItems ==
   candidate_count` against the *entire video's* candidate universe
   (potentially 60-100+ items for a ~6 minute source) with a 5-field,
   two-enum per-item schema. The legacy bounded-Hybrid path
   (`hybrid_google.py` / `hybrid_google_transport.py`) uses the same
   model and the same `thinkingConfig.thinkingLevel` /
   `responseJsonSchema` field names, but with a much smaller schema (2
   fields, small per-group `minItems`/`maxItems`) and has historical
   evidence of working (see `scripts/hybrid_llm_bakeoff.py`, which has
   real per-token pricing recorded for `gemini-3.5-flash-lite` and
   `gemini-3.6-flash`). A fixed-size array schema or combined request
   size at whole-video scale is the leading suspect for the 400.
2. Less likely: the model name itself is stale/wrong. Weighed down by
   the bakeoff script's pricing-table evidence above, but not
   conclusively ruled out without a live API call.
3. Unconfirmed: some other field in the request body (e.g. unbounded
   `family_index` integer, or overall payload size vs. Gemini's actual
   token/byte ceiling, as opposed to the local `max_input_tokens=20_000`
   preflight estimate which is a crude chars/3 heuristic and was NOT
   what raised the exception here — the request reached Gemini and was
   rejected there, not blocked locally).

This sandbox could not verify any of the above further at the time: `ai.google.dev`
was blocked by network egress policy here, and no GEMINI_API_KEY was
available in this session to test the live endpoint directly.

### ROOT CAUSE CONFIRMED (2026-08-29, same checkpoint)

The user independently checked Google's current docs and confirmed
`gemini-3.5-flash-lite` is a valid GA model (do not change it), and that
`responseJsonSchema` generally supports `minItems`/`maxItems` (so those
keys are not inherently invalid). Per the user's instruction, a cheap
isolation probe was built and run *without* a paid GPU RAW:
`scripts/isolate_unified_selection_schema.py`, driven by
`.github/workflows/cutsell-unified-selection-schema-isolate.yml`
(mirrors `cutsell-hybrid-llm-bakeoff.yml`'s plain-`ubuntu-latest`, no-RunPod
pattern). It reuses the real production request builders and makes 5 real,
cheap Gemini calls (run `33271212716`, cost $0.021 total). Result:

| case | N | exact `minItems==maxItems` bound? | schema richness | result |
|---|---|---|---|---|
| unified schema, bounded | 5 | yes | rich (Unified) | 200 |
| unified schema, bounded | 90 | yes | rich (Unified) | **400** |
| unified schema, unbounded | 90 | no | rich (Unified) | 200 |
| bake-off schema, unbounded | 90 | no | simple | 200 |
| bake-off schema, bounded | 90 | yes | simple | **400** |

This isolates the cause completely: schema richness is irrelevant — even
the already-proven-working bake-off schema 400s once given an exact
`minItems==maxItems` bound at N=90, and the rich Unified schema succeeds
fine once that bound is dropped. **Gemini's structured-output validator
rejects an exact-length array bound at whole-video scale** (works at 5
items, fails by 90) with this model, regardless of per-item schema shape.
Legacy bounded-Hybrid groups have always been small enough to never hit
this.

### Fix applied
Both `unified_selection_response_schema()`
(`cutsell_worker/unified_selection_google.py`) and
`editorial_response_schema()` (`cutsell_worker/hybrid_google.py`, same
latent landmine, never previously triggered because Hybrid groups are
small) no longer emit `minItems`/`maxItems` on the `decisions` array. This
loses no correctness guarantee: both call sites already raise
`ValueError` downstream in Python
(`"unified Selection ordered decision count mismatch"` /
`"hybrid provider ordered decision count mismatch"`) if the returned
decision count doesn't match the candidate count exactly — that check now
does the enforcement the wire schema used to attempt and 400 on. No
semantic Selection logic (SELECT/SWAP/DISCARD reasoning) was touched.
Targeted tests updated (`test_cutsell_hybrid_google.py`,
`test_cutsell_hybrid_google_ordered.py`,
`test_cutsell_unified_selection_reasoner.py`) plus the full suite (848
passed / 1 skipped) both green.

## RAW #118 — FIRST EVER `"applied"` UNIFIED SELECTION RUN (2026-08-29)

Confirmatory RAW #118 (run `33271431422`, head `455f8c1`) reached
`selection_reasoner_status: "applied"`, `selection_reasoner_provider:
"google"`, `selection_reasoner_model: "gemini-3.5-flash-lite"` — the
schema fix above resolved the 400 completely. This is the first time
Unified Selection has ever actually executed against Video00.
`diagnostics.unified_selection_reasoner.candidate_count: 32`,
`selected_count: 23` (the reasoner's own internal count, exactly
matching the frozen Human Gold lock's expected count).

### But the final Selection lock still failed — this is now real editorial work, not infra
`Verify frozen Selection lock` reported `actual_selected_count: 24` (not
23) and `selection_locked: false`, `historical_regression_qa_pass:
false` — the same 13 failed-check IDs as the earlier fail-open runs
(`sonography_good_take_part1_present`, `sonography_bad_take_absent`,
`papillary_cancer_preserved`, `pimples_micro_1/2/3_present`,
`pimples_bad_monolith_absent`, `pimples_later_winner_present`,
`family_context_preserved`, `pimples_micro_order`,
`sonography_good_before_diagnosis`, plus `selection_count_23`).

This is NOT the same failure as before. The ordered-text diff against
the frozen lock shows Unified Selection chose genuinely different
phrasing/takes for several story beats than the locked baseline
(`baseline_run_id: 33126865755`) — not simply "1 extra clip". At least
one selected segment reads like an incomplete/stumbled take that likely
should have been DISCARDed rather than SELECTed:

> "Tuve problemas estomacales a un tiempo en donde se me hizo una
> endoscopía y me adeagnosticaron con..." (trails off mid-word/mid-sentence)

immediately before what looks like the clean retake of the same idea
("Tuve problemas de digestión en donde me hicieron una endoscopía y
dijeron que tenía gastritis..."). If confirmed by Watch+Listen, this
would be exactly the "redundant_retry" / best-take case the editorial
contract already calls for SELECTing only the winner, not the stumble.

**Caution**: the fact that the failed-check ID list exactly matches the
earlier fail-open run's list is worth treating skeptically before
concluding the editorial quality is unchanged — these specific gold-lock
checks may simply be strict enough to fire on any rephrasing, valid or
not. Do not assume the reasoner's editorial judgment is equivalent to
the fail-open legacy fallback's without actually watching the produced
MP4 (`artifact/video00-unified-selection.mp4` in the
`cutsell-video00-unified-selection-human-review` artifact on run
`33271431422`) against Human Gold.

## Updated EXACT NEXT ACTION
1. ~~Confirm one RAW reaches `"applied"`~~ — DONE (RAW #118).
2. **Human Gold Watch+Listen is now required** before any further code
   change: download and watch `video00-unified-selection.mp4` from run
   `33271431422`'s artifact against the Human Gold authoritative video,
   and read the full `diagnostics.unified_selection_reasoner.decisions`
   table (candidate_count 32, printed in that run's CI log) per handoff
   item F's process (family_index / previous_bucket / model_action /
   effective_action / relation / confidence / reason_code) before
   concluding anything is a real regression.
3. Do NOT reflexively add a new guard or tune the editorial prompt based
   on the automated lock/regression-QA failure alone — per the editorial
   rules, human judgment on the actual video is required first. If
   Watch+Listen confirms the stumble-take-selected issue above, the
   likely fix is prompt/reason_code guidance in
   `unified_selection_google.py`'s editorial contract (SELECT only the
   winner of a retry family), not a new post-hoc filter — but that is a
   semantic Selection change and needs the same care as any other.
4. Once Selection quality is confirmed, `benchmarks/video00_selection_lock.json`
   itself may need to be re-baselined against the new Unified-Selection
   output rather than the old pre-reasoner lock, if the new phrasing
   choices are judged equal-or-better than the original — that decision
   belongs to Human Gold review, not automated diffing.

---

# PROVIDER RELIABILITY — RAW #119/#120 (2026-08-29/30)

RAW #119 saw `selection_reasoner_status` regress from `"applied"` (RAW
#118) back to `provider_error_fail_open`, with `diagnostics.unified
_selection_reasoner.error` reading a bare `JSONDecodeError` (no
`finishReason`, no distinguishing whether it was truncation or something
else). Diagnosed the provider layer (`unified_selection_google.py`) per
all six requested angles (finish reason, output token budget, schema
compatibility, parser assumptions, retry policy, determinism) and fixed:

- `output_token_reserve()` replaces the old flat `36 tokens/candidate`
  guess with a true upper bound derived from the schema's actual longest
  enum values (~44 tokens/candidate) — the old formula under-provisioned
  by ~23% at Video00's real scale (32 candidates).
- `parse_unified_selection_response()` now names `finishReason` in every
  error and raises a new `UnifiedSelectionUnreliableResponseError`
  instead of a bare `ValueError`/`JSONDecodeError`, for every shape that
  must never be treated as a complete result.
- One retry (`max_retries: int = 1`) with a 1.5x token budget bump on
  the second attempt, plus a fix for a pre-existing ledger-reservation
  leak on failed attempts.
- 14 new targeted tests (`tests/test_cutsell_unified_selection_google.py`)
  — this transport had zero prior direct unit coverage.

No Selection/Boundary code touched; no Video00-specific guard added.
Full suite 869 passed / 1 skipped. Pushed `0666ee1`.

## RAW #120 result: fix confirmed working, but surfaced a NEW distinct failure mode

RAW #120 (run `33281452850`) still did not reach `"applied"`. The
diagnostics now read clearly (proving the observability half of the fix
works exactly as intended):

```
"error": "UnifiedSelectionUnreliableResponseError: unified Selection
ordered decision count mismatch (expected 32, got 31,
finishReason='STOP')",
"status": "provider_error_fail_open"
```

This is a **different** failure than RAW #119's truncation:
`finishReason='STOP'` means the model completed normally — it was not
cut off, and the retry (which bumps the token budget) would not help
here since the problem is not a lack of headroom. The model simply
returned one fewer decision object than there were candidates (31 vs
32), most likely because `unified_selection_response_schema()` no
longer encodes an exact `minItems`/`maxItems` array bound (removed in
the RAW #118 fix, because that exact bound is what caused the earlier
400 at this scale). Removing that bound fixed the 400 but also removed
the one thing that made the schema *force* exactly one decision per
candidate — the model is now free to under-count without any
schema-level pressure not to.

Because the run failed open, `diagnostics.unified_selection_reasoner`
only ever contains `{status, error}` (see
`apply_unified_selection_reasoner()`'s except-branch) — the actual
31-item decisions array was never captured anywhere observable, so
which specific candidate got dropped or merged is not yet known.

**Do not extrapolate a fix from this note alone.** This needs the same
evidence-first treatment as the earlier 400: diagnosing *why* the model
undercounts by exactly one (a merged pair of candidates? one skipped?)
requires either additional response logging before the count check, or
a cheap non-RAW isolation probe (mirroring
`scripts/isolate_unified_selection_schema.py`) that captures the full
returned decisions array on a short-count response rather than just the
count. Two schema options worth weighing, once the actual cause is
known, both requiring the same care as any other provider-layer change:
requiring the model to echo each candidate's index/id (so a short
response can be positionally reconciled instead of zipped), or
re-encoding cardinality some other way that does not reproduce the
RAW #114-117 400 at scale. Whichever fix is chosen, it is a provider
transport/schema question, not a Selection semantics question, and it
should get the same one-RAW confirmatory validation this and the RAW
#118 fix did.

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
