# CutSell Clean Cut Core V1 -- Canonical System Map and AS-IS / SHOULD-BE / GAP / Duplication Audit (D-096)

**Status: CANONICAL AUDIT (read-only; no engine change was made for this document).**
**Date:** 2026-09-06. **Audited head:** `fdce3f79d9ba13ceba0a6faf3a2be91a7205240e`
(`feature/runpod-pod-on-demand`). **Audited execution:** Video00 Modal RAW run
`34008386434` (worker built from `8d2bdb988fcd709653a54313d9496193e65f695b`; the
`cutsell_worker` package is byte-identical to the audited head -- see Part 1).

This document is the single source of truth for WHAT HAS BEEN BUILT, WHAT IS
ACTUALLY ACTIVE, WHAT IS DUPLICATED OR CONTRADICTORY, and WHAT IS MISSING. It is
organised in the twelve parts the Product Owner directive requested and keeps
three layers strictly apart:

- **HISTORICAL DECISIONS** -- every `D-xxx` in `docs/CUTSELL_DECISIONS.md` stays
  the record of why something exists. Nothing here rewrites that history.
- **CURRENT ACTIVE ARCHITECTURE** -- what the code on this head executes on the
  Video00 RAW path, proven by the run's own `active_path_identity`, diagnostics
  and the four-way quality ladder. Marked AS-IS.
- **PROPOSED TARGET ARCHITECTURE** -- Part 11. A proposal only. Nothing in it is
  implemented, and nothing may be implemented from it without Product Owner
  approval (Part 12).

Binding constraints preserved by this audit (all still in force):
1. every canonical component accepted before the Cut.ai-parity directive
   (attempt reconstruction, recording-process cleanup, grouping, Best Take,
   claim protections, composites, editorial-slot resolution, Selection/Boundary
   ownership, Freeze, CanonicalEditPlan/review/repair loop, Boundary, render/QC,
   regression coverage) remains in force unless the diagnostic proves it wrong;
2. the ladder `RAW -> CUT.AI PARITY -> HUMAN GOLD PARITY -> technical post-render
   QC -> perceptual SYSTEM WATCH+LISTEN -> HUMAN WATCH+LISTEN` (D-095 + addendum);
3. "re-base" changed QA/validation priority only, never the implementation;
4. a real BLOCKING perceptual SYSTEM WATCH+LISTEN gate is required after the
   technical post-render QC and before Human Watch+Listen -- it is MISSING today;
5. Human Gold = ultimate editorial oracle; Cut.ai = intermediate commercial
   baseline oracle; both QA-ONLY, never fed to Selection/Boundary/BestTake/LLM;
6. PR #25 stays OPEN / DRAFT / UNMERGED; `main` untouched.

---

## PART 1 -- REAL CURRENT STATE

| Item | Value | How it was established |
|---|---|---|
| CURRENT BRANCH | `feature/runpod-pod-on-demand` (working branch; `cutsell/mobile-v1-clean` = PR #25 is WATCHED read-only and never pushed to, its push trigger fires a paid RunPod RAW) | `git branch --show-current` |
| CURRENT HEAD SHA | `fdce3f79d9ba13ceba0a6faf3a2be91a7205240e` (== `origin/feature/runpod-pod-on-demand`) | `git rev-parse HEAD`, `git fetch` |
| PR #25 STATUS | OPEN, DRAFT, not merged, `mergeable_state: clean`; head `9a277dbbc2bd69d1efd882f0d029e5536e8b81b8`; base `main`; 1780 commits / 690 files | GitHub API |
| PR #25 head vs this branch | `9a277db` is the merge-base of HEAD and `origin/cutsell/mobile-v1-clean`; zero commits on PR #25 that this branch lacks (editorial-slot D-042 line, Cut.ai registration, oracle hierarchy docs are all merged here) | `git merge-base`, `git log HEAD..origin/cutsell/mobile-v1-clean` = empty |
| MAIN SHA | `2fb13e5aa228e8e525b942a9b49182032b797e61` (untouched) | `git rev-parse origin/main` |
| LATEST VALID VIDEO00 RAW RUN | GitHub Actions run `34008386434` -- "CutSell Video00 Modal RAW" #43, `workflow_dispatch`, head `8d2bdb9`, 2026-09-06 03:11-03:24 UTC, ONE authorized L4 run (`hybrid_max_edit_usd=0.02`, F3b bridge OFF). Job conclusion `failure` ONLY because two QA-only validators fail by design on a non-Gold candidate: "Verify frozen Selection lock" (D-032 ordered alignment against the Gold-derived lock) and "Verify Human Gold regression QA" (17 pass / 2 fail: `pimples_micro_2_present`, `pimples_micro_order`). The engine, render, QC, identity, ladder and teardown steps all succeeded. | run + job listing, log tail |
| LATEST VALID VIDEO00 ARTIFACT | `benchmark_id = video00-modal-34008386434-1`; full result `s3://<bucket>/cutsell/serverless/video00-modal-34008386434-1/result.json`; video `.../diagnostic-invalidated-preview.mp4` (delivery gate: `NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW`, so `preview_uri` is null by design, D-036). GitHub artifacts: `cutsell-video00-modal-diagnostics`, validator reports incl. `video00-modal-active-path-identity.json`, ladder JSON/MD/CSV. Four-way ladder re-run on it: CPU run `34009154031` (no paid compute). | workflow steps, ladder log |
| WORKER IMAGE / DIGEST USED | Modal container, GPU `L4`, image built by `modal_video00_full_benchmark.py`: `modal.Image.from_registry("madiator2011/better-pytorch:cuda12.4-torch2.6.0")` + apt packages + `requirements` + `runpod>=1.7,<2`; **no digest pin** (tag only) -- the base image is mutable by tag; the `cutsell_worker` package is MOUNTED from the exact checked-out head, not baked into an image, so engine identity is the checkout, not the image. Python 3.10.12 inside the worker. | `modal_gpu_config.py`, `modal_video00_full_benchmark.py`, identity block |
| RUNPOD ENDPOINT / TEMPLATE USED | None for execution. RunPod template `EditDNA-Worker-2` is read ONLY as the env-var source of truth (S3 keys, Gemini key, flags), overlaid by the workflow with `CUTSELL_UNIFIED_REALIZATION_RESOLVER=AUTHORITATIVE`, the hybrid-semantic overlay, `CUTSELL_HYBRID_MAX_EDIT_USD=0.02`, `CUTSELL_BUILD_GIT_SHA=<sha>`. The RunPod Serverless endpoint and the RunPod Pod paths (D-041/D-042 infra) are alternate backends that did NOT run this RAW. | workflow "Load base RunPod template (env source of truth only)" |
| DID THAT RAW USE THE CURRENT HEAD? | **YES for the engine, proven in-worker:** `active_path_identity.build_git_sha = 8d2bdb9…`, `package.sha256 = fe54d094c069…` over 221 modules / 2,407,635 bytes, `resolver_mode_env = AUTHORITATIVE`, `hybrid_max_edit_usd_env = 0.02`; workflow step "Verify active-path identity" compared it with the checkout and passed. `git diff --stat 8d2bdb9 fdce3f7` = one file, `.github/workflows/cutsell-video00-quality-ladder.yml` (+5 lines, QA tooling). **The engine package on HEAD is identical to the one that produced the audited video.** Component markers: 18/20 present; absent: `DeterministicBestTakeAuthority` and `EditorialSlotResolution` -- both are marker defects, not inactivity (Part 3, rows S-14 and S-9). | identity JSON in run 34009154031 log |

**CURRENT CODE HEAD vs HEAD THAT PRODUCED THE LAST VIDEO THE USER WATCHED.**
The last video this audit analysed is the diagnostic-invalidated preview of run
`34008386434` (engine = HEAD). Which artifact the Product Owner last watched by
eye is not recorded in the repository; if it was the run `33995806350` preview
(head `df3946e`), that worker predates D-095.2 (audio dead-air evidence) and the
D-094.3 F8/F9/F13/F14 fixes and the D-042 editorial-slot line, i.e. it is NOT
the current engine. Every future statement about "the video" must quote the
benchmark id and the identity block; D-095's addendum made this mandatory.

**Configured brain on the RAW path (`brain_runtime.build_brain_runtime`, backend
`runpod_local`)**: `whole_video_provider = RunPodLocalWholeVideoProvider` (local,
no paid whole-video call), `visual_provider = None`, `take_grouping_provider =
None` (deterministic lexical grouping), `take_judge_provider =
HybridTakeJudgeProvider(editorial_judge=None)` -> local DeliveryScorer baseline
(`watch_listen_baseline`), `clean_cut_provider = None`, `composer/draft_review =
None` (clean_cut mode). Paid Gemini calls on the path: EditorialJudge windows
(hybrid_session_cleanup, D-081 evidence-first), SemanticEquivalenceArbiter
(grouping + StoryValidator residuals, D-042 policy injected), ClaimEquivalence-
Arbiter (ambiguous claim coverage). Unified Selection reasoner: disabled
(`clean_cut_core_v1_enabled=True`).

---

## PART 2 -- COMPLETE CANONICAL PIPELINE MAP (INTENDED vs ACTUAL)

Legend: `[AS-IS]` = executes on the audited RAW path (proven by stage_status /
diagnostics / identity markers); `[COND]` = executes only under a flag/provider
that was ON in the RAW; `[MISSING]` = the conceptual stage has no implementation
on the active path (a Protocol, a doc, or a dormant module is NOT an
implementation); `[LEGACY]` = present in code, not on the active path.

Entry chain (actual): `serverless_handler.handler` -> `run_op("focused")` ->
`_focused` -> `universal_clean_cut_validation.run_single_universal_clean_cut_validation`
-> `brain_runtime.build_brain_runtime` -> `universal_clean_cut.process_universal_clean_cut_sources`
-> `flow_b.process_local_sources` -> `pipeline.build_flow_b_draft` (wrapped at
import time by 14 monkeypatch layers, see 2.10) -> Selection authorities (2.11)
-> Freeze -> Boundary -> `render_plan.build_render_plan` ->
`live_render_qc.render_with_post_render_qc` -> delivery gate -> S3 upload.

```
RAW INPUT (S3 key -> local file)                                          [AS-IS] serverless_handler / validation harness
  |
  v
2.1  media ingest / probe  (media_probe.probe_media: duration, fps, audio)  [AS-IS] flow_b
2.2  usage guard (check_processing_allowance)                                [AS-IS] flow_b
  |
  v
2.3  ASR / transcript / word timing (FasterWhisper, language hint;           [AS-IS] asr.py via brain
     canonical ASR evidence + content hashes, D-052/D-055: diagnostics only)
  |
  v
2.4  visual / audio / performance signals
     - local performance timelines @12 fps -> hand_motion_reset (218 events   [AS-IS] local_performance.analyze_local_performance
       in the RAW), body_reset (11), facial_expression_shift (21) CANDIDATES
     - whole-video context (RunPodLocalWholeVideoProvider: local, no LLM)      [AS-IS] whole_video_analysis (local provider)
     - merge local events into context                                         [AS-IS] local_performance.merge_local_events_into_context
     - audio dead-air intervals (ffmpeg silencedetect -35 dB >= 0.6 s;         [AS-IS] audio_silence (D-095.2); 56 intervals in the RAW
       published as audio_silence_interval events)
     - word silence gaps (ASR-timing derived)                                  [AS-IS] silence_analysis.word_silence_gaps
  |
  v
2.5  take segmentation (speech units at >= 0.75 s word gaps, boundary        [AS-IS] take_segmentation.segment_takes
     fragment repair strict 0.16 s / bridge 0.65 s)
2.6  performance confirmation (retry_setup / wrong_take confirmed events)    [AS-IS] performance_confirmation
2.7  per-take visual provider                                                 [COND, OFF] visual_provider=None in the RAW
     local performance fusion into takes (MediaSignals)                        [AS-IS] local_performance.apply_local_performance_to_takes
  |
  v
2.8  ATTEMPT RECONSTRUCTION (ASR chunks -> delivery attempts; never deletes;  [AS-IS] attempt_reconstruction.reconstruct_delivery_attempts
     + D-046B preserved borderline subspans as EXTRA candidates)                       (+ attempt_boundary_integrity wrapper)
  |
  v
2.9  semantic role labels (SemanticProvider)                                  [BYPASSED] clean_cut mode: "not_requested_clean_cut"
  |
  v
2.10 build_flow_b_draft (pipeline.py) -- the DRAFT-LEVEL SELECTION
     a. mint realization_id on the complete pool (D-050D1)                     [AS-IS] canonical_identity.mint_realization_id
     b. Pass 1 RECORDING PROCESS REMOVAL / deterministic clean cut             [AS-IS] clean_cut.apply_clean_cut wrapped by 22 import-time hooks
        (apply_clean_cut + recording_process_context, dangling_delivery,               (see Part 3 table, rows P-1..P-22)
         internal_self_correction, internal_retake_winner, internal_repeat_trim,
         recording_breaks, restart_questions, frustrated_restart, micro_restart,
         product_handling_failure, micro_self_talk, orphan_retry_cleanup,
         incomplete_retry_suffix, interstitial_retry_debris, trailing_retry_restart,
         merged_self_review, word_search_attempts, recording_meta_continuation,
         story_coverage_guard, superseded_attempt_cleanup, lexical_self_correction)
        + provider clean-cut judge                                              [COND, OFF] clean_cut_provider=None
     c. Pass 2 COMPOSITE RESOLVER chain (composite_resolver.apply_composite_    [AS-IS] composite_resolver -> hybrid_session_cleanup
        resolution): Gemini EditorialJudge windows over creator mini-sessions          (EditorialJudge ON: hybrid_editorial=provider_complete)
        -> labels winner/alternate/failed/bts per candidate; D-081: mechanical
        certainty deletes early, semantic judgment only RECOMMENDS delete;
        19 wrapped hooks (semantic_fragment_guard ... hybrid_semantic_conflict_
        arbitration) restore / rescue / mark composites; composite_split_ids
     d. Pass 3 IDEA CLUSTERING / RETRY FAMILY FORMATION                         [AS-IS]
        - session partition + lexical retry grouping (safe_group_takes_by_             session_boundaries + take_grouping_provider.safe_group_takes
          sessions; wrapped by local_retry_grouping, retry_group_integrity,             (+5 grouping hooks)
          final_sibling_grouping, session_grouping_bridge, global_session_sibling_bridge)
        - apply_composite_group_split (composites forced into singleton groups)
        - SEMANTIC ARBITER tier: reconcile_semantic_idea_equivalence (Gemini,          [AS-IS] semantic_idea_equivalence(_google); D-042 policy
          priority-ranked pair budget, coverage-first order D-042)                             injected by editorial_slot_resolution_install
        - split_incohesive_retry_groups (D-058/D-085 cohesion safety, F3b OFF)   [AS-IS] take_grouping_provider
     e. EDITORIAL FUNCTION / SLOT UNDERSTANDING                                  [PARTIAL] exists ONLY as prompt policy text inside the
                                                                                         semantic-equivalence request (D-042); no engine
                                                                                         representation of "slot" exists anywhere (no field,
                                                                                         no diagnostics, no validator) -> effectively MISSING
                                                                                         as an engine stage
     f. CANDIDATE GENERATION = the retry-family members (no separate stage)      [AS-IS] (implicit)
     g. DELIVERY / PERFORMANCE SCORING: DeliveryScorer per family                 [AS-IS] take_judge.rank_takes via safe_rank_takes
        (0.70 completeness + 0.30 duration fit, fragment penalties,                       (`watch_listen_baseline`; provider judge OFF)
         handling-failure penalty from MediaSignals)
     h. BEST TAKE RESOLUTION (draft level): family-scoped hybrid labels (F8)      [AS-IS] pipeline.family_scoped_semantic_decisions +
        -> _semantic_best_take ladder: single winner >= 0.85 | D-081 delete-              pipeline._semantic_best_take (wrapped by
        recommended | completeness | CRITICAL_COVERAGE_DOMINANCE | unique-fact             semantic_best_take_integrity)
        asymmetry -> unresolved | delivery tie-break -> TakeGroup.selected_clip_id
     i. Pass 4 physical edge trims BEFORE Selection is final                       [AS-IS] temporal_editing.refine_takes_with_temporal_context
        (refine_takes_with_temporal_context + delivery_edge_trim, interior_                (+5 edge hooks)  <-- physical trimming pre-Freeze
         performance_break, recording_suffix_trim, script_consult_trim,
         temporal_word_boundary_integrity)
     j. compose_selected: one winner per family + every ungrouped take            [AS-IS] composer.compose_selected (+ selection_integrity)
     k. composer / draft review providers                                         [OFF]  clean_cut mode
     l. DraftTimeline (selected / alternates / discarded + diagnostics)           [AS-IS]
     m. apply_composite_family_stabilization (CompositeResolver step 16)          [AS-IS] composite_resolver
     n. import-time wrappers AROUND build_flow_b_draft, applied in install order   [AS-IS] final_draft_retry_integrity, selected_failed_bridge_
        AFTER the inner draft exists (each is a further Selection-membership              integrity, round8_retry_reconciliation (+round9),
        edit): retry integrity, failed-bridge collapse, round8/9/11 reconciliation,        round11_semantic_retry_cleanup, short_bts_process_
        short BTS cleanup, incomplete-bridge authority, internal retake trim,              cleanup, post_selection_incomplete_bridge_authority,
        FINAL SELECTION RETRY ARBITER, then a PREMATURE FREEZE (install_selection_         post_selection_internal_retake_trim, final_selection_
        freeze), then Boundary-flavoured passes: edge-only boundary, INTERIOR GAP           retry_arbiter, selection_boundary_contract.install_
        TRIM (word-gap / visual reset / D-095.2 audio silence), continuity                 selection_freeze, post_selection_edge_only_boundary,
        coalescer, then the boundary invariant; audio_boundary_completion wraps            post_selection_interior_gap_trim, post_selection_
        process_local_sources itself                                                       continuity_coalescer, audio_boundary_completion,
                                                                                           install_boundary_selection_invariant
  |
  v
2.11 universal_clean_cut.process_universal_clean_cut_sources -- the AUTHORITY-LEVEL SELECTION
     a. apply_selection_phase_authority: complementary_family_stabilizer (AGAIN), [AS-IS] selection_phase_authority
        internal retake trim (AGAIN), final_selection_retry_arbiter (AGAIN),
        swap-alternate preservation (writes `alternates`), final retry guards
        (failed / redundant-bridge / short-alternate removal)
     b. apply_selection_conflicted_bridge_guard (selected -> alternates)          [AS-IS] selection_conflicted_bridge_guard
     c. DETERMINISTIC BEST TAKE AUTHORITY (clear local winner, gap >= 0.30;       [AS-IS, NO-OP in the RAW] deterministic_best_take_authority
        swap_enabled=False)                                                               (no family had a 0.30 gap; writes no diagnostics when
                                                                                          it moves nothing -> identity marker "absent")
     d. CRITICAL-CLAIM / SEMANTIC PROTECTION: apply_claim_coverage_best_take       [AS-IS] claim_coverage_best_take (D-038/D-040/D-048/D-059/
        (winner override when another member covers every CRITICAL claim;                 D-063/D-066); ClaimEquivalenceArbiter ON
         CRITICAL_COVERAGE_DOMINANCE on multi-selected families; composites)
     e. STORY VALIDATION pass 1 (legacy evidence)                                 [AS-IS] final_story_coherence_validation (arbiters ON)
     f. CanonicalEditPlan + FinalEditReviewer + bounded repair loop pass 1        [AS-IS] repair_loop / canonical_edit_plan / final_edit_reviewer
     g. SEMANTIC LEDGER (read-only reconstruction of every decision)              [AS-IS] semantic_ledger
     h. UNIFIED REALIZATION RESOLVER, shadow report                               [AS-IS] realization_resolver.resolve_realizations_shadow
     i. UNIFIED REALIZATION RESOLVER, AUTHORITATIVE APPLICATION (the ONE           [AS-IS] realization_resolver.apply_authoritative_realization_
        semantic Selection authority): per idea RESOLVED_WINNER / RESOLVED_               resolution; status SEMANTICALLY_RESOLVED in the RAW
        COMPOSITE / REVIEW_REQUIRED; moves clips selected/discarded/alternates;
        RESTORES losers into `selected` (story placement units, D-089)
        + D-092 fold alternates -> discarded (KEEP/DISCARD only)
     j. authoritative plan source, semantic preservation proofs (PRE_GROUP,       [AS-IS] canonical_edit_plan / realization_resolver
        intra-idea), effective importance index (D-076/D-079/D-087/D-089)
     k. STORY VALIDATION pass 2 -- VALIDATION ONLY (D-090 signature invariant)    [AS-IS] post_authority_validation + apply_post_authority_story_validation
     l. CanonicalEditPlan + FinalEditReviewer + repair loop pass 2 (order only)   [AS-IS] repair_loop (authoritative_source)
     m. post-authority integrity record -> freeze_blocked gate                    [AS-IS] universal_clean_cut
  |
  v
2.12 complete-idea boundary recovery (source transcript envelope expansion,      [AS-IS] final_boundary_authority.enforce_complete_idea_
     word lock, same-source overlap reconciliation, preserve polished gaps)               boundaries (+ terminal_sentence_boundary_guard,
                                                                                           boundary_retry_tail_guard wrappers)  <-- runs BEFORE Freeze
2.13 SELECTION FREEZE (plan id/version/semantic hash; token-stream digest)        [AS-IS] selection_boundary_contract.freeze_selection_contract
  |
  v
2.14 BOUNDARY ENGINE after Freeze: polish_human_boundaries_v5 (micro visual-     [AS-IS] human_boundary_polish_v5 (1 split in the RAW)
     reset word gaps only) -> enforce_selection_contract (token stream unchanged)         selection_boundary_contract.enforce_selection_contract
2.15 physical micro-trimming / continuity: NOTHING further before render          [AS-IS] (the interior/edge/continuity passes ran in 2.10n,
                                                                                           i.e. before the authority-level Selection, not here)
  |
  v
2.16 RENDER PLAN (selected only; contiguous coalescing)                           [AS-IS] render_plan.build_render_plan
2.17 RENDERER (per-segment ffmpeg: tighten_trailing_silence >= 0.28 s tail,       [AS-IS] render.render_preview inside live_render_qc.
     volume filter, 12 ms join fades, concat)                                             render_with_post_render_qc
  |
  v
2.18 TECHNICAL POST-RENDER QC (decoded MP4): structural (plan coverage,           [AS-IS, BLOCKING] post_render_watch_listen_qc structural checks +
     sequence, duplicates) + media (decode integrity, accidental silence                 post_render_media_qc; bounded physical repair loop
     >= 1.2 s @ -35 dB, frozen frames, black frames, audio discontinuity at              (edge trims only); delivery gate `deliverable`
     joins) + bounded Boundary-only repair (<= max attempts) -> PASS /
     PHYSICAL_FAIL_* / SEMANTIC_MISMATCH_INVALIDATED / NEEDS_HUMAN_REVIEW
     RAW 34008386434: NEEDS_HUMAN_REVIEW (LINGERING_ACCIDENTAL_SILENCE 2.32 s
     at render 71.6-74.0 unrepairable; 6 ABRUPT_AUDIO_DISCONTINUITY joins)
2.19 auto speech-safe visual microtrim (post-render, only if deliverable)         [COND, skipped] speech_visual_microtrim (candidate not deliverable)
  |
  v
2.20 PERCEPTUAL SYSTEM WATCH + LISTEN (facial/body/gesture continuity, reset      [MISSING] post_render_watch_listen_qc declares the Protocol
     debris, entry/exit quality, clipped phonemes, breath cuts, cadence,                  PostRenderWatchListenQCProvider and the finding kinds
     visual jumps, repeated audience-facing content, process residue)                     CLIPPED_WORD, UNSAFE_WORD_BOUNDARY, AWKWARD_PHYSICAL_CUT,
                                                                                          RESET_DEBRIS, AWKWARD_POST_LINE_EXPRESSION, AV_SYNC_DRIFT,
                                                                                          FRAMING_INTEGRITY -- NO producer exists for any of them
  |
  v
2.21 HUMAN WATCH + LISTEN                                                          [MANUAL] no gate object; QA-only helpers exist (Gold 18-check
                                                                                          manifest, selection lock, four-way ladder) -- none blocks
  |
  v
FINAL DELIVERABLE: `preview.mp4` only when the technical gate PASSes; otherwise    [AS-IS] serverless_handler._focused (D-036)
`diagnostic-invalidated-preview.mp4` + null preview_uri
```

**Additional actual stages not in the conceptual list:** realization identity
minting (2.10a), the pre-Selection physical edge trims (2.10i), the fourteen
import-time draft wrappers (2.10n) that re-edit membership after the draft is
built, the premature freeze inside 2.10n (superseded by the authority-level
freeze at 2.13 -- D-025 makes the record consistent), the double execution of
complementary-family stabilizer / internal-retake trim / final-selection retry
arbiter (2.10n then 2.11a), the two StoryValidator passes and two plan/review/
repair passes (legacy evidence + authoritative), the semantic ledger, the
shadow resolver, and the post-authority integrity signature.

**Conceptual stages that do NOT exist as engine stages:** Editorial Function /
Slot understanding (prompt policy only), "complete-realization competition"
and "minimum-sufficient-editorial-set decision" as explicit stages (see Part
7: the resolver's decision model is CRITICAL-coverage maximisation, the
opposite objective), perceptual System Watch+Listen (Protocol only), Human
Watch+Listen gate (manual).

---

## PART 3 -- AUTHORITY TABLE (every stage / module / authority on the path)

Column legend. **Active**: PROD = active in production Clean Cut Core V1 path /
RAW = active in the audited Video00 RAW (proven by diagnostics or identity
markers) -- values YES / NO / COND / NO-OP (ran, changed nothing) / UNKNOWN.
**Caps** (each Y or N, in this order): **M** can change semantic membership ·
**F** can change retry family · **W** can change the Best-Take winner · **R**
can restore discarded content · **C** can create/preserve composites · **S** can
change a clip's source range · **B** boundaries-only (never membership) · **D**
can block delivery. **Tests** = number of test files under `tests/` that
reference the module (262 `test_cutsell_*` files, 2100 tests passing at
D-095.2). **STATUS** uses the directive vocabulary; KEEP is never granted
merely because tests exist -- it requires proof of presence on the RAW path
AND one clear responsibility.

### 3A. Perception and reconstruction (no editorial authority)

| # | STAGE / module | Purpose -- canonical responsibility | Files / primary functions | Called by | Input -> Output | Active PROD / RAW | Caps M F W R C S B D | Runs after it that can modify/undo | Tests | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|
| A-1 | Media ingest / probe | duration, fps, has_audio for every source | `media_probe.probe_media` | `flow_b.process_local_sources` | file -> SourceAsset metadata | YES / YES | N N N N N N N N | none | 2 | KEEP |
| A-2 | Usage guard | monthly minutes allowance | `usage_limits.check_processing_allowance` | flow_b | durations -> allowed/denied | YES / YES | N N N N N N N D | none | 2 | KEEP |
| A-3 | ASR / word timing | transcript segments + words | `asr.FasterWhisperASR.transcribe` (brain provider) | flow_b | media -> TranscriptSegment[] | YES / YES | N N N N N N N N | every downstream stage trusts these timings; D-053 proved ASR is not deterministic run-to-run | 7 | KEEP (known variance, D-053/D-055) |
| A-4 | Canonical ASR evidence | content / equivalence hashes (D-052/D-055) | `canonical_asr_evidence.build_canonical_asr_evidence` | flow_b | transcript -> diagnostics | YES / YES | N N N N N N N N | none (observability) | 3 | KEEP (observability) |
| A-5 | Local performance timelines | 12 fps face/hand/body candidate events | `local_performance.analyze_local_performance`, `merge_local_events_into_context`, `apply_local_performance_to_takes` | flow_b | media -> TemporalEvents + MediaSignals | YES / YES (218 hand, 11 body, 21 facial) | N N N N N N N N | consumed as evidence by attempt reconstruction, clean-cut hooks, interior trims, polish v5 | 4 | KEEP |
| A-6 | Whole-video context | global intent/story context (+events) | `whole_video_analysis.safe_whole_video_analyze` with `RunPodLocalWholeVideoProvider` | flow_b | frames+transcript -> WholeVideoContext | YES / YES (local provider) | N N N N N N N N | none | 37 | KEEP (local; the Gemini whole-video provider is not on this path) |
| A-7 | Audio dead-air evidence (D-095.2) | ffmpeg silencedetect on SOURCE, -35 dB >= 0.6 s -> `audio_silence_interval` events | `audio_silence.detect_audio_silence_intervals`, `audio_silence_events`, `merge_audio_silence_into_context` | flow_b (after A-5 merge) | source -> 56 intervals (RAW) | YES / YES (`stage_status.audio_silence = complete`) | N N N N N N N N | consumed only by P-33 interior-gap trim | 1 | WIRED-BUT-NOT-PROVEN-IN-RAW for its purpose: ran, produced 1 split, but the 2.32 s silence QC later found at render 71.6-74.0 (raw ~138.9-141.2, inside `clip_f8a423f3`) produced neither a split nor a rejection trace -> INVESTIGATE (Part 9 G-14) |
| A-8 | Word silence gaps | ASR-timing pauses | `silence_analysis.word_silence_gaps` | flow_b | words -> gaps | YES / YES | N N N N N N N N | none | 1 | KEEP (evidence) |
| A-9 | Take segmentation | speech units (>= 0.75 s gaps), boundary-fragment repair (strict 0.16 s / bridge 0.65 s) | `take_segmentation.segment_takes`, `_speech_units`, `_repair_boundary_fragments` | flow_b | transcript -> CandidateTake[] | YES / YES | N N N N N N N N (defines candidate granularity; a wrong split IS a membership risk downstream: a lone "No" became its own candidate in the RAW) | attempt reconstruction may re-merge; D-046B subspans may re-split | 8 | KEEP (D-095.3 fix proposed in Appendix A, NOT applied) |
| A-10 | Performance confirmation | confirm retry_setup / wrong_take events against takes | `performance_confirmation.confirm_local_performance_events` | flow_b | takes+timelines -> confirmed events | YES / YES | N N N N N N N N | none | 1 | KEEP |
| A-11 | Per-take visual provider | frame-level MediaSignals from a vision model | `visual_analysis.safe_visual_analyze` | flow_b | frames -> observations | COND / NO (`visual_provider=None`) | N N N N N N N N | -- | 2 | DEAD/INACTIVE on the RAW path (MediaSignals come only from A-5) |
| A-12 | AttemptReconstructor | ASR chunks -> complete delivery attempts; never deletes | `attempt_reconstruction.reconstruct_delivery_attempts` (+ `attempt_boundary_integrity` wrapper), `preserved_subspan_candidates` (D-046B) | flow_b | takes+context -> attempts (+extra subspan candidates) | YES / YES | N N N N N N N N (but decides candidate shape: merges/splits) | every Selection stage | 9 | KEEP |
| A-13 | Semantic role labelling | hook/problem/CTA roles | `providers.safe_semantic_classify` | flow_b (full mode only) | -- | NO / NO (clean_cut mode) | N N N N N N N N | -- | -- | LEGACY (full mode) |
| A-14 | Realization identity | mint `realization_id` on the complete pool (D-050D1) | `canonical_identity.mint_realization_id`, `build_identity_chain_diagnostics` | pipeline | takes -> takes with identity | YES / YES | N N N N N N N N | none (identity is immutable) | 6 | KEEP |

### 3B. Draft-level Selection (`pipeline.build_flow_b_draft` and its wrappers)

| # | STAGE / module | Purpose -- canonical responsibility | Files / primary functions | Called by | Input -> Output | Active PROD / RAW | Caps M F W R C S B D | Runs after it that can modify/undo | Tests | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|
| P-1 | Deterministic Clean Cut (Pass 1) | RecordingProcessRemoval backbone: obvious recording garbage | `clean_cut.apply_clean_cut` | pipeline | takes+context -> kept / discarded + decisions | YES / YES (`clean_cut = context_aware_deterministic_complete`) | Y N N N N N N N | everything below; CompositeResolver and Resolver can restore | 13 | KEEP |
| P-2..P-22 | 21 import-time hooks wrapping `apply_clean_cut` (recording_process_context, recording_suffix_trim*, script_consult_trim*, delivery_edge_trim*, interior_performance_break*, dangling_delivery, internal_self_correction, internal_retake_winner, internal_repeat_trim, recording_breaks, restart_questions, frustrated_restart, micro_restart_cleanup, product_handling_failure, micro_self_talk, orphan_retry_cleanup, incomplete_retry_suffix, interstitial_retry_debris, trailing_retry_restart, merged_self_review, word_search_attempts, recording_meta_continuation, story_coverage_guard, superseded_attempt_cleanup, lexical_self_correction) -- (*) the four starred ones wrap `refine_takes_with_temporal_context` instead | each: one narrow recording-process / failed-fragment heuristic (see docstrings in `cutsell_worker/__init__.py` order) | `cutsell_worker/__init__.py` installs at package import; each patches `clean_cut.apply_clean_cut` (or `temporal_editing.refine_takes_with_temporal_context`) by module attribute | import time | kept/discarded -> kept/discarded (delete / trim / split) | YES / UNKNOWN per hook (only `clean_cut_decisions` reasons in the result reveal which fired; no per-hook activity marker exists) | Y N N N N Y N N (most delete or trim; `story_coverage_guard` restores) | every later stage | 1-2 each (avg) | UNCERTAIN as a set: individually KEEP-able heuristics, collectively an unowned RecordingProcessRemoval authority with 22 partial owners and no single decision record (Part 10 D-1) |
| P-23 | Clean-cut provider judge | LLM clean-cut verdicts | `clean_cut_provider.safe_clean_cut_judge`, `apply_provider_judgements` | pipeline | kept -> kept/discarded + mixed trims | COND / NO (`clean_cut_provider=None`) | Y N N N N Y N N | -- | -- | DEAD/INACTIVE on the RAW path |
| P-24 | CompositeResolver -- hybrid session cleanup core (Pass 2) | Gemini EditorialJudge labels winner/alternate/failed/bts per creator mini-session; D-081: deletes only on mechanical certainty (micro failed + local performance) or `high_confidence_semantic` (>= `delete_confidence`); otherwise records `semantic_delete_recommended` | `composite_resolver.apply_composite_resolution` -> `hybrid_session_cleanup.apply_hybrid_session_cleanup`; `hybrid_editorial.EditorialJudge` | pipeline | kept -> kept/deleted + semantic_decisions + window diagnostics | YES / YES (`hybrid_editorial = provider_complete`; budget ledger 0.02 USD) | Y N N N N N N N | the 19 hooks below, grouping, BestTake, Resolver | 23 | KEEP (D-081 evidence-first) |
| P-25..P-43 | CompositeResolver chain -- 19 hooks wrapping P-24 in historical order: semantic_fragment_guard (deletes `semantic_failed_micro_fragment` <= 1.6 s/<= 3 tokens/failed >= 0.74 -- deleted the lone "No" `clip_37c525bd` in the RAW), hybrid_retry_completion_integrity, hybrid_story_guard (restores unique story coverage), hybrid_alternate_integrity, hybrid_cross_group_retry_integrity, incomplete_bridge_retry_authority, hybrid_failed_continuation_integrity, hybrid_retry_winner_authority, hybrid_gold_reconciliation, failed_prefix_completion_rescue, final_delivery_integrity, terminal_delivery_reconciliation, hybrid_failed_soft_restore (restores weak failed deletions), hybrid_unavailable_retry_fallback, hybrid_complementary_delivery_guard, hybrid_semantic_complementary_rescue (restores complementary alternates), hybrid_semantic_composite_bridge, hybrid_composite_best_take (marks composites), hybrid_semantic_conflict_arbitration | delete / restore / rescue / composite-mark on the hybrid cleanup output | each module's own `install_*` called once by `composite_resolver._build_take_level_chain` against private scratch state (D-023) | composite_resolver | HybridSessionCleanupResult -> HybridSessionCleanupResult (+ composite_split_ids) | YES / UNKNOWN per hook (no per-hook marker; only the merged `hybrid_editorial_chunks` decisions are recorded) | Y N Y Y Y N N N (as a set: delete, restore, pick winners, mark composites) | grouping may still split them; BestTake; ClaimCoverage; Resolver can re-decide everything | 2-10 each | DUPLICATED as a set (four independent restore paths: story_guard, failed_soft_restore, semantic_complementary_rescue, complementary_delivery_guard; two composite paths: semantic_composite_bridge, composite_best_take) -- see Part 10 D-2; individually UNCERTAIN because their live firing is not observable |
| P-44 | Session partition + lexical retry grouping (IdeaClusterer tier 1) | retry families by lexical similarity within creator sessions | `session_boundaries.safe_group_takes_by_sessions` -> `take_grouping_provider.safe_group_takes` -> `take_grouping.group_takes` (+ hooks local_retry_grouping, retry_group_integrity, final_sibling_grouping, session_grouping_bridge, global_session_sibling_bridge) | pipeline | kept -> groups | YES / YES (`take_grouping = baseline_complete`) | N Y N N N N N N | composite split, semantic tier, cohesion split, StoryValidator residual re-grouping | 9 / 23 | KEEP (tier) ; the 5 grouping hooks: UNCERTAIN (no live evidence which fired) |
| P-45 | Composite group split | force accepted composites into singleton groups so BestTake cannot collapse them | `composite_resolver.apply_composite_group_split` | pipeline | groups+split_ids -> groups | YES / YES | N Y N N Y N N N | semantic tier / cohesion honour `protected_ids` | 8 | KEEP |
| P-46 | SemanticArbiter tier (IdeaClusterer tier 2) | Gemini pairwise "same intended idea" merge of lexically-separate groups; priority-ranked pair budget; D-042 editorial-slot policy text + coverage-first pair order injected by `editorial_slot_resolution_install` at import | `take_grouping_provider.reconcile_semantic_idea_equivalence`, `_rank_candidate_pairs`; `semantic_idea_equivalence(_google).build_semantic_equivalence_request` | pipeline | groups -> merged groups + diagnostics (merges/blocked) | YES / YES (`semantic_idea_equivalence` present: 9 confirmed pairs) | N Y N N N N N N | cohesion split; StoryValidator residual families; the whole downstream chain | 20 / 2 | KEEP; the D-042 injection: IMPLEMENTED-BUT-NOT-OBSERVABLE (no diagnostics key; identity marker probes a key nothing writes) |
| P-47 | Cohesion safety split (D-058/D-083/D-085) | split a merged family whose members are distinct ideas; bridge-aware; F3b singleton bridge OFF | `take_grouping_provider.split_incohesive_retry_groups` | pipeline | groups -> groups + `distinct_idea_grouping_safety` | YES / YES | N Y N N N N N N | none for families; StoryValidator may re-pair | 23 | KEEP |
| P-48 | DeliveryScorer | rank family members: 0.70 completeness + 0.30 duration fit, fragment penalties (-0.22/-0.28/-0.18), handling-failure penalty | `take_judge.rank_takes`, `score_take`; `take_judge_provider.safe_rank_takes` | pipeline per family | members -> RankedTake[] (`watch_listen_baseline`) | YES / YES (`take_judge = baseline_complete`) | N N Y N N N N N (sets `local_selected_clip_id`) | P-50, P-51, S-3, S-4, S-10 | 12 | KEEP -- but note: no visual/audio cleanliness feature beyond MediaSignals from A-5; visual_fumble is ONLY a penalty input, never an objective take-cleanliness score (Part 9 G-6) |
| P-49 | Family-scoped hybrid labels (D-094.3 F8) | prefer labels from a window that saw the whole family | `pipeline.family_scoped_semantic_decisions` | pipeline | window labels -> per-family labels | YES / YES (`semantic_label_source` present) | N N N N N N N N | P-50 | 51 (pipeline) | KEEP |
| P-50 | Draft Best Take (`_semantic_best_take`) | one winner per family: single hybrid winner >= 0.85 -> D-081 delete-recommended exclusion -> completeness -> CRITICAL_COVERAGE_DOMINANCE -> unique-fact asymmetry = unresolved (keep local) -> delivery tie-break; wrapped by `semantic_best_take_integrity` (rejects incomplete semantic overrides) | `pipeline._semantic_best_take`, `semantic_best_take_integrity` | pipeline per family | labels+ranked -> `TakeGroup.selected_clip_id` + reason | YES / YES (6 multi-member families: reasons single_semantic_winner x3, delivery_tie_break x2, unresolved_unique_fact_asymmetry x1) | N N Y N N N N N | P-52 (compose), the 14 wrappers, S-1..S-4, S-10 (Resolver re-decides per idea) | 51 / 5 | KEEP -- with a proven defect class: when every member is labelled `failed` the ladder still elects a survivor ("delivery_tie_break_among_survivors" on tg_5e4d80379e913bb381: both members failed 0.90/0.95) -- a failed-vs-failed family has no clean winner and should not produce a KEEP by tie-break (Part 4 C-2, Part 9 G-3) |
| P-51 | Pre-Selection physical edge trims (Pass 4) | trim high-confidence recording-process events at take edges (0.30 s tolerance) + 5 edge hooks | `temporal_editing.refine_takes_with_temporal_context` (+ delivery_edge_trim, interior_performance_break, recording_suffix_trim, script_consult_trim, temporal_word_boundary_integrity) | pipeline AFTER Best Take, BEFORE compose | kept -> kept with new start/end + `temporal_performance_trims` | YES / YES | N N N N N Y N N | Boundary stages (2.12-2.14) and render trailing trim may re-cut the same edges | 3 | CONFLICTING with D-021 ownership ("Boundary-only physical timing after freeze"): a physical authority running before Selection is final (Part 4 C-6) |
| P-52 | compose_selected | one winner per family + every ungrouped take -> selected | `composer.compose_selected` (+ `selection_integrity` wrapper: drops weak singleton retry fragments) | pipeline | kept+groups -> selected takes | YES / YES | Y N N N N N N N | draft review (OFF), the 14 wrappers, all of 3C | 5 | KEEP (rule "ungrouped = keep" is the structural source of Level-1 false keeps when family formation misses a retry: Part 4 C-1) |
| P-53 | DraftTimeline assembly + composite family stabilization (step 16) | draft buckets + diagnostics; repair concise-discard + later-winner composites vs a monolith | `pipeline.build_flow_b_draft` tail, `composite_resolver.apply_composite_family_stabilization` | pipeline | -> DraftTimeline | YES / YES | Y N N Y Y N N N | 3C | 51 / 8 | KEEP |
| P-54..P-62 | Draft wrappers in install order (each re-edits membership AFTER the inner draft): final_draft_retry_integrity, selected_failed_bridge_integrity, round8_retry_reconciliation (+round9 orphan prefix), round11_semantic_retry_cleanup, short_bts_process_cleanup, post_selection_incomplete_bridge_authority, post_selection_internal_retake_trim, final_selection_retry_arbiter (D-042 slot install rides on it) | each: a retry structure "only visible after Best Take" | `cutsell_worker/__init__.py` installs; each patches `pipeline.build_flow_b_draft` | import time | DraftTimeline -> DraftTimeline | YES / UNKNOWN per wrapper (only their own diagnostics keys when they fire; none printed for the RAW) | Y Y Y N N Y N N (as a set) | 3C stages; Resolver | 1-3 each | DUPLICATED / UNCERTAIN: they are a second, undeclared draft-level Selection authority stack outside D-021's component map; `post_selection_internal_retake_trim` and `final_selection_retry_arbiter` then run AGAIN inside S-1 (Part 10 D-3) |
| P-63 | Premature freeze | freeze the draft before StoryValidator/plan exist (pre-V1 holdover; D-025 corrects the record) | `selection_boundary_contract.install_selection_freeze` | import time wrapper on build_flow_b_draft | -> `selection_boundary_contract` v0 | YES / YES (superseded by S-16) | N N N N N N N N | S-16 overwrites; `superseded_premature_freeze_status` recorded | 8 | LEGACY (harmless but misleading; a second freeze) |
| P-64 | Edge-only boundary after (premature) freeze | edge trims on the draft | `post_selection_edge_only_boundary` | wrapper on build_flow_b_draft | selected -> selected (edges) | YES / UNKNOWN | N N N N N Y Y N | 2.12-2.14, render | 2 | UNCERTAIN (Boundary work executed before the authority-level Selection; Part 4 C-6) |
| P-65 | Interior gap trim (D-046 / D-095.2) | split a kept clip at word-gap+visual-reset or at a proven >= 1.2 s audio silence (0.12 s pads); fragment provenance | `post_selection_interior_gap_trim.split_selected_interior_performance_gaps` | wrapper on build_flow_b_draft | selected -> fragments (`__psig*`) + trace | YES / YES (2 splits in the RAW: 1 multimodal, 1 long_audio_silence; 18 edge-margin rejections) | N N N N N Y Y N | `final_boundary_authority` (preserves gaps by parent id, D-095.2), render | 8 | KEEP -- but ORDER DEFECT: runs before the Resolver, so a clip the Resolver RESTORES later (e.g. `clip_54d6051e`) is never interior-trimmed (Part 4 C-7) |
| P-66 | Continuity coalescer | re-merge over-segmented selected clips when source continuity is proven | `post_selection_continuity_coalescer` | wrapper | selected -> selected | YES / UNKNOWN | N N N N N Y Y N | render_plan coalesces again (`_coalesce_contiguous_segments`) | 1 | DUPLICATED with `render_plan._coalesce_contiguous_segments` (Part 10 D-6) |
| P-67 | Audio boundary completion | source-audio boundary completion after the Flow B draft | `audio_boundary_completion_install` (patches `flow_b.process_local_sources` and the universal entry) | import time | result -> result (edges) | YES / UNKNOWN | N N N N N Y Y N | 2.12-2.14 | 1 | UNCERTAIN (a third edge-completion authority besides `final_boundary_authority` envelope + `tighten_trailing_silence`) |
| P-68 | Boundary selection invariant (draft level) | refuse a timeline whose token stream differs from the frozen one | `selection_boundary_contract.install_boundary_selection_invariant` | wrapper | -> raises on drift | YES / YES | N N N N N N N D | S-17 repeats the same check after the real freeze | 8 | DUPLICATED with S-17 (harmless) |

### 3C. Authority-level Selection, Freeze, Boundary (`universal_clean_cut.process_universal_clean_cut_sources`)

| # | STAGE / module | Purpose -- canonical responsibility | Files / primary functions | Called by | Input -> Output | Active PROD / RAW | Caps M F W R C S B D | Runs after it that can modify/undo | Tests | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|
| S-1 | Selection phase authority | "execute final semantic Selection deterministically in one place": complementary_family_stabilizer (AGAIN), internal retake trim (AGAIN), final_selection_retry_arbiter (AGAIN), swap-alternate preservation (writes `alternates`), final retry guards (failed / redundant bridge / short-alternate) | `selection_phase_authority.apply_selection_phase_authority` | universal_clean_cut | draft -> draft (`selection_phase_authority` diag) | YES / YES | Y N Y N Y N N N | S-2..S-4, S-10 | 5 | DUPLICATED (repeats P-53/P-60/P-61) and CONFLICTING with D-019 (still routes losers to SWAP/alternates; D-092 folds them back to discarded 9 stages later) |
| S-2 | Conflicted bridge guard | selected -> alternates for proven conflicted redundant bridges | `selection_conflicted_bridge_guard.apply_selection_conflicted_bridge_guard` | universal_clean_cut | draft -> draft | YES / YES (no-op unless bridges) | Y N N N N N N N | S-10 (re-decides), D-092 fold | 2 | KEEP-as-evidence per D-021 ("feeds BestTake"); actually an independent membership editor -> UNCERTAIN |
| S-3 | DeterministicBestTakeAuthority (D-021 "BestTakeResolver") | lock a CLEAR local-ranker winner (score gap >= 0.30); losers discarded (swap off) | `deterministic_best_take_authority.apply_deterministic_best_take_authority` | universal_clean_cut | draft+take_judge_groups -> draft | YES / NO-OP in the RAW (no family gap >= 0.30; returns the draft untouched and writes NO diagnostics -> identity marker "absent") | Y N Y N N N N N | S-4, S-10 | 4 | WIRED-BUT-NOT-PROVEN-IN-RAW: has never been observed moving a clip in any recorded Video00 run; it is ALSO latently CONTRADICTORY with P-50 (a clear local score gap would override a hybrid `winner` label, re-selecting a `failed`-labelled take -- e.g. tg_473eca7ba3cb7667a6 local 0.7292 `failed` vs 0.6663 `winner`, gap 0.063, saved only by the 0.30 threshold) -- Part 4 C-3. The D-021 map names it "KEEP AS CORE / PROMOTED"; the RAW evidence says the effective BestTakeResolver is P-50 + S-10 |
| S-4 | ClaimCoverageBestTake (critical-claim / semantic protection) | winner override when another member covers every CRITICAL claim; CRITICAL_COVERAGE_DOMINANCE on multi-selected families; composite formation; incidental-claim gate | `claim_coverage_best_take.apply_claim_coverage_best_take` | universal_clean_cut | draft -> draft (`claim_coverage_best_take` diag) | YES / YES | Y N Y Y Y N N N | S-5, S-10 (the Resolver recomputes the same claim coverage and can disagree) | 13 | DUPLICATED with S-10's requirement-group model (same claims, same importance rules, two decision makers) -- Part 10 D-4 |
| S-5 | StoryValidator pass 1 (legacy evidence) | contradiction invariant, idea coverage, lost atoms / lost critical claims, residual family resolution via arbiter; folds alternates | `final_story_coherence_validation.apply_final_story_coherence_validation` | universal_clean_cut | draft -> draft (`*_legacy_evidence`) | YES / YES | Y Y N R? C? N N D (in this pass it CAN edit) | S-10 overrides; its output is relabelled legacy evidence | 25 | LEGACY-EVIDENCE (kept alive so the Ledger can read it; D-050C3) |
| S-6 | CanonicalEditPlan + FinalEditReviewer + repair loop pass 1 | plan v1, findings (DUPLICATE_IDEA, UNRESOLVED_RETRY, INCOMPLETE_DELIVERY, ORPHAN_FRAGMENT, UNIQUE_FACT_LOST, IDEA_COVERAGE_LOST, CONTRADICTION, INCOMPATIBLE_COMPOSITE, REQUIRED_CONTINUATION_LOST, STORY_ORDER_BREAK, CAUSAL_ORDER_BREAK, CRITICAL_CLAIM_LOST), bounded reorder repair | `repair_loop.run_repair_loop`, `canonical_edit_plan.build_canonical_edit_plan`, `final_edit_reviewer.review` | universal_clean_cut | draft -> plan+review (`*_legacy_evidence`) | YES / YES | N N N N N N N D (order repair only) | S-12 recomputes | 7/24/13 | LEGACY-EVIDENCE (second copy at S-12) |
| S-7 | Semantic Ledger | read-only reconstruction of ideas / realizations / claims / decisions / discards | `semantic_ledger.build_semantic_ledger_shadow` | universal_clean_cut | draft -> ledger + parity | YES / YES | N N N N N N N N | none | 19 | KEEP (single input of S-10) |
| S-8 | Resolver shadow report | per-idea decision model (pure) | `realization_resolver.resolve_realizations_shadow` | universal_clean_cut | ledger -> report | YES / YES | N N N N N N N N | S-10 applies it | 18 | KEEP |
| S-9 | Editorial-slot resolution (D-042) | minimum-sufficient-set policy for the semantic-equivalence prompt + coverage-first pair order (+ legacy unified payload) | `editorial_slot_resolution_install.install_editorial_slot_resolution` (runs inside `install_final_selection_retry_arbiter` at import) | import time; effect at P-46 | request payload -> request payload | YES / UNKNOWN (no result-level proof; identity probes `diagnostics.editorial_slot_resolution`, a key no code writes) | N N N N N N N N (advisory to the arbiter) | nothing reads "slot" afterwards | 2 | IMPLEMENTED-BUT-NOT-OBSERVABLE; as an engine capability ("slot understanding") MISSING |
| S-10 | UNIFIED REALIZATION RESOLVER -- authoritative application (the ONE semantic Selection authority, D-050C2/C3, D-087, D-089, D-092) | per idea: contradiction -> REVIEW_REQUIRED; if one realization covers every CRITICAL requirement group -> winner by tiers (not-proven-incomplete, high-confidence semantic winner, critical-claim richness, delivery score, richness); else MINIMAL COMPOSITE covering all CRITICAL groups; else REVIEW_REQUIRED. Applies buckets; RESTORES losers into `selected` (placement units); retained-for-context -> alternates -> folded to discarded (D-092) | `realization_resolver._resolve_one_idea`, `_find_minimal_composite`, `apply_authoritative_realization_resolution`, `_place_restored_clips_at_story_position` | universal_clean_cut (AUTHORITATIVE mode) | draft+ledger+report -> draft (`realization_resolver_authority`, `authoritative_story_placement`) | YES / YES (status SEMANTICALLY_RESOLVED; 18 ideas: 16 winners, 2 composites; one composite `idea_c50529df…` differs from the legacy winner -> a discarded realization was RESTORED) | Y N Y Y Y N N D | S-11..S-13 are validation-only (D-090); Boundary is physical-only; render renders `selected` | 18 | KEEP as the single authority -- but its OBJECTIVE FUNCTION is "cover every CRITICAL claim" (coverage maximisation), which structurally contradicts the accepted D-042 doctrine "minimum sufficient editorial set; GOOD+GOOD => one winner" whenever a losing take carries any extra CRITICAL-classified claim (Part 7, Part 4 C-4). Realization ids in its diagnostics are not mapped to clip ids/text (Part 5 observability gap) |
| S-11 | Semantic preservation proofs + effective importance index (D-076/D-079/D-089) | prove a discarded claim survives elsewhere; ONE importance truth per claim | `realization_resolver.build_semantic_preservation_proofs`, `build_effective_claim_importance_index` | universal_clean_cut | ledger -> proofs / index | YES / YES | N N N N N N N N | consumed by S-12 | 18 | KEEP |
| S-12 | StoryValidator pass 2 -- validation only (D-090) | re-validate on the resolved selection; NEVER edits; signature invariant fails closed | `post_authority_validation.*`, `final_story_coherence_validation.apply_post_authority_story_validation` | universal_clean_cut | draft -> draft (diag) + `freeze_blocked` | YES / YES | N N N N N N N D | none | 3 / 25 | KEEP |
| S-13 | CanonicalEditPlan + FinalEditReviewer + repair loop pass 2 (authoritative source, D-087) | the plan Freeze consumes; order-only repair | `repair_loop.run_repair_loop(authoritative_source=…)` | universal_clean_cut | draft -> plan v2 + review | YES / YES (plan v2, 23 fragments) | N N N N N N N D | none | 7 | KEEP |
| S-14 | Post-authority integrity + freeze gate | signature drift / REVIEW_REQUIRED / reviewer FAIL -> not frozen | `universal_clean_cut` inline | -- | -> `freeze_blocked_pending_coherence_review` | YES / YES (not blocked in the RAW) | N N N N N N N D | none | 16 | KEEP |
| S-15 | Complete-idea boundary recovery (BEFORE Freeze) | expand each selected clip to its complete-idea / complete-word envelope from the full source transcript; reconcile same-source overlaps; preserve polished interior gaps by parent id (D-095.2) | `final_boundary_authority.enforce_complete_idea_boundaries`, `_clip_from_envelope`, `_reconcile_same_source_overlaps` (+ `terminal_sentence_boundary_guard`, `boundary_retry_tail_guard` wrappers) | universal_clean_cut | selected -> selected (ranges) | YES / YES (23 rows, all `keep_complete_idea_envelope`, 0 s added in the RAW) | N N N N N Y Y N | S-16 freezes the result; S-17 polish; render trailing trim | 4 | KEEP -- but it is a SOURCE-RANGE authority that runs BEFORE Freeze while D-021 states Boundary runs after Freeze; and it can ONLY EXPAND (never shrink), so a loose tail (the 23 LEVEL-1 boundary regions / 8.3 s: `loose_exit_edge` 0.79-1.14 s on `clip_829d`, `clip_eda4`, `clip_d3d8`) has no owner before render except `tighten_trailing_silence` (Part 4 C-8) |
| S-16 | SELECTION FREEZE | freeze plan id/version/semantic hash + ordered token digest | `selection_boundary_contract.freeze_selection_contract` | universal_clean_cut | draft+plan -> frozen contract | YES / YES (`frozen`, plan v2) | N N N N N N N D | S-17 verifies | 8 | KEEP |
| S-17 | BoundaryEngine after Freeze | remove micro visual-reset word gaps (0.3-0.4 s) with strong reset evidence; mint fragment identity | `human_boundary_polish_v5.polish_human_boundaries_v5` | universal_clean_cut | selected -> fragments | YES / YES (1 split: `clip_8fdc…psigr` 53.93-54.25) | N N N N N Y Y N | `enforce_selection_contract`; render | 6 | KEEP -- this is the ONLY Boundary work after Freeze; it cannot fix entry debris, loose exits or interior dead air (all done, if at all, before Freeze) |
| S-18 | Selection contract enforcement | fail closed if the ordered spoken token stream changed | `selection_boundary_contract.enforce_selection_contract` | universal_clean_cut | -> `verified` | YES / YES | N N N N N N N D | none | 8 | KEEP |
| S-19 | Unified Selection reasoner + swap-enabled deterministic authority (pre-V1 branches) | whole-video SELECT/SWAP/DISCARD | `unified_selection_reasoner`, `unified_selection_google` | `clean_cut_core_v1_enabled=False` only | -- | NO / NO | Y Y Y Y N N N N | -- | 4 | LEGACY (rollback only, D-019/D-020) |

### 3D. Render, technical QC, perceptual QC, delivery

| # | STAGE / module | Purpose -- canonical responsibility | Files / primary functions | Called by | Input -> Output | Active PROD / RAW | Caps M F W R C S B D | Runs after it that can modify/undo | Tests | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|
| R-1 | Render plan | selected clips -> source-safe segments; coalesce contiguous | `render_plan.build_render_plan` | validation harness / export job | draft -> RenderSegment[] | YES / YES | N N N N N N N N (reads `selected` only) | R-2 trailing trim | 9 | KEEP |
| R-2 | Renderer | per-segment ffmpeg: `tighten_trailing_silence` (silence >= 0.28 s reaching the tail -> cut to +0.04 s, cap 12 s), `volume`, 12 ms join fades, concat, captions/overlays | `render.render_preview`, `tighten_trailing_silence` | `live_render_qc.render_with_post_render_qc` | segments -> MP4 | YES / YES (23/23 fragments located in the MP4; trailing trims up to 1.24 s) | N N N N N Y Y N | R-3 repair loop re-renders | 25 | KEEP -- but it is a HIDDEN Boundary authority: the largest physical tail edits in the RAW (0.79-1.24 s on 8 fragments) happen here, after Freeze, outside any Boundary diagnostics (Part 4 C-8) |
| R-3 | Technical post-render QC + bounded physical repair (D-028/D-030/D-036/D-094.3 F13) | structural: plan coverage / sequence / duplicates (attempt 1 only); media on the DECODED MP4: decode integrity, LINGERING_ACCIDENTAL_SILENCE (>= 1.2 s @ -35 dB), FROZEN_OR_REPEATED_FRAME, DEAD_BLACK_FRAME, ABRUPT_AUDIO_DISCONTINUITY at joins; one Boundary-only edge repair per attempt; delivery gate | `live_render_qc.render_with_post_render_qc`, `post_render_media_qc.*`, `live_boundary_repair.repair_segment_for_finding`, `post_render_watch_listen_qc.check_*` | validation harness (and the real export job) | MP4 -> LiveRenderQCResult (PASS / NEEDS_HUMAN_REVIEW / SEMANTIC_MISMATCH_INVALIDATED) | YES / YES, BLOCKING (RAW: NEEDS_HUMAN_REVIEW; attempts: PHYSICAL_FAIL_REPAIRED x2 with 0.05 s trailing trims on one fragment; the 2.32 s silence unrepairable) | N N N N N Y Y D | R-4 only if deliverable | 7 / 2 / 0 (live_boundary_repair has NO test file) | KEEP (blocking, real, decoded); live_boundary_repair: TEST-ONLY-GAP |
| R-4 | Auto speech-safe visual microtrim | post-render frame-aware microtrims between word envelopes | `speech_visual_microtrim.detect_speech_safe_visual_microtrims`, `locked_selection_replay._apply_review_cuts` | serverless_handler (only when deliverable) | MP4 -> MP4 | COND / NO (candidate not deliverable) | N N N N N Y Y N | none | 4 | KEEP (dormant on failing candidates); a THIRD post-Freeze physical editor besides R-2 and R-3 |
| R-5 | PERCEPTUAL SYSTEM WATCH + LISTEN | perceptual gate on the decoded MP4 that routes to the owning authority, never changes membership | `post_render_watch_listen_qc.PostRenderWatchListenQCProvider` (Protocol), finding kinds declared, `routes_to` field exists | NOBODY | -- | NO / NO | (would be: N N N N N N N D) | -- | 7 (structural checks only) | MISSING (DOC/CONTRACT-ONLY) |
| R-6 | Delivery gate + artifact naming | `deliverable` iff technical QC PASS; invalidated preview never uploaded as `preview.mp4` | `serverless_handler._focused` | handler | result -> S3 keys | YES / YES | N N N N N N N D | none | 5 | KEEP |
| R-7 | Active-path identity (D-095 addendum) | in-worker build sha + package fingerprint + component markers | `active_path_identity.build_active_path_identity` | serverless_handler | result -> identity block | YES / YES | N N N N N N N N | none | 1 | KEEP -- with two probe defects (S-3 no-op ambiguity; S-9 key never written) |
| R-8 | Human Watch + Listen | final human acceptance | -- | -- | -- | MANUAL | -- | -- | -- | MISSING as a gate object (by design a human step; needs a recorded verdict artifact) |
| R-9 | QA-only oracles and validators | Gold 18-check manifest (`validate_video00_regression_qa`), selection lock (`validate_video00_selection_lock`), architecture verifier, four-way ladder (`video00_quality_ladder`) | `benchmarks/*` | workflows only | result/MP4 -> reports | QA only (never imported by `cutsell_worker`) | none | none | 37 (ladder) | KEEP (QA-ONLY; import guard tested) |

### 3E. Modules present in the package but NOT on the active path

| Module | Evidence | STATUS |
|---|---|---|
| `cross_group_truncated_winner_authority`, `hybrid_performance_retry_restore_guard`, `incomplete_unique_bridge_completion_rescue`, `post_selection_composite_handoff_trim` | their `install_*()` is called by no production module (only by their own tests); `post_selection_composite_handoff_trim` is intentionally not installed (phase ownership) | DEAD/INACTIVE (TEST-ONLY) |
| `speech_safe_dead_air_guard` | installed at import but patches `human_boundary_polish_v3._remove_interval`; v3 is not on the active path | DEAD/INACTIVE (patches legacy code) |
| `human_boundary_polish` v1-v4 | not imported by `universal_clean_cut` (only v5 is) | LEGACY |
| `unified_selection_reasoner`, `unified_selection_google`, `deterministic_best_take_authority(swap_enabled=True)` branch | rollback only | LEGACY (D-019/D-020) |
| `clean_cut_openai`, `composer_openai`, `draft_review_openai`, `semantic_openai`, `take_judge_openai`, `visual_openai`, `whole_video_openai`, `take_grouping_openai` | providers not configured on the RAW path | LEGACY / INACTIVE providers |
| `live_boundary_repair` | active (R-3) but has zero test files | WIRED, TEST-GAP |
| `attempt_boundary_integrity`, `retry_group_integrity`, `selection_integrity`, `session_grouping_bridge`, `audio_boundary_completion_install`, `locked_selection_replay` | active wrappers with zero test files referencing them | WIRED, TEST-GAP |

---

## PART 4 -- AUTHORITY COLLISION MAP

Every collision below is proven either in the audited RAW (`34008386434`) or
structurally in the code on this head. Classification vocabulary:
INTENTIONAL COMPLEMENT / REDUNDANT / CONTRADICTORY / ORDER-DEPENDENT /
REGRESSION RISK / UNKNOWN.

**C-1 -- "ungrouped = keep" turns a missed family into a Level-1 false keep (RAW proven).**
```
IdeaClusterer lexical tier (P-44) + SemanticArbiter tier (P-46; pair budget)
  ACTION: forms family tg_5e4d80379e913bb381 = {clip_59e27e7e (82.82-90.60), clip_68697ccd (91.20-94.34)};
          leaves clip_1c5a0826 (95.52-107.48, "al terminar mi contrato cambié de ginecóloga…") UNGROUPED
  v
Draft Best Take (P-50)
  ACTION: both members labelled failed (0.90 / 0.95) -> no winner -> "delivery_tie_break_among_survivors"
          -> KEEPS clip_59e27e7e (a failed take, 0.7105 vs 0.704)
  v
compose_selected (P-52)
  ACTION: keeps the ungrouped clip_1c5a0826 as "independent material"
  v
Resolver (S-10)
  ACTION: two different ideas -> two winners, no competition
  v
FINAL EFFECT: failed take (6.1 s physical) + its clean retry both play back-to-back;
              ladder: 7.31 s LEVEL-1 take_choice_against_both_references + 0.81 s ungrouped_retry_of_kept_idea
```
Classification: CONTRADICTORY (a family of only-failed members must not elect
a KEEP) + ORDER-DEPENDENT (family completeness decides everything downstream).
Owner of the wrong decision: IdeaClusterer (missed member) AND BestTake ladder
(no "all failed" branch).

**C-2 -- Best Take ladder vs hybrid `failed` labels (structural + RAW).**
```
EditorialJudge (P-24)  ACTION: labels a member failed 0.90-0.98 (window saw the family)
  v
D-081 policy (P-24)    ACTION: semantic judgment may NOT delete -> "semantic_delete_recommended" only
  v
_semantic_best_take (P-50) ACTION: step 1 excludes delete-recommended members "unless that would eliminate every candidate"
                                   -> when ALL are failed, all survive -> delivery tie-break elects one
  v
S-10 Resolver          ACTION: single realization full critical coverage -> confirms the tie-break winner
FINAL EFFECT: tg_5e4d… (above) and tg_a7d4b8d6299e8803bd (acne: clip_6c372ce1 failed 0.90 vs clip_9f8d4903 failed 0.98 -> 6c372 kept; both references chose the realization that CutSell discarded, 0.5 s LEVEL-1)
```
Classification: CONTRADICTORY between the D-081 fail-open rule and the
Level-1 requirement "failed material is not preserved merely because it
contains semantic atoms". Note: fail-open is right for ONE candidate ("WHEN
UNCERTAIN, KEEP"); it is wrong as a tie-break among several failed ones.

**C-3 -- Two Best-Take resolvers with opposite priors (latent, structural).**
```
P-50 _semantic_best_take   ACTION: single hybrid winner >= 0.85 overrides the local ranker (tg_473eca…: winner clip_abcbb706 over local clip_b77dbf41 0.7292)
  v
S-3 DeterministicBestTakeAuthority ACTION: IF local gap >= 0.30, re-selects the LOCAL winner and discards the rest (would re-select the failed-labelled clip_b77dbf41)
  v
S-10 Resolver             ACTION: tier 2 prefers the recorded SEMANTIC_WINNER_OVERRIDE >= 0.85 -> would flip it back
FINAL EFFECT (RAW): none -- no family reached the 0.30 gap; S-3 was a NO-OP in every recorded Video00 run
```
Classification: CONTRADICTORY (two opposite priors: local score vs semantic
label) + REGRESSION RISK (any family with a large ranker gap and a decisive
hybrid label flips twice). D-021 names S-3 the canonical BestTakeResolver; in
practice P-50 + S-10 decide.

**C-4 -- Best Take discards, Resolver restores: the conclusion (RAW proven).**
```
P-46 SemanticArbiter    ACTION: pairs clip_dcc5b751 (295.52-313.50, "Esta es mi experiencia. Soy la única en mi familia…") with clip_54d6051e (319.38-334.24, "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcino…") -> family tg_03324733ca1695c430
  v
P-50 Best Take          ACTION: dcc5b751 winner 0.96, 54d6051e failed 0.88 -> single_semantic_winner = dcc5b751; 54d6051e DISCARDED
  v
S-4 ClaimCoverageBestTake ACTION: no override recorded
  v
S-10 Resolver           ACTION: requirement groups from BOTH texts; the loser carries >= 1 CRITICAL-classified claim the winner does not cover
                                 (generalising / hereditary-statistic language is CRITICAL by rule `generalizing_statistic_language`)
                                 -> no single realization covers all CRITICAL groups -> _find_minimal_composite -> RESOLVED_COMPOSITE
                                 (idea_c50529df3b774a8864c4: legacy winner real_d4893…, authoritative composite {real_8135…, real_d4893…}, legacy_vs_authoritative_same=false)
                                 -> 54d6051e RESTORED into `selected` and placed after dcc5b751 (D-089 unit)
  v
S-12 StoryValidator (validation-only)  ACTION: cannot touch it (D-090)
  v
R-1/R-2 render          ACTION: renders both (dcc5b751 17.98 s + 54d6051e 14.89 s)
FINAL EFFECT: 14.9 s restatement after a complete conclusion -- the exact Level-2 defect the D-042 doctrine forbids;
              ladder: gold_removes_cutai_keeps (Cut.ai also keeps a restatement; Gold keeps one conclusion + CTA)
```
Classification: CONTRADICTORY (later module undoing an earlier correct
decision) -- not a bug in either module: the Resolver's objective (cover every
CRITICAL requirement group) is coverage maximisation; D-042's accepted doctrine
is minimum sufficient set. The two objectives cannot both be the semantic
authority. Note `_effective_importance` already downgrades CRITICAL ->
SUPPORTING for low-information incidental claims, but a statistic/hereditary
generalisation is protected by design.

**C-5 -- Composite formation vs family competition: pimples (RAW proven, mixed direction).**
```
P-44/P-46 IdeaClusterer  ACTION: family tg_8819941afa23231b2b = {a93a9633 (192.44-198.12 "También me salían espinillas. Era como un rush, una alergia."),
                                  c041d216 (198.88-211.02 monolith), ec3ef606 (213.34-222.98 "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte…")}
  v
P-24 EditorialJudge     ACTION: family window: a93a alternate 0.70, c041 alternate 0.75, ec3ef winner 0.95 (global merge had c041 winner 0.95 -> F8 family window wins)
  v
P-50 Best Take          ACTION: single_semantic_winner ec3ef606; a93a9633 + c041d216 DISCARDED (local ranker preferred c041 0.6671)
  v
S-10 Resolver           ACTION: single realization full critical coverage -> ec3ef606 confirmed; a93a stays discarded
  v
FINAL EFFECT: Gold and Cut.ai BOTH keep a93a9633 (191.77-197.52) AND ec3ef606 (213.46-221.71) as complementary pieces; CutSell drops a93a (5.75 s missing_delivery, LEVEL-1)
              and keeps the ec3ef tail 221.71-222.98 (1.27 s dead air; source silence 221.673-224.401 rejected by the interior trimmer as edge-margin)
```
Classification: CONTRADICTORY with the accepted composite doctrine ("clean
subpart A + clean complementary subpart B -> COMPOSITE") -- here the family
competed instead of compositing; in C-4 the family composited instead of
competing. Same two modules, opposite errors: the decision "compete vs
composite" has no owner that reasons about editorial function; it falls out
of claim-coverage arithmetic (a93a's claims are subsumed by ec3ef's -> "safely
redundant"; 54d6's extra statistic is not -> "composite").

**C-6 -- Physical trimming before Selection is final, then Boundary "after Freeze" (structural).**
```
P-51 refine_takes_with_temporal_context (+5 hooks)   ACTION: trims take edges (pre-compose)
  v
P-64 post_selection_edge_only_boundary / P-65 interior gap trim / P-66 coalescer / P-67 audio completion  ACTION: edge + interior physical edits on the draft
  v
S-15 enforce_complete_idea_boundaries               ACTION: EXPANDS to the complete-idea envelope (can only grow; 0 s in the RAW)
  v
S-16 FREEZE
  v
S-17 polish v5                                       ACTION: only micro visual-reset gaps (1 split in the RAW)
  v
R-2 tighten_trailing_silence                         ACTION: cuts silent tails up to 1.24 s (8 fragments in the RAW) -- invisible to Boundary diagnostics
  v
R-3 bounded repair                                   ACTION: 0.05 s trailing trims for ABRUPT_AUDIO_DISCONTINUITY
FINAL EFFECT: five physical editors before Freeze, one after, two inside the renderer; the D-021 sentence "Boundary-only physical timing after freeze"
              describes S-17 only. Entry-edge quality (tight entries 0.38-0.64 s at 48.45, 82.29, 275.71, 294.88 -- the last one cuts the negation "No" of "No quiero sonar a conspiración")
              has NO owner at all after Freeze: S-15 can only expand from ASR words, and the ASR never produced that "No".
```
Classification: ORDER-DEPENDENT + REGRESSION RISK (a restored clip skips
P-64..P-67 entirely, see C-7).

**C-7 -- Interior gap trimmer runs before the Resolver restores (structural, RAW-relevant).**
```
P-65 interior gap trim   ACTION: splits selected clips at proven silence/reset (ran on the draft's selected set)
  v
S-10 Resolver            ACTION: restores clip_54d6051e (and any future composite member) from `discarded` into `selected`
  v
S-15/S-17/R-2            ACTION: envelope keep, no interior work, only trailing silence
FINAL EFFECT: a restored take is never interior-trimmed; its dead air (if any) reaches the render and is unrepairable by R-3 (mid-segment)
```
Classification: ORDER-DEPENDENT (the D-095.2 fix is correct but sits at the
wrong point relative to the authority).

**C-8 -- Loose exits: three partial owners, none complete (RAW proven).**
```
S-15 envelope    ACTION: cannot shrink a tail
  v
R-2 tighten_trailing_silence   ACTION: cuts a tail only if silence >= 0.28 s reaches the segment end (cut 0.79-1.24 s on 8 fragments)
  v
R-3 repair       ACTION: 0.05 s nudges for join discontinuities
FINAL EFFECT: 23 boundary-scope LEVEL-1 regions / 8.3 s in the frozen plan, 20 / 3.4 s still in the MP4 (loose_exit_edge 0.47-1.14 s on clip_d3d8, clip_829d, clip_eda4, clip_ec3ef…; tight_edge entries)
```
Classification: REDUNDANT + INCOMPLETE ownership (the renderer is doing
Boundary's job with an audio-only rule; no visual exit/entry evidence is used
after Freeze).

**C-9 -- Four restore paths + two composite paths inside CompositeResolver, then ClaimCoverage, then the Resolver (structural).**
```
P-24 hybrid cleanup deletes (mechanical/high-confidence)  -> hybrid_story_guard restores unique story coverage
                                                          -> hybrid_failed_soft_restore restores weak failed deletions
                                                          -> hybrid_semantic_complementary_rescue restores complementary alternates
                                                          -> hybrid_complementary_delivery_guard (final authority for complementary deliveries)
                                                          -> hybrid_semantic_composite_bridge / hybrid_composite_best_take mark composites
  v
S-4 ClaimCoverageBestTake  -> can override the winner or form a composite again (claim coverage)
  v
S-10 Resolver              -> recomputes coverage per idea and can restore / composite again (`_find_minimal_composite`)
  v
S-5 StoryValidator (pass 1) -> can fold / re-pair (legacy evidence only)
FINAL EFFECT: at least SEVEN modules can put a discarded delivery back into KEEP; only S-10 is the declared authority; the six others act before it
              and shape the candidate pool it sees. In the RAW no clip was restored by P-25..P-43 (hybrid deletions were 0 after D-081), so today
              the over-preservation is produced by S-10 (C-4) and by "ungrouped = keep" (C-1), not by the rescue hooks.
```
Classification: REDUNDANT (historically INTENTIONAL COMPLEMENT, each added for
one RAW regression, D-023 kept them "in the same order, same algorithms"). The
protections listed in the directive (unique-fact, critical-claim coverage,
semantic preservation, complementary rescue, composite rescue, claim
equivalence, orphan protection, when-uncertain-keep, retry completion,
post-selection reconciliation) are collectively OVER-PRESERVING through TWO
mechanisms proven live: the Resolver's CRITICAL-coverage objective (C-4) and
the all-failed tie-break (C-2). The rescue hooks themselves are not the live
cause in this RAW.

**C-10 -- Same pass executed twice (structural).**
`post_selection_complementary_family_stabilizer`, `post_selection_internal_retake_trim`
and `final_selection_retry_arbiter` run once as `build_flow_b_draft` wrappers
(P-53/P-60/P-61) and again inside `apply_selection_phase_authority` (S-1).
Classification: REDUNDANT (idempotent by luck, not by contract).

**C-11 -- SWAP still written, then folded (structural).**
S-1 `_restore_semantic_alternates_to_swap` and S-2 write `alternates`; the
Resolver writes `alternates` for `retained_for_contextual_value`; D-092 folds
`alternates` to `discarded` at the authority boundary. Classification: REDUNDANT
with D-019 (KEEP/DISCARD only); harmless today, confusing in diagnostics
(`alternate_count`, `selection_swap_preserved_alternates`).

**C-12 -- Source-level vs render-level silence measurement disagree (RAW proven, cause unknown).**
A-7 found no >= 1.2 s silence inside `clip_f8a423f3` (135.44-149.52; no split, no
rejection trace), yet R-3 measured 2.32 s of -35 dB silence at render
71.64-73.96 = raw ~138.9-141.2 inside that fragment, both references cut
138.59-140.57, and the region is the LEVEL-1 `redundant_realization_both_kept`
1.98 s row. Same threshold, same filter, different answer. Candidate causes:
per-segment `volume` filter and re-encode changing the noise floor; the
whole-video event list being consumed from a truncated diagnostics copy; a
0.6 s minimum splitting one long pause into sub-threshold pieces. Classification:
UNKNOWN -> INVESTIGATE (Part 9 G-14). Until resolved, D-095.2 is NOT proven to
remove the dead air it was built for.

---

## PART 5 -- END-TO-END TRACES (Video00 as QA EVIDENCE ONLY)

Format per trace: RAW SOURCE RANGE -> ASR/TEXT -> ATTEMPT -> IDEA -> EDITORIAL
FUNCTION -> RETRY FAMILY -> ALL CANDIDATES -> DELIVERY SCORES -> BEST TAKE ->
RESCUE/PROTECTION -> COMPOSITE -> FINAL SELECTION -> CANONICAL EDIT PLAN ->
FREEZE -> BOUNDARY IN -> BOUNDARY OUT -> RENDER RANGE -> POST-RENDER QC -> FINAL
MP4 RANGE, with the responsible module at each arrow. Values come from run
`34008386434` (result.json via the CPU ladder log) and the ladder's plan->MP4
template verification. Where a link cannot be produced from the recorded
diagnostics it is marked **[NOT OBSERVABLE]** with the missing key named --
nothing below is inferred.

**Observability gaps common to all five traces (must be closed before any
trace can be called complete):**
1. `attempt_id` per selected clip is not printed by any workflow step and the
   ladder does not read it -> ATTEMPT link [NOT OBSERVABLE] from logs (the
   field exists on `CandidateTake`; `canonical_identity_chain` carries it).
2. `realization_resolver_authority` rows carry `realization_id`s only; no
   clip_id / text / start-end -> the RESOLVER decision cannot be tied to a clip
   without the Ledger dump -> IDEA/RESOLVER link is [PARTIALLY OBSERVABLE].
3. EDITORIAL FUNCTION: no field exists anywhere -> [MISSING CAPABILITY].
4. The activity of the 21 clean-cut hooks, the 19 composite hooks and the 14
   draft wrappers is not recorded per hook -> RESCUE/PROTECTION link is
   observable only when a hook writes its own diagnostics key.
5. `authoritative_story_placement` (restore units) was not printed in the
   ladder run; the RAW log tail did not include it either.

**Trace A -- straightforward successful take: `clip_829d145e5fc45e30cdac`**
- RAW 13.78-23.28 -> ASR (A-3): "No es secreto para nadie que llevo unos 10
  años trabajando para cruceros. Tenía como costumbre, cada vez que terminaba
  un contrato, hacerme un chequeo de rutina con mi ginecóloga." (words
  `No`…`ginecóloga.`) -> attempt (A-12) [attempt_id NOT OBSERVABLE] ->
  idea: singleton (no `take_judge_groups` row; P-44/P-46 formed no family) ->
  editorial function [MISSING] -> family: none -> candidates: itself ->
  delivery score: not ranked (singleton) -> Best Take (P-52 compose): kept as
  ungrouped material -> rescue/protection: none recorded -> composite: none ->
  final selection (S-10): RESOLVED_WINNER single realization -> plan v2
  fragment (S-13) -> frozen (S-16) -> Boundary in 13.78-23.28 -> S-15
  `keep_complete_idea_envelope` first `No` last `ginecóloga.` 0 s added ->
  Boundary out 13.78-23.28 -> render (R-2) render_start 2.69, rendered 8.71 s,
  `tighten_trailing_silence` cut 0.79 s -> physical raw end 22.49 -> QC: no
  finding on this fragment -> FINAL MP4 2.69-11.40 = RAW 13.78-22.49. Ladder:
  LEVEL-3 for 13.78-22.49; 22.49-23.28 LEVEL-1 boundary `loose_exit_edge`
  (both references end earlier; the renderer's silence rule cut it anyway ->
  the frozen plan is looser than the MP4). Chain complete except links 1-3.

**Trace B -- failed take -> clean retry family: stomach / gastritis
`tg_473eca7ba3cb7667a6`**
- Candidates (A-9/A-12 -> P-1..P-43 kept all three): `clip_b77dbf41`
  236.23-244.19 "Tuve problemas estomacales a un tiempo en donde se me hizo una
  endosco…", `clip_a4ef78c5` 245.39-251.61 "Tuve problemas de estómago en una
  temporada, en 2023, no hay que pregu…", `clip_abcbb706` 258.87-269.37 "Tuve
  problemas de digestión en donde me hicieron una endoscopía y dijeron que
  te…pastillas." (`clip_74445a9c` 251.87-253.35 "Tuve problemas de estómago,"
  was discarded earlier by the clean-cut / hybrid stage -- which hook: NOT
  OBSERVABLE).
- Family (P-44/P-46/P-47): the three above (window chunks 4,5,3 saw the whole
  family, F8). Labels (P-24/P-49): b77dbf41 failed 0.85, a4ef78c5 failed 0.90,
  abcbb706 winner 0.95. Delivery scores (P-48): 0.7292 / 0.6511 / 0.6663
  (`watch_listen_baseline` -- the FAILED take scores highest). Best Take
  (P-50): `single_semantic_winner` -> abcbb706, `semantic_override_applied:
  true` (local winner was the failed b77dbf41). S-3: no-op (gap 0.063 < 0.30).
  S-4: no override recorded. S-10: RESOLVED_WINNER, `legacy_vs_authoritative_
  same: true` (which idea row: NOT OBSERVABLE without the realization map).
  Composite: none. Final selection: abcbb706 only. Plan v2 / freeze: 1
  fragment. Boundary in 258.87-269.37 -> S-15 envelope `Tuve`…`pastillas.`
  0 s -> out 258.87-269.37 -> render start 109.51, 10.54 s, trailing trim 0.0
  -> QC: nothing on this fragment -> MP4 109.51-120.05 = RAW 258.87-269.37.
  Ladder: LEVEL-3 except the 0.98 s tail 268.39-269.37 (both references end
  earlier -- BoundaryEngine-scope in the MP4 rows). This is the canonical
  SUCCESS shape: failed -> clean retry resolved by the hybrid label overriding
  a completeness-blind ranker. It also shows why C-3 matters: had the ranker
  gap been >= 0.30, S-3 would have re-selected the failed take.

**Trace C -- pimples / hair-loss composite region (Gold composite 185.47-189.97
+ 191.77-197.52 + 213.46-221.71 + 226.49-231.74)**
- Acne family `tg_a7d4b8d6299e8803bd`: `clip_9f8d4903` 166.56-182.36 "Por
  temporada me salía en la Por temporada me salía acné en la espalda…" (failed
  0.98, score 0.6138; carries the 2.36 s source silence 173.28-175.64) vs
  `clip_6c372ce1` 185.24-189.84 "Por temporada me salió un acné en la espalda
  con la que yo resolvía con…" (failed 0.90, score 0.6534). Best Take:
  `delivery_tie_break_among_survivors` -> 6c372ce1 (C-2 shape: both failed).
  Both references chose a realization inside 9f8d4903's span at 175.33-175.83
  (0.5 s) and the 185.47-189.97 take -- ladder attributes the 0.5 s to
  BestTakeResolver `take_choice_against_both_references`; the rest of
  9f8d4903 is consensus delete. Final: 6c372ce1; S-15 envelope `Por`…`con` (the
  take ends mid-clause: "…con", continuation "resorcina." is the next clip);
  render 89.36-94.00 = RAW 185.24-189.84.
- Continuation `clip_62108ee5` 191.14-191.74 "resorcina." -- singleton, kept by
  P-52 ungrouped rule; both references remove 191.14-191.61 (0.47 s LEVEL-1
  `failed_or_process_material_retained`, AttemptReconstructor scope: the
  word belongs to the previous delivery, the segmenter split it).
- Pimples family `tg_8819941afa23231b2b` (see C-5): winner ec3ef606
  213.34-222.98 by `single_semantic_winner`; a93a9633 192.44-198.12 DISCARDED
  (both references keep 191.77-197.52 -> 5.75 s LEVEL-1 missing_delivery,
  BestTakeResolver); c041d216 198.88-211.02 discarded (consensus). Composite:
  NONE formed (S-4 no composite; S-10 single realization full critical
  coverage). Boundary: envelope `Otro`…`temporadas.` 0 s; render 94.64-103.04,
  trailing trim 1.24 s -> physical end 221.74 (the frozen plan kept dead air
  to 222.98; the RENDERER removed it). Ladder MP4 rows: 213.46-221.71 LEVEL-3.
- Hair-loss `clip_82625138` 226.74-233.18 -- singleton kept; both references
  end at 231.74 -> 1.37 s LEVEL-1 `failed_or_process_material_retained` tail
  (231.81-233.18 "…y pensaba qu…" abandoned continuation) -- attempt/segment
  scope; render 103.04-109.51 = RAW 226.74-233.18 (no trailing silence, so R-2
  could not help). Gold manifest: `pimples_micro_2_present` FAIL,
  `pimples_micro_order` FAIL (17/19 checks pass).
- Verdict for the region: the composite doctrine is NOT executed here; the
  family competed (one winner) where both references composite two clean
  complementary pieces. Owner: the compete-vs-composite decision (S-10's
  claim-subsumption rule says "safely redundant"; the D-042 slot policy is
  advisory text only).

**Trace D -- conclusion region (Gold 294.87-313.87 + CTA 358.11-361.41)**
- Family `tg_03324733ca1695c430`: `clip_dcc5b751` 295.52-313.50 "Esta es mi
  experiencia. Soy la única en mi familia que tiene este tipo…cuídate." (winner
  0.96, score 0.6321) vs `clip_54d6051e` 319.38-334.24 "Soy la primera en mi
  familia con este tipo de cáncer. Nadie en mi familia tiene un carcino…los"
  (failed 0.88, score 0.6032); `clip_f1400e48` 340.18-346.52 "cánceres son
  hereditarios. Soy la única en mi familia…" discarded (consensus).
- Best Take (P-50): `single_semantic_winner` dcc5b751 -> 54d6051e discarded.
  S-3 no-op. S-4: no override. S-10 (C-4): RESOLVED_COMPOSITE
  `idea_c50529df3b774a8864c4` = {real_8135ee78…, real_d4893cc7…}, legacy winner
  real_d4893cc7…, `legacy_vs_authoritative_same: false` -> 54d6051e RESTORED
  (ladder refinement `restored_by_realization_resolver`; placement after
  dcc5b751; unit diagnostics NOT printed). Composite: YES (authoritative), of
  a complete conclusion + a `failed`-labelled restatement. Plan v2: both
  fragments; frozen. Boundary: both `keep_complete_idea_envelope`
  (`Esta`…`cuídate.`, `Soy`…`los` -- the second envelope ends on the word
  "los", i.e. mid-sentence). Render: dcc5b751 127.25-145.23 (17.98 s),
  54d6051e 145.23-160.12 (14.89 s, no trailing trim), CTA `clip_7c44ce74`
  160.12-165.33 (RAW 356.21-361.42). QC: no finding on these. Ladder:
  295.52-313.50 LEVEL-3; 319.38-334.24 LEVEL-2 `gold_removes_cutai_keeps`
  (Cut.ai keeps a ~22.6 s restatement too); 294.88-295.52 LEVEL-1 `tight_edge`
  entry (0.64 s both references keep -> ASR/segmentation started the take late;
  S-15 could not expand: no earlier ASR word).
- Verdict: Best Take was RIGHT; the Resolver's coverage objective UNDID it.
  No module after S-10 can correct it (D-090 makes S-12 validation-only by
  design). This is the single clearest "later module undoing an earlier
  correct decision" in the system.

**Trace E -- visually/aurally bad region in the latest output: `clip_f8a423f3`
(biopsy -> symptoms)**
- RAW 135.44-149.52 "La biopsia confirmó que era un cáncer papilar de
  tiroides. Síntomas que tuve seg…atrás." Family `tg_993ddc810832431886` with
  `clip_ec29f388` 150.68-158.14 "Síntomas que no me parecían sospechosos pero
  que ahora que lo analizo si eran so…" -- BOTH labelled winner 0.95 (family
  window); scores 0.663 / 0.7153; Best Take: `unresolved_unique_fact_asymmetry`
  -> local winner ec29f388 selected AND f8a423f3 kept (both selected: the
  family is left multi-selected on purpose, D-082 "preserve safe behaviour").
  S-4 CRITICAL_COVERAGE_DOMINANCE: no dominance recorded -> both stay. S-10:
  idea resolved with both (winner + retained? -- realization map NOT
  OBSERVABLE). Plan/freeze: both fragments. Boundary: envelopes unchanged.
  Interior trim (P-65): no split and NO rejection trace for f8a423f3 (C-12).
  Render: f8a423f3 68.18-81.87 (13.69 s, trailing trim 0.39), ec29f388
  81.87-89.36 (7.49 s). QC (R-3): LINGERING_ACCIDENTAL_SILENCE 71.641-73.963
  (2.32 s) = RAW ~138.90-141.22 inside f8a423f3, UNREPAIRABLE (mid-segment) ->
  render attempts 1-2 `PHYSICAL_FAIL_REPAIRED` on other findings, final
  `NEEDS_HUMAN_REVIEW`, candidate NOT deliverable. Ladder: 138.59-140.57 LEVEL-1
  `redundant_realization_both_kept` 1.98 s + tails 148.57-149.13 / 157.45-158.14
  (both references keep both takes but cut the pause and the tails).
- Verdict: the perceptual defect (2.3 s dead air mid-take) has an owner
  (P-65, D-095.2) that did not fire, an objective source measurement that
  disagrees with the render measurement (C-12), and a post-render gate that
  correctly refused delivery. THE PRODUCT IS THE RENDERED VIDEO: this run is
  not deliverable, and the reason is physical, not semantic.

---

## PART 6 -- CUT.AI / HUMAN GOLD / CUTSELL GAP MAP (run 34008386434, four-way ladder)

QA references (QA-ONLY): RAW `Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4`
(366.997 s), CUT.AI `…/D40F1D43-7391-44D5-8D83-09CB62FBF397.MP4` (keep 167.83 s),
HUMAN GOLD `…/5E01F214-A364-4F4B-8F25-D39B1E2B21D2.MP4` (keep 141.67 s). CutSell
frozen plan keep 173.70 s; rendered MP4 165.33 s (23/23 fragments located, 0
order inversions). Selection F1: CutSell vs Cut.ai 0.859, vs Gold 0.821 (Cut.ai
vs Gold 0.900). Physical (MP4) F1: 0.879 / 0.845.

| Level | Frozen plan: selection regions / s | Frozen plan: boundary regions / s | FINAL MP4: selection regions / s | FINAL MP4: boundary regions / s |
|---|---|---|---|---|
| LEVEL 1 (CutSell worse than Cut.ai) | 25 / 33.2 | 16 / 3.6 | 17 / 25.0 | 20 / 3.4 |
| LEVEL 2 (~Cut.ai, Gold better) | 5 / 17.8 | 16 / 1.9 | 4 / 17.3 | 17 / 2.0 |
| LEVEL 3 (matches / exceeds Gold) | 53 / 307.9 | 19 / 2.6 | 52 / 316.3 | 21 / 3.0 |

Previous artifact (run 33995806350, pre-D-095.2/D-094.3): LEVEL-1 selection
60.7 s in the plan / 54.8 s in the MP4. Trend: LEVEL-1 halved; NOT parity.

**LEVEL-1 by responsible authority (FINAL MP4):** BestTakeResolver 7 regions /
16.6 s · BoundaryEngine 24 / 5.5 s · IdeaClusterer/RetryFamilyFormation 3 /
4.2 s · AttemptReconstructor/RecordingProcessRemoval 2 / 1.8 s ·
RealizationResolver 1 / 0.4 s.

**LEVEL-1 regions (MP4), ranked by seconds, with the authority and the collision that produced them:**

| # | RAW range | s | Kind / refinement | Authority | Collision |
|---|---|---|---|---|---|
| 1 | 83.29-89.39 | 6.10 | false_keep -- family winner both references rejected (`clip_59e27e7e`) | BestTakeResolver (+IdeaClusterer) | C-1, C-2 |
| 2 | 191.77-197.52 | 5.75 | missing_delivery -- `clip_a93a9633` lost to `clip_ec3ef606`; both references keep it | BestTakeResolver | C-5 |
| 3 | 104.56-107.48 | 2.92 | false_keep -- `clip_1c5a…psigr` "me mandó a hacer sonografías." never grouped with `clip_333334c8` "Ahí fue cuando me mandaron…" (both kept: redundancy) | IdeaClusterer | C-1 |
| 4 | 138.59-140.57 | 1.98 | false_keep -- pause inside `clip_f8a423f3` (redundant_realization_both_kept) | BestTakeResolver / physical | C-12 |
| 5 | 231.81-233.18 | 1.37 | failed/process tail after hair-loss take | AttemptReconstructor | C-8 |
| 6 | 268.39-269.37 | 0.98 | tail of the gastritis winner | BestTakeResolver (edge) | C-8 |
| 7 | 95.52-96.33 | 0.81 | head of ungrouped retry | IdeaClusterer | C-1 |
| 8 | 157.45-158.14 | 0.69 | tail (both kept) | BestTakeResolver (edge) | C-8 |
| 9 | 294.88-295.52 | 0.64 | tight entry of the conclusion (cuts before "Esta") | BoundaryEngine | C-6 |
| 10 | 148.57-149.13 | 0.56 | tail | BestTakeResolver (edge) | C-8 |
| 11 | 82.29-82.82 | 0.53 | tight entry "Al terminar" | BoundaryEngine | C-6 |
| 12 | 48.45-48.97 | 0.52 | tight entry "El año pasado…" | BoundaryEngine | C-6 |
| 13 | 175.33-175.83 | 0.50 | missing_delivery inside the discarded acne take | BestTakeResolver | C-2 |
| 14 | 123.92-124.39 | 0.47 | ungrouped retry tail | IdeaClusterer | C-1 |
| 15 | 191.14-191.61 | 0.47 | "resorcina." orphan continuation | AttemptReconstructor | segmentation |
| 16 | 275.71-276.09 | 0.38 | tight entry -- the negation "No" of "No quiero sonar a conspiración" is cut: MEANING INVERSION, mis-scored as a 0.38 s boundary nit by the ladder | BoundaryEngine (root: segmentation/ASR) | C-6, Appendix A |
| 17 | 319.38-319.75 | 0.37 | head of the restored restatement | RealizationResolver | C-4 |
| + 20 boundary-scope rows (< 0.35 s each, 3.4 s): loose exits 0.79-1.14 s on `clip_d3d8` (11.05-11.89), `clip_829d` (22.49-23.28), `clip_eda4` (45.37-46.42) -- cut by the renderer in the MP4 -- and the entry/exit nudges above | | | BoundaryEngine | C-8 |

**LEVEL-2 regions:** the restored conclusion restatement 319.75-334.24 (14.5 s,
C-4; Cut.ai keeps a comparable restatement, Gold does not) plus ~2.8 s of
edge preferences where Gold cuts tighter than Cut.ai.

**Level-1 read-out against the directive's six commercial criteria:**
1. recording-process removal: mostly reached (consensus deletes are LEVEL-3);
   residual 1.8 s of failed tails (#5, #15).
2. basic retry resolution: NOT reached -- #1, #2, #3, #13 (15.3 s) are all
   retry-family failures (missed member, all-failed tie-break, compete-vs-
   composite).
3. take quality: NOT reached -- a `failed`-labelled take wins by tie-break
   (#1) and the DeliveryScorer ranks a failed take first (Trace B).
4. visual cleanliness: NOT reached -- 2.3 s dead air mid-take (#4) makes the
   candidate non-deliverable; loose exits survive to the plan.
5. basic continuity: partially -- 0 order inversions in the MP4 (fixed since
   33995806350); 6 abrupt audio joins flagged by QC.
6. basic redundancy removal: NOT reached -- #3 (two "sonografías" sends) and
   the LEVEL-2 conclusion restatement.

Widespread Level-1 failures are NOT hidden behind the Level-2 conclusion
issue: 25.0 s of LEVEL-1 in the MP4 vs 17.3 s of LEVEL-2.

---

## PART 7 -- VERIFICATION OF THE EXISTING EDITORIAL DISCOVERIES

| Accepted conclusion | Where it exists in production code | Verified behaviour on the audited path | Can a later authority undo it? |
|---|---|---|---|
| GOOD + GOOD does not mean KEEP BOTH; two complete realizations of the same editorial function COMPETE -> ONE WINNER | (a) `pipeline._semantic_best_take` (one winner per family; "unresolved_unique_fact_asymmetry" keeps BOTH -- a deliberate fail-open, D-082); (b) `deterministic_best_take_authority` (clear gap only); (c) `claim_coverage_best_take` CRITICAL_COVERAGE_DOMINANCE (multi-selected families: one dominant wins, else both stay); (d) `realization_resolver._resolve_one_idea` (one winner IF one realization covers all CRITICAL groups); (e) D-042 policy TEXT in the semantic-equivalence prompt (family formation) | Exists at family level only. It is defeated whenever (1) the two realizations are not in one family (C-1: sonografías; the conclusion's third restatement `clip_f1400e48` was in the family, the second was too, yet…), or (2) the loser carries an extra CRITICAL-classified claim -> (d) forms a composite instead (C-4), or (3) both carry asymmetric unique facts -> (a)/(c) keep both (Trace E). | YES -- S-10 undid P-50's correct single-winner in the conclusion; nothing after S-10 can restore the single winner (D-090). |
| Complementary incomplete pieces where neither alone is sufficient -> MINIMAL COMPOSITE | `realization_resolver._find_minimal_composite` (sizes 2..N, all CRITICAL groups covered, no redundant member, temporal compatibility, contradiction-free); `claim_coverage_best_take` composites; CompositeResolver hooks (`hybrid_semantic_composite_bridge`, `hybrid_composite_best_take`) | Exists, but the trigger is CLAIM COVERAGE, not INCOMPLETENESS: a composite forms whenever a single realization misses a CRITICAL group (C-4: complete conclusion + failed restatement) and does NOT form when the pieces' claims subsume each other (C-5: pimples, both references composite). The doctrine's "neither alone is sufficient" is never evaluated as delivery sufficiency. | YES/NO -- S-10 is final; its composites stand. |
| Required proposition vs supporting / rephrased / elaborative detail | `semantic_claims` importance rules (negation, correction, unit-quantity, generalising statistic, result-state, diagnosis, entity-relation -> CRITICAL); `_effective_importance` downgrade for low-information incidental claims; D-089 effective-importance index; D-042 policy text | Exists as a DETERMINISTIC LINGUISTIC CLASSIFIER of claims, not as an editorial judgment of what the slot REQUIRES. Any generalising statistic, number with a unit, negation or result-state sentence in a rephrase becomes a "required" group. This is exactly the mechanism of C-4. | n/a (it is an input to S-4/S-10). |
| Human Gold targets the MINIMUM SUFFICIENT EDITORIAL SET, not MAXIMUM SEMANTIC COVERAGE | Only as prompt text (`editorial_slot_resolution_install._SLOT_RULES`, `_SEMANTIC_EQUIVALENCE_POLICY`) and in docs (`docs/CUTSELL_DECISION_D042_EDITORIAL_SLOT_RESOLUTION.md`, D-095) | NOT implemented as an engine objective. The engine's authoritative objective (`_resolve_one_idea`: "critical_group_ids.issubset(coverage)") is maximum CRITICAL coverage per idea; StoryValidator's `_lost_critical_claims`/`_lost_semantic_atoms` and FinalEditReviewer's CRITICAL_CLAIM_LOST / UNIQUE_FACT_LOST enforce coverage from the other side (blocking Freeze on loss). The whole post-BestTake stack is coverage-protective by construction. | The doctrine cannot be undone because it is never applied. |

Conclusion for Part 7: the discoveries are correctly RECORDED and partially
IMPLEMENTED at the family/winner level, but the authoritative semantic
objective on this head is the opposite objective (coverage maximisation), so
the accepted doctrine is structurally overridable -- and was overridden in the
audited RAW (C-4) while the reverse error (compete instead of composite)
happened in the pimples region (C-5).

---

## PART 8 -- SYSTEM WATCH + LISTEN AUDIT

**A. TECHNICAL / STRUCTURAL POST-RENDER QC (IMPLEMENTED, ACTIVE, BLOCKING)**
`live_render_qc.render_with_post_render_qc` -> `post_render_watch_listen_qc.check_render_plan_covers_edit_plan`,
`check_render_sequence_matches_edit_plan`, `check_no_duplicate_render_segments`
(structural, attempt 1) + `post_render_media_qc.run_post_render_media_qc`
(decoded MP4) + `run_bounded_physical_repair_loop`/`live_boundary_repair`
(edge trims only) -> `deliverable`. Verified live in the RAW (NEEDS_HUMAN_REVIEW,
candidate withheld).

**B. PERCEPTUAL SYSTEM WATCH + LISTEN (MISSING)** -- `PostRenderWatchListenQCProvider`
is a `typing.Protocol` with the docstring "Not implemented in Clean Cut Core V1
-- no provider is constructed or invoked anywhere in the active pipeline".

| Check | Evaluates decoded MP4? | Audio? | Frames? | Status | Blocking? |
|---|---|---|---|---|---|
| decode / export integrity (`probe_decode_integrity`) | yes | yes | yes | IMPLEMENTED, ACTIVE | BLOCKING |
| plan coverage / sequence / duplicate segments (structural) | plan vs segments (not the media) | -- | -- | IMPLEMENTED, ACTIVE | BLOCKING |
| lingering accidental silence >= 1.2 s @ -35 dB | yes | yes | -- | IMPLEMENTED, ACTIVE | BLOCKING (unrepairable mid-segment) |
| frozen / repeated frames (>= 0.5 s) | yes | -- | yes | IMPLEMENTED, ACTIVE | BLOCKING |
| dead black frames (>= 0.3 s) | yes | -- | yes | IMPLEMENTED, ACTIVE | BLOCKING |
| abrupt audio discontinuity at joins (PCM jump ratio) | yes | yes | -- | IMPLEMENTED, ACTIVE | BLOCKING (repairable by 0.05 s trims) |
| facial / body continuity across a cut | no | -- | no | MISSING (no producer) | -- |
| gesture continuity / abandoned gesture | no | -- | no | MISSING | -- |
| reset debris at entry/exit (`RESET_DEBRIS` declared) | no -- only PRE-render evidence (`human_boundary_polish_v5`, interior trim) | -- | no | MISSING post-render | -- |
| entry / exit quality (ugly first frame, mid-gesture exit) | no | -- | no | MISSING | -- |
| clipped words / phonemes (`CLIPPED_WORD`, `UNSAFE_WORD_BOUNDARY` declared) | no -- word lock is PRE-render (ASR words) | no | -- | MISSING post-render (no re-ASR of the MP4 for the gate; `speech_visual_microtrim` re-ASRs only when already deliverable) | -- |
| breath cuts / cut inside a breath | no | no | -- | MISSING | -- |
| conversational cadence / rhythm across joins | no | no | -- | MISSING | -- |
| visual jumps / framing jumps (`FRAMING_INTEGRITY` declared) | no | -- | no | MISSING | -- |
| performance continuity (energy / expression across the join, `AWKWARD_POST_LINE_EXPRESSION` declared) | no | -- | no | MISSING | -- |
| repeated audience-facing content (the same sentence twice) | no -- only PRE-render semantic checks (DUPLICATE_IDEA on the plan) | no | -- | MISSING post-render | -- |
| recording-process residue visible in the output | no | -- | no | MISSING | -- |
| A/V perceptual continuity / sync (`AV_SYNC_DRIFT` declared) | no | no | no | MISSING | -- |
| routing of a perceptual failure to the owning authority (`routes_to` field exists on `PostRenderFinding`) | contract only | | | DOC/CONTRACT-ONLY | -- |
| "failing candidate never presented as successful preview" | yes (D-036 gate) | | | IMPLEMENTED, ACTIVE | BLOCKING |

Desired order `RENDER -> TECHNICAL QC -> PERCEPTUAL SYSTEM W+L -> HUMAN W+L`:
positions 1-2 exist and block; position 3 does not exist; position 4 has no
artifact (a human verdict is not recorded anywhere machine-readable, so "HUMAN
WATCH+LISTEN PASS" cannot be asserted by the pipeline).

---

## PART 9 -- SHOULD-BE vs AS-IS GAP TABLE

Recommended Action vocabulary: KEEP / MERGE / RETIRE / REWIRE / IMPLEMENT /
SIMPLIFY / INVESTIGATE. None of MERGE/RETIRE/REWIRE/SIMPLIFY is executed by
this audit.

| # | Capability | SHOULD BE | ACTUALLY IS | GAP | ROOT CAUSE | Severity | Action |
|---|---|---|---|---|---|---|---|
| G-1 | One semantic Selection authority | one owner decides KEEP/DISCARD per idea with the minimum-sufficient-set objective | S-10 is the declared owner, but P-50 (draft Best Take), S-1 (retry arbiter x2), S-3, S-4 and 14 draft wrappers all edit membership before it, and S-10's objective is coverage maximisation | objective mismatch + 20 pre-authority editors | incremental RAW-driven additions (D-023 kept all historical hooks); Resolver designed as a claim-coverage ledger (D-050C1) | P0 | INVESTIGATE (objective) + SIMPLIFY (pre-authority stack) -- after approval |
| G-2 | Retry family formation completeness | every retry of an intended idea lands in one family before Best Take | lexical tier + bounded Gemini pair budget + cohesion split; misses (C-1) turn into "ungrouped = keep" false keeps | 4.2 s LEVEL-1 + the 6.1 s failed winner depends on it | pair budget / coverage-first ordering cannot see every pair; no post-hoc "kept idea already delivered" check outside StoryValidator's DUPLICATE_IDEA (which is family-based) | P0 | INVESTIGATE (why the `1c5a0826` pair was never asked / rejected: `semantic_idea_equivalence` blocked/budget rows) |
| G-3 | Best Take among only-failed members | a family with no clean member yields NO KEEP (or escalates), never a tie-break survivor | `_semantic_best_take` step 1 falls open when every member is delete-recommended | 6.1 s + 0.5 s LEVEL-1 | D-081/D-082 "WHEN UNCERTAIN, KEEP" applied to a tie among failed takes | P0 | INVESTIGATE -> small targeted change in P-50 (owner), no new layer |
| G-4 | Compete vs composite decision | decided by delivery sufficiency and editorial function | decided by CRITICAL-claim set arithmetic (subsumption -> compete; extra group -> composite) | C-4 and C-5 (opposite errors) | no engine representation of editorial function / sufficiency | P0 | INVESTIGATE (design) -- Part 11 |
| G-5 | Editorial function / slot understanding | an engine field per candidate/idea, used by family formation and competition, observable in diagnostics | prompt policy text only (D-042) | MISSING capability | D-042 was implemented as a prompt injection because family formation was the cheapest lever | P1 | IMPLEMENT (design first) |
| G-6 | Objective take-cleanliness signal for Best Take | delivery score includes measured verbal/visual cleanliness (dead air, resets, fumbles inside the take) | DeliveryScorer = completeness + duration fit + lexical fragment penalties + MediaSignals penalty; audio silence events exist (D-095.2) but are not a scoring feature | failed takes can out-score clean ones (Trace B: 0.7292 vs 0.6663) | scorer predates the evidence sources | P1 | INVESTIGATE (feature design; keep the ladder to measure it) |
| G-7 | Interior dead-air removal proven in the video | a >= 1.2 s pause inside a kept take never reaches the render | ran (1 split) but missed the 2.32 s pause QC found (C-12); also runs before the Resolver (C-7) | candidate non-deliverable | measurement disagreement + ordering | P0 (blocks delivery) | INVESTIGATE (C-12) + REWIRE (C-7) after approval |
| G-8 | Boundary after Freeze owns entries/exits | one Boundary pass after Freeze with audio+visual evidence for entry and exit | S-15 can only expand; S-17 only micro reset gaps; the renderer's trailing-silence rule does the real exit work; entries have no owner | 20 boundary rows / 3.4 s + negation-cut entry | Boundary responsibilities spread over P-51/P-64..P-67/S-15/S-17/R-2/R-3 | P1 | MERGE (design) after approval |
| G-9 | Physical work strictly after Freeze | no source-range change before the semantic authority except candidate shaping | P-51 edge trims and P-64..P-67 run inside the draft build | ownership leak, restored clips skip them | pre-V1 layering | P2 | REWIRE after approval |
| G-10 | Perceptual System Watch+Listen gate | blocking, routes to owner, never changes membership | Protocol + constants, no producer | MISSING | never built (D-025/D-028 built the technical checks) | P0 for the ladder | IMPLEMENT (after G-1/G-3/G-7 decisions; see Part 11) |
| G-11 | Human Watch+Listen verdict | recorded, machine-readable, tied to benchmark id + identity | none | MISSING | -- | P2 | IMPLEMENT (small: a verdict manifest) |
| G-12 | Active-path markers | every canonical component proves it ran | 18/20; S-3 writes nothing when no-op; S-9 probe key never written; 21+19+14+5 hooks have no markers | markers ambiguous | probes written from expectations, not from code | P2 | KEEP + small IMPLEMENT (marker fixes) |
| G-13 | Trace observability (Part 5 gaps 1-5) | attempt_id, realization->clip map, story placement units, hook activity printed | partially | cannot produce a complete end-to-end trace from logs | -- | P1 | IMPLEMENT (printing only) |
| G-14 | Source-vs-render silence consistency | one silence measurement | two measurements disagree (C-12) | D-095.2 unproven | unknown | P0 | INVESTIGATE (needs the media; offline replay on the RAW file) |
| G-15 | KEEP/DISCARD only (D-019) | no SWAP writes | S-1/S-2/S-10 write `alternates`, D-092 folds | cosmetic | legacy | P3 | SIMPLIFY later |
| G-16 | Dead code on the path | none | 4 uninstalled installers, `speech_safe_dead_air_guard` patching v3, polish v1-v4, OpenAI providers | confusion, test cost | history | P3 | RETIRE later (not now) |
| G-17 | Image reproducibility | digest-pinned base image | tag-pinned `madiator2011/better-pytorch:cuda12.4-torch2.6.0` | engine identity is fine (mounted checkout), runtime libs are not pinned | infra choice (D-043) | P2 | INVESTIGATE (pin) |
| G-18 | Test coverage of active Boundary/repair code | every active module has tests | `live_boundary_repair` 0 files, 6 active wrappers 0 files | regression risk | -- | P2 | IMPLEMENT tests |

---

## PART 10 -- DUPLICATION / COMPLEXITY REPORT

Counts on the active RAW path (this head):

| Question | Count | Members |
|---|---|---|
| Authorities that can alter Selection membership | **32** | apply_clean_cut + 21 hooks (as one set counts 22), hybrid_session_cleanup, 19 composite hooks (set), compose_selected/selection_integrity, 8 draft wrappers (final_draft_retry_integrity, selected_failed_bridge_integrity, round8/9, round11, short_bts, post_selection_incomplete_bridge_authority, post_selection_internal_retake_trim, final_selection_retry_arbiter), apply_selection_phase_authority, selection_conflicted_bridge_guard, deterministic_best_take_authority, claim_coverage_best_take, StoryValidator pass 1 (legacy), realization_resolver (authoritative), D-092 fold. Counting sets as one: 13 distinct membership editors before the declared authority. |
| Authorities that can RESTORE discarded content | **8** | story_coverage_guard (clean-cut hook), hybrid_story_guard, hybrid_failed_soft_restore, hybrid_semantic_complementary_rescue, hybrid_complementary_delivery_guard, composite_family_stabilization (step 16), claim_coverage_best_take, realization_resolver |
| Authorities that alter retry-family behaviour | **9** | take_grouping.group_takes (lexical), local_retry_grouping, retry_group_integrity, final_sibling_grouping, session_grouping_bridge, global_session_sibling_bridge, apply_composite_group_split, reconcile_semantic_idea_equivalence (+D-042 injection), split_incohesive_retry_groups; plus StoryValidator residual-family resolution (legacy pass) |
| Involved in Best Take | **6** | take_judge (score), family_scoped_semantic_decisions, _semantic_best_take (+semantic_best_take_integrity), final_selection_retry_arbiter (x2), deterministic_best_take_authority, claim_coverage_best_take, realization_resolver `_pick_winner` |
| Involved in composites | **6** | hybrid_semantic_composite_bridge, hybrid_composite_best_take, apply_composite_group_split, composite_family_stabilization, claim_coverage_best_take, realization_resolver `_find_minimal_composite` (+ canonical_edit_plan validation, StoryValidator INCOMPATIBLE_COMPOSITE) |
| Involved AFTER the initial draft Selection (compose_selected) | **22** | 14 draft wrappers, S-1..S-4, S-5, S-10, D-092 fold, S-12 (validation), S-13 (order repair) |
| Involved in Boundary (source-range changes) | **10** | refine_takes_with_temporal_context + 5 edge hooks (set), post_selection_edge_only_boundary, post_selection_interior_gap_trim, post_selection_continuity_coalescer, audio_boundary_completion, enforce_complete_idea_boundaries (+2 guards), polish_human_boundaries_v5, render.tighten_trailing_silence, live_boundary_repair, speech_visual_microtrim |
| Involved in post-render QC | **4 modules / 8 checks** | post_render_watch_listen_qc (3 structural), post_render_media_qc (5 media), live_render_qc (loop + gate), live_boundary_repair (repair) |

**Duplications -- why each exists, what introduced it, is the need still valid, can one owner take it:**

| # | Duplication | Why each exists / what introduced it | Original need still valid? | One canonical owner possible? |
|---|---|---|---|---|
| D-1 | 22 recording-process heuristics around `apply_clean_cut` (D-002/D-013 era, one per RAW regression: rounds 4-11) | each fixed one Video00 symptom (restart questions, frustration, word search, script consult, product handling…) | the CLASS of need (recording-process removal) is valid and Level-1; the per-symptom split is not observable or ownable | YES: RecordingProcessRemoval as one component with one decision record and one evidence model (audio silence + local performance + lexical restart), keeping the heuristics as internal rules, not wrappers |
| D-2 | 19 CompositeResolver hooks: 4 restore paths, 2 composite paths, 5 integrity passes (D-023 consolidated the CALL, not the logic) | each hook answered one hybrid-label failure mode before D-081 made deletion evidence-first | after D-081 the hybrid stage rarely deletes (0 deletions in the RAW), so most restore hooks guard against a deletion that no longer happens | YES: since S-10 owns restoration by claim proof (D-076/D-079), the pre-Resolver rescue hooks are candidates to become evidence producers only -- to be proven by CleanCutBench before any retirement |
| D-3 | Draft-level Selection stack (14 wrappers) + S-1 repeating three of them | pre-V1 architecture where `build_flow_b_draft` was the final answer; S-1 was added to "execute Selection in one place" but kept the wrappers | the retry structures they detect (bridges, orphan prefixes, superseded attempts) are real | YES: fold into the Resolver's candidate model or into IdeaClusterer evidence; today they are invisible (no markers) |
| D-4 | Three claim-coverage decision makers: `_semantic_best_take` step 3 (dominance), `claim_coverage_best_take`, `realization_resolver` requirement groups -- same claims, same importance rules | D-038 (ClaimCoverageBestTake) predates the Resolver (D-050); D-082 reused dominance inside P-50 | one coverage decision is enough once S-10 is authoritative | YES: S-10 (already the declared owner) -- but only after G-1's objective question is settled |
| D-5 | Two StoryValidator passes, two plan/review/repair passes | D-050C3: legacy evidence retained for the Ledger + comparison | evidence value only | YES: keep pass 2; pass 1 becomes a diagnostics-only ledger input (already the intent) |
| D-6 | Two coalescers (post_selection_continuity_coalescer, render_plan._coalesce_contiguous_segments) and three tail trimmers (S-15 guards, tighten_trailing_silence, live_boundary_repair) | added at different layers for different RAWs | valid need, wrong layering | YES: one BoundaryEngine pass after Freeze (G-8) |
| D-7 | Two Best-Take resolvers with opposite priors (P-50 semantic-first, S-3 local-score-first) | Phase 0/1 rebalance promoted S-3 for the Unified-Selection era (D-021 "PROMOTED"); Clean Cut V1 kept both | S-3's need (a decisive local ranker beating an ambiguous whole-video reasoner) no longer exists on this path | YES: P-50 (+S-10 tiers) is the effective owner; S-3 is a no-op with a latent flip risk |
| D-8 | Two freezes (P-63 premature, S-16 real) and two boundary invariants (P-68, S-18) | pre-V1 holdover (D-025 documents it) | only the real one | YES: S-16/S-18 |

Fewer modules is not automatically better; the recommendation is clear
ownership: RecordingProcessRemoval (D-1), IdeaClusterer (D-3 evidence),
BestTakeResolver (D-7 -> P-50 + S-10 tiers), Realization/Composite authority
(D-2, D-4), Boundary (D-6), Freeze (D-8).

---

## PART 11 -- PROPOSED CLEAN CANONICAL TARGET ARCHITECTURE (PROPOSAL ONLY)

Evaluation of the conceptual sequence proposed by the Product Owner against the
evidence above. Deviations are explained; nothing here is implemented.

```
1. Attempt Detection                       KEEP as is (A-9 + A-12). Add: negation-particle rejoin (Appendix A) as a
                                            segmentation rule; attempt_id printed in every result.
2. Recording Process Removal               ONE component (today: 22 hooks + hybrid mechanical deletes). Same rules,
                                            one decision record per candidate ("removed because…"), fed by the three
                                            objective evidence sources that already exist: audio dead air (A-7),
                                            local performance events (A-5), lexical restart evidence.
3. Idea / Editorial Function Understanding IdeaClusterer keeps both tiers. ADD the missing engine field:
                                            editorial_function (hook/setup/symptom/diagnosis/reflection/conclusion/CTA…)
                                            per attempt, produced by the SAME bounded Gemini arbiter call that already
                                            judges idea equivalence (no new provider), recorded in diagnostics.
4. Retry Family Formation                  = IdeaClusterer output + cohesion split (KEEP). Deviation: add a
                                            completeness check "a kept ungrouped candidate whose editorial function
                                            AND idea are already delivered by a kept family winner" -> it must enter
                                            that family (the C-1 shape) BEFORE Best Take, not be left to
                                            StoryValidator's DUPLICATE_IDEA after the fact.
5. Clean Take Ranking                      DeliveryScorer + an objective cleanliness term (measured dead air inside
                                            the take, reset events inside the take, fragment penalties). Hybrid
                                            failed/winner labels stay family-scoped evidence (F8).
6. Best Take / Minimum Sufficient          ONE resolver per family: (i) a family with no clean member yields NO KEEP
   Realization                              (escalate, never tie-break); (ii) one complete clean realization of the
                                            function wins; (iii) coverage of CRITICAL claims is a SAFETY VETO on
                                            discarding a claim that appears nowhere else in the KEEP set -- not an
                                            objective that manufactures composites. This is the inversion of S-10's
                                            current objective and is the single most consequential design decision
                                            (needs Product Owner approval: it changes editorial policy).
7. Composite only when necessary           composite iff NO single realization is a sufficient delivery of the
                                            function (incompleteness / abandoned continuation), pieces are clean and
                                            temporally compatible -- never because a rephrase adds a statistic.
8. Commercial Clean Cut                    = the KEEP set after 2-7 (Cut.ai-parity target of the ladder).
9. Human-Gold Editorial Refinement         minimum-sufficient-set pass over the KEEP set: a second complete
                                            realization of an already-delivered function is removed unless it advances
                                            a genuinely new required proposition (D-042 rules as ENGINE logic with a
                                            diagnostics record; today prompt text only).
10. Story Validation                       KEEP (validation-only, D-090) -- with CRITICAL_CLAIM_LOST downgraded to a
                                            veto that can only be satisfied by a claim present in NO kept realization
                                            (aligned with 6.iii); contradiction gate unchanged (D-020).
11. Selection Freeze                       KEEP (S-16). Retire the premature freeze (P-63) once the wrappers move.
12. Boundary Polish                        ONE BoundaryEngine AFTER Freeze owning entry, exit and interior physical
                                            cuts with audio (A-7) + visual (A-5) evidence, including the work now done
                                            by P-51/P-64..P-67 (before Freeze) and by the renderer's trailing-silence
                                            rule. Runs on the FINAL KEEP set, so restored/composited clips are covered
                                            (fixes C-7). Renderer becomes range-faithful.
13. Render                                 KEEP (R-1/R-2) minus the hidden trimming.
14. Technical QC                           KEEP (R-3), blocking, unchanged.
15. Perceptual System Watch + Listen       IMPLEMENT as a BLOCKING gate on the decoded MP4 with routing: reset debris /
                                            entry-exit quality / clipped phonemes / breath cuts / visual jumps /
                                            performance continuity -> BoundaryEngine; repeated audience-facing content
                                            / wrong take -> Selection (6/9); A/V defects -> Renderer. Never changes
                                            membership; a failing candidate is never a preview.
16. Human Watch + Listen                   record a verdict manifest per benchmark id + identity block.
```

Deviations from the proposed sequence and why: (a) "Clean Take Ranking" and
"Best Take" are kept as two steps but ONE owner (the score is evidence, the
resolver decides); (b) "Human-Gold Editorial Refinement" (9) is placed AFTER
the commercial clean cut and BEFORE Story Validation, as proposed -- the
evidence supports it because Level-1 failures (2, 4-7) are upstream of it and
Level-2 (the restatement) is exactly its job; (c) Story Validation stays
validation-only (D-090) -- the audit found no reason to give it authority
back; (d) the Semantic Ledger and the Resolver's per-idea model are KEPT as
the decision substrate -- only the OBJECTIVE changes (6.iii), not the
machinery; (e) SWAP stays out of scope (D-019).

One owner per decision in the target: candidate shape -> AttemptReconstructor;
removal of process material -> RecordingProcessRemoval; family membership ->
IdeaClusterer; winner / composite / minimum set -> RealizationResolver (with
the inverted objective); coverage safety -> StoryValidator (veto only);
physical ranges -> BoundaryEngine (after Freeze); perceptual acceptance ->
System Watch+Listen (routing only).

---

## PART 12 -- STOP. DELIVERABLES, RANKED ROOT CAUSES, NEXT PLAN (awaiting approval)

No rewrite, no deletion, no merge of authorities, no refactor and no paid RAW
were performed for this audit. The only working-tree change made during the
D-095.3 investigation (a negation-particle rejoin in `take_segmentation.py`)
has been REVERTED from the tree and is preserved verbatim in Appendix A as a
proposal.

**Deliverables in this document:** 1. AS-IS map (Part 2 + Part 3). 2. SHOULD-BE
map (Part 11). 3. Gap table (Part 9). 4. Authority collision map (Part 4).
5. Duplication report (Part 10). 6. Active-path report (Part 1 + Part 3
STATUS column + 3E). 7. Cut.ai / Human Gold / CutSell gap map (Part 6).
8. Watch+Listen gap report (Part 8). 9. Ranked root causes (below).
10. Recommended next implementation plan (below).

### Answer to the primary question

The rendered CutSell MP4 is still substantially worse than Cut.ai and far from
Human Gold for FIVE proven, ranked causes -- none of them "missing capability
in general" and none requiring a large rewrite:

| Rank | Root cause | Category (directive vocabulary) | Evidence | Seconds (MP4) |
|---|---|---|---|---|
| 1 | The authoritative Resolver's objective is CRITICAL-claim coverage maximisation; it restores a `failed`-labelled restatement into a composite after Best Take correctly chose one winner, and it decides compete-vs-composite by claim arithmetic (both directions wrong) | wrong authority objective -> later module undoing an earlier correct decision; over-preservation; incorrect composite behaviour | C-4, C-5, Trace C, Trace D, Part 7 | 14.5 (L2) + 5.75 (L1) |
| 2 | Best Take elects a survivor among only-failed members (fail-open tie-break) | incorrect Best Take scoring / policy | C-2, Trace C (acne), region #1 | 6.6 |
| 3 | Retry-family formation misses a member; "ungrouped = keep" then plays the retry and its failed sibling | incorrect retry-family formation + structural default | C-1, regions #3/#7/#14 | 4.2 (+ enables #1's 6.1) |
| 4 | Physical ownership is split across five pre-Freeze editors, one post-Freeze polish and two renderer-side trimmers; entries have no owner; the D-095.2 interior trimmer runs before the authority and its source measurement disagrees with the render measurement | Selection/Boundary responsibility leakage + Boundary/render failure + inactive-in-effect implementation | C-6, C-7, C-8, C-12, Trace E, Part 6 boundary rows | 5.5 (+ the 2.3 s pause that blocks delivery) |
| 5 | No perceptual System Watch+Listen exists; the technical QC catches only silence/black/frozen/join defects, so everything visual (reset debris, entry frames, gesture continuity) passes to the human | missing capability / insufficient post-render QA | Part 8 | unmeasured by the pipeline |

Secondary (real, lower impact): identity markers ambiguous (G-12); trace
observability incomplete (G-13); dead/legacy modules (G-16); two Best-Take
resolvers with opposite priors, latent (C-3); DeliveryScorer ranks a failed
take first (G-6); segmentation isolates a bare negation and a downstream
guard deletes it (Appendix A).

Explicitly NOT the cause in this RAW: the rescue/complementary hooks (0
hybrid deletions, no rescue fired); the Gemini arbiters (their merges were
correct where asked); the render/QC gate (it correctly refused delivery);
code-vs-video mismatch (identity proves the engine is HEAD).

### Recommended next implementation plan (needs Product Owner approval; nothing started)

Step 0 -- Observability only, no behaviour change (safe to authorise
independently): print `attempt_id`, the realization->clip map and
`authoritative_story_placement` in both workflows; fix the two identity
probes (S-3 writes a `no_op` record; S-9 writes a marker when the policy is
injected); add per-hook activity counters for the clean-cut / composite /
draft-wrapper sets. Closes G-12, G-13.

Step 1 -- Product decision (escalation A): adopt Part 11 item 6.iii -- the
Resolver keeps its ledger and tiers, but CRITICAL coverage becomes a veto on
losing a claim that exists in NO kept realization instead of a composite-
forcing objective; and item 6.i -- an all-failed family yields no KEEP by
tie-break. Both are editorial policy changes; both target root causes #1 and
#2 in their owning modules (`realization_resolver._resolve_one_idea` /
`_find_minimal_composite`; `pipeline._semantic_best_take`). No new layer.

Step 2 -- Root cause #3 investigation (no code): read the
`semantic_idea_equivalence` blocked/budget rows for the `1c5a0826` pair; then
decide between raising the pair-budget coverage for kept-idea overlaps or the
Part 11 item 4 completeness check (small, in IdeaClusterer).

Step 3 -- Root cause #4: (a) INVESTIGATE C-12 offline on the RAW file
(silencedetect on the source vs the rendered segment; no GPU needed);
(b) REWIRE the interior trimmer to run on the final KEEP set after the
authority (C-7) -- a move, not a new layer; (c) design the single post-Freeze
BoundaryEngine (G-8) -- design only until approved.

Step 4 -- Root cause #5: implement the perceptual System Watch+Listen gate
(Part 8 rows marked MISSING, starting with reset debris / entry-exit quality
/ clipped phonemes using A-5 + A-7 evidence re-measured on the decoded MP4),
BLOCKING, routing only. Requires the Product Owner's go-ahead on the gate's
acceptance criteria (escalation A).

Step 5 -- Only after Steps 1-3 pass offline qualification (CleanCutBench both
modes, full `tests/test_cutsell_*.py`, ladder replay on the persisted
result): ONE authorised RAW, ladder, MP4 inspection, repeat toward Cut.ai
parity, then Gold parity, then perceptual gate, then Human Watch+Listen.

Retire/merge candidates (Part 10) are NOT scheduled; they need CleanCutBench
proof per module and a separate approval.

HUMAN ACTION REQUIRED: YES -- escalation A (product/editorial policy in Steps
1 and 4) and D (any structural change). Step 0 and Step 2 are technical and
can proceed on request without changing behaviour.

---

## APPENDIX A -- Parked proposal D-095.3 (NOT applied; working tree reverted)

Context: in the audited RAW the segmenter isolated a bare "No" (`clip_37c525bd`,
269.37-270.17) as its own candidate; `semantic_fragment_guard` then deleted it
as `semantic_failed_micro_fragment` (failed 0.99), and the kept clip
`clip_46af66ed` starts at "quiero sonar a conspiración…" (276.09). Both
references keep a "No" at 275.71-276.09 that no CutSell candidate contains, so
the negation in the output is lost either way (ladder region #16, 0.38 s
`tight_edge`, a meaning inversion mis-scored as a boundary nit). Two owners are
involved: take segmentation (should not isolate a bare negation particle from
the clause it negates) and the guard (should never delete a bare negation).
Because the ASR-visible "No" sits 5.9 s before the clause, the rejoin below
(max gap 2.0 s) would NOT have fixed this exact instance; it remains a
general segmentation improvement. Also required by Part 9 G-13: word-level
timings of that region printed, to establish whether ASR dropped the second
"No". Status: PROPOSED; needs the Part 12 approval before tests/commit.

```diff
diff --git a/cutsell_worker/take_segmentation.py b/cutsell_worker/take_segmentation.py
index b98c42b..fcf1c41 100644
--- a/cutsell_worker/take_segmentation.py
+++ b/cutsell_worker/take_segmentation.py
@@ -35,6 +35,8 @@ def _canonical_normalization_enabled(env: Mapping[str, str] | None = None) -> bo
 # This is intentionally multilingual for the English/Spanish creator footage used by
 # Clean Cut. Ending an ASR chunk on one of these tokens is strong evidence that the
 # chunk boundary is transcription segmentation, not an editorially valid cut point.
+_TOKEN_RE = re.compile(r"[0-9A-Za-z\u00C0-\u024F']+")
+
 _BRIDGE_CONNECTORS = frozenset({
     # English
     "a", "an", "and", "as", "at", "because", "but", "by", "for", "from", "i", "if",
@@ -189,6 +191,29 @@ def _join_takes(left: CandidateTake, right: CandidateTake) -> CandidateTake:
     )
 
 
+# D-095.3: a bare negation particle is NOT a discourse marker. When ASR puts
+# it in its own speech unit because the speaker paused for emphasis ("No ...
+# quiero sonar a conspiracion"), leaving it orphaned lets a downstream
+# cleanup delete it as micro debris and the following clause is then
+# delivered with its meaning INVERTED. Run 34008386434 (Video00): "No" (0.8 s)
+# split from "quiero sonar a conspiracion ..." at a >= 0.75 s pause, refused
+# by the strict one-word bridge, deleted by the semantic fragment guard,
+# rendered as "quiero sonar a conspiracion". General bilingual vocabulary,
+# no video-specific strings.
+_NEGATION_PARTICLES = frozenset({
+    "no", "nunca", "jamas", "jamás", "tampoco", "ni", "nada",
+    "not", "never", "no", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
+    "cant", "can't", "cannot", "wont", "won't", "isnt", "isn't", "wasnt", "wasn't",
+    "arent", "aren't", "werent", "weren't", "neither", "nor",
+})
+_MAX_NEGATION_REJOIN_GAP_SEC = 2.0
+
+
+def _is_bare_negation_particle(text: str) -> bool:
+    tokens = tuple(t.casefold() for t in _TOKEN_RE.findall(str(text or "")))
+    return 1 <= len(tokens) <= 2 and all(t in _NEGATION_PARTICLES for t in tokens)
+
+
 def _repair_boundary_fragments(
     takes: Iterable[CandidateTake],
     *,
@@ -198,6 +223,7 @@ def _repair_boundary_fragments(
     max_bridge_fragment_sec: float = 2.8,
     max_bridge_gap_sec: float = 0.65,
     max_open_tail_join_sec: float = 20.0,
+    max_negation_rejoin_gap_sec: float = _MAX_NEGATION_REJOIN_GAP_SEC,
 ) -> Tuple[CandidateTake, ...]:
     """Reattach contiguous ASR fragments without deleting real short lines.
 
@@ -239,6 +265,17 @@ def _repair_boundary_fragments(
                 and (previous_word_count >= 2 or strict_contiguous)
             )
 
+            # D-095.3: a leading negation particle rejoins the clause it
+            # negates across a normal (even emphatic) pause, never across a
+            # real section boundary.
+            previous_is_negation_particle = (
+                previous.duration_sec <= max_fragment_sec
+                and _is_bare_negation_particle(previous.text)
+                and _word_count(take.text) >= 3
+            )
+            if same_source and previous_is_negation_particle and -0.02 <= gap <= max_negation_rejoin_gap_sec:
+                repaired[-1] = _join_takes(previous, take)
+                continue
             if same_source and strict_contiguous and (previous_is_open_micro or current_closes_open_previous):
                 repaired[-1] = _join_takes(previous, take)
                 continue
```

---

## APPENDIX B -- D-097 implementation status against this audit (added 2026-09-06; the audit above is unchanged)

The Product Owner approved the post-audit plan (D-097) with the execution
adjustment §1-§5. This appendix records, per audit row, what was implemented;
it does not rewrite the AS-IS map, which remains the description of the engine
that produced RAW 34008386434. Every "expected MP4 effect" is unproven until the
next authorised RAW is run on this code.

| Audit row | Status | Where |
|---|---|---|
| C-1 / root cause #3 (missed family member, "ungrouped = keep") | IMPLEMENTED (A) | `take_grouping.same_opening_restart`, `take_grouping_provider._bridge_aware_components` (deterministic restart edge; semantic >= 0.90 attach to a restart-cohesive component; `accepted_by`, `arbiter_rejected_pairs`) |
| C-2 / root cause #2 (all-failed tie-break survivor) | IMPLEMENTED (B, adjustment §1) | `pipeline._semantic_best_take` -> `no_usable_realization` only with deterministic usability evidence; Ledger `RESOLVED_NONE`; plan/reviewer/validator `dropped_no_usable_realization`; `story_completeness`; harness `NOT_DELIVERABLE_INCOMPLETE_STORY_REVIEW` |
| C-4 / C-5 / root cause #1 (Resolver restores failed restatement; compete-vs-composite by arithmetic) | IMPLEMENTED (adjustment §3) | `realization_resolver`: failed >= 0.85 = unusable (tier 0, never composite member, exclusive CRITICAL groups waived + mirrored into the D-089 index); all-failed cancels out; D-063 dominance untouched for usable restatements |
| G-6 (DeliveryScorer has no cleanliness feature) | IMPLEMENTED (adjustment §2) | `take_judge.delivery_cleanliness_evidence` / `apply_delivery_cleanliness_evidence` (interior dead air, multimodal resets; negative controls) |
| Appendix A (bare negation isolated then deleted) | IMPLEMENTED (D) | `polarity_safety.py`, `take_segmentation` rejoin (<= 2.0 s, word timings published), `semantic_fragment_guard` protection |
| C-6 / C-7 / C-8, G-7, G-8, D-6 (split physical ownership; trimmer before authority; entries unowned) | IMPLEMENTED (C/E) | `boundary_engine_pass.apply_post_freeze_boundary_pass` after `freeze_selection_contract`; draft wrappers skip via `boundary_owner="post_freeze"`; ownership contract in the module; renderer trailing trims recorded |
| C-12 / G-14 (source vs render silence disagree) | MECHANISM REPRODUCED OFFLINE + INSTRUMENTED | synthetic experiment (AAC re-encode + fragmented near-floor pause); `audio_silence` probe/merge + relaxed floor; `dead_air_reconciliation` per finding settles the Video00 case on the next RAW |
| Part 8 / root cause #5 (no perceptual System Watch+Listen) | IMPLEMENTED v1 (advisory) | `perceptual_watch_listen.py`: 4 evaluated capabilities + 4 declared NOT_IMPLEMENTED; never auto-PASS; `DELIVERABLE_PENDING_HUMAN_WATCH_LISTEN`; blocking mode needs PO approval (escalation A) |
| G-12 (identity markers) | PARTIAL | markers `BoundaryEnginePass.post_freeze`, `TakeSegmentation.polarity_rejoin` added; the two probe defects noted in Part 1 are unchanged |
| CLEAN RAW gate (approval) | IMPLEMENTED (QA-only) | `benchmarks/clean_raw_gate.py` + both workflows |
| Part 10 retire/merge candidates | NOT SCHEDULED | unchanged (needs per-module CleanCutBench proof + separate approval) |
| D-062.2 "human review as convenience valve" / D-076 discovery gap (new, RAW 34028202024) | IMPLEMENTED (D-097.1) | a failed retry deleted BEFORE grouping blocked Freeze over a CONTEXTUAL year with no repair strategy; `realization_resolver._pre_group_retry_relation` discovery tier + sanitized-claim reclassification + retry consultation floor; `final_story_coherence_validation._pre_group_restart_credit` asks the bounded SemanticEquivalenceArbiter the question grouping would have asked (validation evidence only) |
| Renderer output timeline / QC window disagreement (new, RAWs 34008386434 + 34029861712: 8-9 false ABRUPT_AUDIO_DISCONTINUITY per attempt, NEEDS_HUMAN_REVIEW, no deliverable MP4) | IMPLEMENTED (D-097.2 R3) | `render.render_preview` = one gapless concat-filter pass with frame-exact per-segment durations (`rendered_segment_duration_sec`); `live_boundary_repair.segment_output_windows` uses the same function; measured join drift <= 3 ms (was +41 ms per part) |
| D-094.F2 label-window starvation (per-edit $0.0075 ledger refused 2 of 6 windows in four RAWs; families decided without labels) | IMPLEMENTED (D-097.2 R2) | `hybrid_editorial = provider_partial:N/M:budget_refused=K`; CLEAN RAW gate INCOMPLETE_EVIDENCE on any refused window; default `max_cost_per_edit_usd` 0.015 (env override kept) |
| C-1 one tier later (abandoned restart + clean retry rejected by the arbiter as "incomplete fragments" at the reconcile tier; both played) | IMPLEMENTED (D-097.2 R1) | `reconcile_semantic_idea_equivalence` merges `same_opening_restart` / short-prefix pairs deterministically before the arbiter (D-083 gate kept, `restart_evidence_merges` traced); pre-reconcile separation not reproduced offline (residual) |
| Part 8 gap: perceptual review only on deliverable candidates (null on the artifact under diagnosis) | IMPLEMENTED (D-097.2 §4) | `_perceptual_review(..., rendered_path)` reviews the diagnostic-invalidated MP4, `artifact_kind` marked, delivery status untouched |
| D-046 class, new instance (RAW 34033468088): `post_selection_continuity_coalescer` re-minted a two-clip family winner as `A__continuity__B` without provenance -> false IDEA_COVERAGE_LOST / UNIQUE_FACT_LOST, Freeze blocked | IMPLEMENTED (D-097.3) | the coalescer restores the micro-gap by extending the leading clip and keeps both identities; `render_plan._coalesce_contiguous_segments` joins them physically; ownership table forbids identity re-minting |
| D-050B ledger consumers hashed `missing_idea_coverage` dict rows (RAW 34032322925 worker crash) | IMPLEMENTED (D-097.2.1) | `semantic_ledger.missing_idea_coverage_idea_ids`; traceback captured in the focused job |
| Technical QC join probe judged speech transients, not the join (RAW 34034507983: 9/21 frame-exact, click-free joins flagged; 3 repairs wasted; no deliverable) | IMPLEMENTED (D-097.4 R4) | `check_audio_discontinuity_at_boundaries` = isolated step within 5 ms of the join vs flank medians AND flank peaks; `offset_ms` in the finding; reproduced with real synthesized speech |
| Physical repair trimmed the plan edge while the Renderer had tightened it (RAW 34034507983: 50 ms "repair" made the segment 0.236 s LONGER, joins +0.233 s) -- ownership collision Renderer.tighten vs BoundaryEngine.repair | IMPLEMENTED (D-097.4 R5) | `repair_segment_for_finding` trims from `tighten_trailing_silence(seg).end` (`tightened_end` recorded); a repair only ever shortens the output |
| Same-idea survivors tied on delivery below the decisive gap decided by score noise (RAW 34034507983 `tg_7765ab`: 0.6846 vs 0.6742 chose the earlier take against both references, 19.7 s) | PROPOSED (D-097.4, escalation A) | prefer the later complete attempt when the gap is below `CLEAR_WINNER_MINIMUM_GAP`; not implemented -- editorial policy |
| Hybrid `failed` label conflates "physically fumbled" with "abandoned" (RAW 34034507983 `clip_98baea`: kept fail-open by the chunk, then deleted by `hybrid_retry_completion_integrity`; Gold keeps it as 3 micro-pieces) | RECORDED (D-097.4) | needs the perceptual/Boundary layer, not a discard rule |
| AttemptReconstructor reads only the ASR clock: two sentences fused across 2.96 s of MEASURED dead air (RAW 34040848026; the papillary-cancer diagnosis then left the edit with Freeze passing) | IMPLEMENTED (D-097.5 R6) | `_attempt_boundary_reason` -> `measured_dead_air_pause` from the D-097 C `audio_silence` events already in the context (>= 1.2 s reaching the transition, confidence >= 0.8) |
| D-089/D-097 §3 waiver: a realization-scoped `failed` label waives a CRITICAL requirement group even when the failure evidence sits in another sentence of the same realization (RAW 34040848026) | PROPOSED (D-097.5, escalation B/A) | waive only when the claim's own span overlaps the realization's measured failure evidence; otherwise CRITICAL_CLAIM_LOST blocks Freeze |
| F3b `accept_complete_pairwise_singleton_bridge` (OFF by PO decision): abandoned stomach take arbiter-confirmed with every family member, component probe declined, kept as a singleton three runs in a row | RECORDED (D-097.5) | PO decision |
