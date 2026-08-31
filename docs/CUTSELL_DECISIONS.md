# CutSell.ai — Canonical Decision Log

Current product generation: **CutSell.ai 7**.

This file records product and Brain decisions that agents and engineers must treat as current unless a later decision explicitly supersedes them.

## Status vocabulary

- **CANONICAL** — current product truth.
- **SUPERSEDED** — was once valid, but a later decision replaced it.
- **REJECTED** — explicitly not allowed.
- **POST-BETA** — valid future scope, but must not block the current beta.

## D-001 — Flow B before Flow A
**Status: CANONICAL**

Flow B ships first. Flow A remains part of CutSell but follows after the Flow B core is stable.

## D-002 — Clean Cut first
**Status: CANONICAL**

CutSell must first understand and clean the raw recording before commercial composition. Clean Cut owns production cleanup: failed takes, false starts, dead air, fumbles, retry debris, boundary issues and related recording mistakes.

## D-003 — Clean Cut cannot delete valid speech for commercial reasons
**Status: CANONICAL**

Commercial labels such as Hook, Benefit, Proof, Story or CTA are descriptive. Clean Cut must not delete otherwise valid speech merely because it is weak commercially or does not fit a sales slot. When uncertain about deletion, preserve valid speech.

## D-004 — Rigid sales funnel
**Status: SUPERSEDED**

A fixed `HOOK -> PROBLEM -> BENEFIT -> PROOF -> CTA` composition is no longer canonical.

Replaced by flexible sales/story understanding that discovers the strongest coherent narrative actually present in the footage.

## D-005 — Flexible funnel and storytelling
**Status: CANONICAL**

Sales Edit must recognize available functions such as hook, problem, discovery, product introduction, feature, benefit, demo, proof, testimonial, objection, result, story and CTA without requiring every function or a fixed order. Story continuity can dominate the edit.

## D-006 — Global-context editing
**Status: CANONICAL**

Master reasoning order:

**Understand globally -> decide locally with global context -> compose globally -> review globally.**

A locally good clip may still be wrong if it breaks chronology, duplicates a better take, kills momentum or damages narrative coherence.

## D-007 — No Frankenstein speech
**Status: REJECTED**

Do not create sentences the creator never delivered by stitching words or sentence fragments across distinct takes or source assets.

## D-008 — Immutable source identity
**Status: CANONICAL**

Every source asset and selected segment must retain deterministic source identity and source timestamps. Never merge identities between source assets.

## D-009 — Retry groups and Best Take
**Status: CANONICAL**

Retries are grouped by the same specific underlying communication attempt, even when wording changes. Best Take ranks valid alternatives using multimodal evidence and context. Best Take is a ranking authority, not a deletion authority.

## D-010 — Visual type is separate from semantic role
**Status: CANONICAL**

Visual presentation and commercial meaning are different dimensions.

Examples of visual type: talking head/headshot, product shot, demo, B-roll/lifestyle, screen recording, close-up, before/after.

Examples of semantic role: hook, problem, story, discovery, feature, benefit, proof, objection, result, CTA.

One segment may carry multiple semantic roles.

## D-011 — Storytelling is relational, not only classification
**Status: CANONICAL**

CutSell must preserve setup, progression, cause/effect, discovery, payoff and required context. Storytelling quality is not achieved merely by tagging a clip as STORY.

## D-012 — Editable Draft, not destructive final-only output
**Status: SUPERSEDED for Clean Cut Core V1 by D-019 (SWAP out of scope). Still CANONICAL for the editor/mobile-app layer's own manual controls (Restore, Remove, Reorder, Trim, Split, captions, text/overlays, audio, undo/redo) and for a future manual "swap take" editor feature, if reintroduced -- see D-019.**

AI output is an editable draft timeline. Selected clips appear in the draft; valid alternatives remain swappable; discarded production mistakes remain recoverable when appropriate. User controls include Swap Take, Restore, Remove, Reorder, Trim, Split, captions, text/overlays, audio and undo/redo.

## D-013 — Human-performance errors matter
**Status: CANONICAL**

A transcript can be linguistically complete and still be a failed take because of visible frustration, lost-line behavior, eye/camera disengagement, body reset, camera adjustment, product-handling mistake, visual fumble or recording-process behavior.

## D-014 — Preserve authentic personality
**Status: CANONICAL**

Do not classify authentic personality, intentional humor, normal mannerisms or meaningful reactions as recording mistakes merely because they are not optimized sales delivery.

## D-015 — Benchmark success is not quality success
**Status: CANONICAL**

A workflow completing successfully is necessary but insufficient. A benchmark only passes the editing-quality gate when the rendered output itself is judged postable under the current doctrine.

## D-016 — Flow A
**Status: POST-BETA**

Flow A remains on the same platform: product/script intake, script generation/editing, recording cards, teleprompter, multiple takes per card, then the same Watch + Listen / Clean Cut / Best Take / editable-draft pipeline.

## D-017 — Current beta priority
**Status: CANONICAL**

Current priority order:
1. Flow B Brain / Clean Cut editorial quality.
2. Flexible sales/story understanding and continuity.
3. Reliable editable mobile draft.
4. Physical iPhone validation.
5. Apple signing and closed TestFlight.

## D-018 — Spend and irreversible actions
**Status: CANONICAL**

Paid RunPod work, new paid infrastructure, production deployment, release merge, destructive legacy changes and Apple/TestFlight release actions require explicit user approval at their respective gates.

## D-019 — SWAP is out of scope for Clean Cut Core V1
**Status: CANONICAL**

Effective the Clean Cut Core V1 migration: SWAP is removed from the ACTIVE Clean Cut
decision path. The semantic membership model for Clean Cut Core V1 is SELECT/KEEP vs
DISCARD only -- there is no alternate-take inventory in the winning timeline. A
non-winning retry does not belong in the final edit and is excluded from it. The goal
is one coherent winning edit, not a winning edit plus alternate inventory.

This is a product-scope decision, not a technical limitation: alternate-take swapping
may return in a future version, another product layer, or an editor feature -- that
decision is explicitly deferred to the user. **SWAP IS OUT OF SCOPE FOR CLEAN CUT V1
UNTIL THE USER EXPLICITLY REINTRODUCES IT.**

Legacy SWAP machinery is not deleted, only deactivated for this path: `deterministic_
best_take_authority.py`'s `swap_enabled` parameter defaults to `False` (a legitimate
losing retry is DISCARDed, not parked as SWAP); the whole-video Unified Selection
reasoner (which natively reasons in SELECT/SWAP/DISCARD) is deactivated in the active
path via `clean_cut_core_v1_enabled` (default on) and kept only for rollback; render_
plan.py already renders `selected` only and was unaffected. `draft_edits.py`'s
`swap_take` (a manual, user-initiated mobile-editor operation, not an automatic
Selection decision) is a different product layer and was not touched by this decision.

## D-020 — Clean Cut Core V1: idea-first architecture
**Status: CANONICAL**

Clean Cut Core V1 reasons idea-first, not pairwise-first: complete intended ideas ->
all delivery attempts belonging to each idea -> quality/completeness competition ->
one winning delivery or a necessary composite -> KEEP/DISCARD. An offline audit of a
real run found that discovering retry families as an unranked, fixed-budget pairwise
comparison can systematically fail to even present real retries to the semantic
arbiter (pairs late in a video were less likely to ever be proposed, independent of
how obvious a retry they were); pairwise discovery with a fixed top-K is therefore not
the fundamental retry-family-discovery mechanism -- `take_grouping_provider.py`'s
candidate-pair generation now ranks eligible pairs by priority (temporal proximity,
raw lexical overlap, continuation/restart evidence) before spending the batch budget,
rather than truncating in enumeration order.

Active pipeline: ASR/aligned speech timeline -> Watch+Listen evidence -> attempt
reconstruction -> idea/intent clustering (lexical tier + bounded `semantic_idea_
equivalence` arbiter tier) -> delivery completeness + performance-quality competition
(`take_judge.rank_takes`) -> `deterministic_best_take_authority` (KEEP/DISCARD, SWAP
out of scope per D-019) -> composite resolution when genuinely required -> `final_
story_coherence_validation` (residual-ambiguity resolution via the same bounded
arbiter, missing-story-ending observability) -> KEEP/DISCARD final membership ->
Selection Freeze -> Boundary/Microtrim. Gemini is never the primary editor; it is a
bounded semantic arbiter only, reused at two points (idea clustering, final coherence
ambiguity) rather than the single whole-video SELECT/SWAP/DISCARD reasoner call, which
is deactivated in the active path (rollback: `CUTSELL_CLEAN_CUT_CORE_V1=0`).

`final_story_coherence_validation.py` additionally enforces two hard pre-Freeze
invariants, both deterministic and evidence-based (no new heuristic invented): the
**contradiction invariant** (two still-co-selected members of the same retry family
that disagree on a number or an explicit negation are factually incompatible, not
alternate phrasings -- reuses `final_sibling_grouping.py`'s own `_numbers`/
`_negations` extractors) and the **idea coverage invariant** (a retry-family group
with zero surviving selected members means that intended idea vanished from the
winning edit entirely). Either sets `freeze_blocked=True` in diagnostics;
`universal_clean_cut.py` checks this and skips Selection Freeze/Boundary entirely
rather than let a self-contradictory or idea-incomplete draft reach Freeze --
Boundary never gets the chance to paper over a semantic membership mistake.

Not yet implemented in V1, documented as a known gap rather than silently absent:
exhaustive unique-fact-loss detection beyond the number/negation contradiction check,
general (non-numeric/non-negation) factual contradiction, and full integration of
`hybrid_composite_best_take.py`'s dedicated composite-reconciliation machinery into
the idea-first chain (composite resolution today relies on genuinely different ideas
never being forced into competition in the first place, per the "continuation must
not collapse" invariant) -- see D-021's CompositeResolver row.

## D-021 — Canonical Clean Cut Core V1 component map
**Status: CANONICAL**

The canonical engine converges toward 11 responsibilities. Every active behavior in
the Clean Cut Core V1 path (`clean_cut_core_v1_enabled=True`, the default) maps to
exactly one, so there is one owner per responsibility rather than several semantic
brains rewriting the same membership sequentially:

| Canonical component | Current code | Classification |
|---|---|---|
| AttemptReconstructor | `attempt_reconstruction.py` (`reconstruct_delivery_attempts`) | KEEP AS CORE |
| IdeaClusterer | `take_grouping_provider.py` (`safe_group_takes` lexical tier + `reconcile_semantic_idea_equivalence` bounded semantic tier, priority-ranked candidate pairs per D-020); `take_grouping.py` (`retry_similarity`/`semantic_key`) | KEEP AS CORE / MERGED |
| RetryFamilyResolver | The resolved groups IdeaClusterer produces ARE the retry families. `final_sibling_grouping.py`, `global_session_sibling_bridge.py`, `session_boundaries.py` contribute merge evidence/session partitioning into that same result | DEMOTE TO EVIDENCE (feed IdeaClusterer's output, not a separate invoked step) |
| DeliveryScorer | `take_judge.py` (`score_take`/`rank_takes`) | KEEP AS CORE |
| BestTakeResolver | `deterministic_best_take_authority.py` | KEEP AS CORE / PROMOTED (Phase 1) |
| SemanticArbiter | `semantic_idea_equivalence.py` + `semantic_idea_equivalence_google.py` | KEEP AS CORE, bounded role only (D-020) |
| CompositeResolver | `hybrid_composite_best_take.py`, `post_selection_complementary_family_stabilizer.py`, plus **at least 12 more** `install_hybrid_*`/`install_post_selection_*` hooks in `cutsell_worker/__init__.py` that monkeypatch the same shared functions (`apply_hybrid_session_cleanup`, `safe_group_takes_by_sessions`) -- see D-022 for the full inventory | DEMOTE TO EVIDENCE / NOT YET CONSOLIDATED -- reachable in the active V1 path (see the `editorial_judge` starvation fix below). `hybrid_composite_best_take.py`'s own two chained wraps are individually sound (identifies composite-worthy pieces, marks them via `_COMPOSITE_SPLIT_IDS` so grouping cannot re-collapse them), but it is one of roughly 14 legacy authorities installed at import time that can each restore, delete, or suppress members of the same retry family, layered in installation order with no single arbiter among them. D-022 (RAW 33345946000's content-loss finding) treats full consolidation into one directly-callable, non-conflicting component as a real, still-open, higher-risk item -- not something to claim done. Its decisions surface in `diagnostics.hybrid_editorial_chunks` (now printed in CI, see D-020's observability note) |
| StoryValidator | `final_story_coherence_validation.py` | KEEP AS CORE (new; folds alternates to discard, resolves residual ambiguity via SemanticArbiter, contradiction invariant, idea-coverage invariant, general lost-semantic-atoms coverage ledger against the ACTUAL final KEEP timeline (D-022), hard pre-Freeze gate via `freeze_blocked`). Per D-022, this is also the structural backstop for the CompositeResolver consolidation gap above: because it checks final `selected`/`discarded` content directly, it catches unique-fact/idea loss regardless of which of the ~14 upstream hybrid_* authorities caused it. |
| SelectionFreeze | `selection_boundary_contract.py` (`freeze_selection_contract`/`enforce_selection_contract`) | KEEP AS CORE, unchanged |
| BoundaryEngine | `final_boundary_authority.py`, `human_boundary_polish_v5.py`, speech-safe/microtrim guards | KEEP AS CORE, physical-only, unchanged. Also satisfies the canonical directive's audio/visual pacing QA requirement: `human_boundary_polish_v5.py` already consolidates the dead-air/reset pacing work older v1-v4 passes mixed with Selection responsibilities, using multimodal reset evidence (`_reset_score`/`_micro_reset_evidence`, reused from v2) rather than naive silence-length deletion, and is Boundary-only by construction. No new pacing module was built; pacing *quality* (as opposed to architecture) still needs empirical validation via Human Watch+Listen. |
| Renderer | `render_plan.py`/`render.py` | KEEP AS CORE, unchanged (renders `selected` only, never `alternates`) |

Reused as BestTakeResolver's own upstream input in the active path (not a parallel
authority): `selection_phase_authority.py`, `selection_conflicted_bridge_guard.py`
(Hybrid-vote-informed reconciliation, unchanged, now feeding into
`deterministic_best_take_authority` rather than being a competing final say).

DEPRECATED FROM ACTIVE PATH (kept only behind `clean_cut_core_v1_enabled=False` for
rollback, per D-019/D-020): `unified_selection_reasoner.py`, `unified_selection_google.py`
(the whole-video SELECT/SWAP/DISCARD reasoner).

KEPT AS SAFETY/ROLLBACK INFRASTRUCTURE (dormant, not deleted, per D-019):
`deterministic_best_take_authority.py`'s `swap_enabled` parameter;
`draft_edits.py`'s `swap_take` (a different, manual editor-layer product surface,
never part of this decision).

## D-022 — RAW 33345946000 content-loss finding: general coverage ledger
**Status: CANONICAL**

RAW 33345946000 (head `0ea0adf`, Clean Cut Core V1's first controlled run)
resolved the flagship hereditary-cancer contradiction cleanly and produced an
8x semantic-equivalence merge improvement, but Human Watch+Listen against the
actual result JSON found genuine content loss `final_story_coherence_
validation` reported as `freeze_blocked: false`: a papillary-thyroid-cancer
diagnosis confirmation, a sonography/ultrasound causal transition, and an
entire pimples/rash symptom beat were all missing from the final KEEP
timeline, with no other delivery covering that content.

**Root cause (code-grounded, not inferred):** `apply_hybrid_session_cleanup`
(`pipeline.py` Pass 2) is a per-clip failed/BTS classifier that runs BEFORE
IdeaClusterer (`safe_group_takes_by_sessions` / `reconcile_semantic_idea_
equivalence`) ever sees a candidate, and has no idea-coverage awareness of
its own -- it judges one take in isolation, with no concept of whether it is
the sole carrier of an audience-facing fact. A clip it deletes never enters
any `take_judge_groups` entry. `final_story_coherence_validation`'s idea-
coverage check (`_missing_idea_coverage`) was scoped only to that
`take_judge_groups` diagnostic, so it reported nothing missing -- not
because coverage was actually fine, but because the check could not see
past grouping to the clips deleted before grouping began. Separately,
`hybrid_composite_best_take.py`'s rescue path
(`_restore_performance_only_unique_deliveries`) only reconsiders clips
deleted via `delete_basis == "semantic_failed_plus_local_performance"`;
`hard_semantic_delete` (`delete_basis == "high_confidence_semantic"`,
confidence >= 0.94 -- the exact basis on RAW 33345946000's confirmed
deletion) is structurally excluded from ever being rescued. This is root
causes **C** (Hybrid editorial deletion has no coverage/uniqueness
awareness) and **F** (StoryValidator's coverage check was scoped to
post-grouping state, invisible to pre-grouping deletion) combined, not a
clustering-quality problem (A): IdeaClusterer never got the chance to see
the deleted takes at all.

**General fix implemented:** `final_story_coherence_validation.py` gained
`_lost_semantic_atoms`, a coverage ledger that compares every discarded
clip's own content and critical (number/negation) atoms directly against
the union of the final `selected` text -- independent of which stage
discarded it, whether it was ever grouped, or which of the ~14 legacy
hybrid_* authorities (see D-021's CompositeResolver row) touched it. Any
missing number/negation atom blocks freeze unconditionally; a broader loss
of ordinary content vocabulary blocks freeze only past a volume+coverage
floor, so a genuinely redundant, correctly-discarded retry (which shares
most of its topic vocabulary with the surviving winner) is not mistaken for
real information loss. This closes the "exhaustive unique-fact-loss
detection" gap D-020 had left open, and makes StoryValidator's coverage
invariant true to what CLAUDE.md already declared ("an intended idea must
not silently vanish from the winning edit") regardless of which upstream
authority is responsible.

**What is not yet done:** full consolidation of the ~14 hybrid_*/post_
selection_* legacy authorities that can each restore/delete/suppress
members of the same retry family into one directly-callable, non-
conflicting CompositeResolver component remains open and materially
riskier than this cycle's fix -- see D-021's CompositeResolver row. The
coverage ledger above is the structural backstop for that gap (it does not
care which upstream authority caused a loss), not a replacement for
consolidating the authorities themselves. General (non-numeric/negation)
factual contradiction detection also remains open (unchanged from D-020).

**Test coverage:** `tests/test_cutsell_final_story_coherence_validation.py`
(general fixtures: pre-grouping-deleted unique content, correctly-discarded
redundant retry does not false-positive, missing numeric fact despite high
overlap, short filler is not flagged) and two new CleanCutBench categories
in `tests/test_cutsell_clean_cut_core_evaluation_suite.py` (retry winner
missing the loser's unique numeric fact; retry winner missing the loser's
unrelated symptom beat) reachable through the real grouping/Best-Take/
coherence chain.

## Change rule

When a new decision changes product behavior, update this file in the same development cycle. Do not silently redefine CutSell through code alone.