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
| CompositeResolver | `composite_resolver.py` (`apply_composite_resolution`, `apply_composite_group_split`, `apply_composite_family_stabilization`) -- builds its chain by calling the real `install_*()` of `semantic_fragment_guard.py`, `hybrid_retry_completion_integrity.py`, `hybrid_story_guard.py`, `hybrid_alternate_integrity.py`, `hybrid_cross_group_retry_integrity.py`, `incomplete_bridge_retry_authority.py`, `hybrid_failed_continuation_integrity.py`, `hybrid_retry_winner_authority.py`, `hybrid_gold_reconciliation.py`, `failed_prefix_completion_rescue.py`, `final_delivery_integrity.py`, `terminal_delivery_reconciliation.py`, `hybrid_failed_soft_restore.py`, `hybrid_unavailable_retry_fallback.py`, `hybrid_complementary_delivery_guard.py`, `hybrid_semantic_complementary_rescue.py`, `hybrid_semantic_composite_bridge.py`, `hybrid_composite_best_take.py`, `hybrid_semantic_conflict_arbitration.py` (19 take-level hooks), plus `post_selection_complementary_family_stabilizer.py` downstream | KEEP AS CORE / CONSOLIDATED (D-023) -- one directly-callable component, one documented historical order, called explicitly from `pipeline.py` (no monkeypatching left in this domain outside the one lazy chain-build). `cutsell_worker/__init__.py` no longer installs any of these 20 modules' own `install_*()` hooks; each keeps its existing pure logic and its own tests unchanged, consumed by calling their real installers once rather than reimplementing them (D-023 -- an earlier hand-transcribed version of this row's own consolidation missed 5 of the 19 and is why the final design calls the real installers instead of re-typing their logic). Its decisions surface in `diagnostics.hybrid_editorial_chunks` (printed in CI, see D-020's observability note) |
| StoryValidator | `final_story_coherence_validation.py` | KEEP AS CORE (new; contradiction invariant, idea-coverage invariant, general lost-semantic-atoms coverage ledger against the ACTUAL final KEEP timeline (D-022), hard pre-Freeze gate via `freeze_blocked`). **D-090 authority boundary:** in LEGACY/SHADOW mode and in AUTHORITATIVE mode's first (pre-resolver, legacy-evidence) pass it still folds alternates to discard and resolves residual ambiguity via SemanticArbiter; in AUTHORITATIVE mode's post-resolver pass it is VALIDATION-ONLY (`apply_post_authority_story_validation`, `post_authority_validation.py`): it never edits membership, winners, composites, speech or order -- family bookkeeping is answered by the resolver's D-087 verdict through the shared `assess_authoritative_membership`, and a signature invariant fails closed on any drift. The pre-D-090 claim that StoryValidator is the last/final selection authority is RETIRED for AUTHORITATIVE mode; the one semantic Selection authority there is the Unified Realization Resolver, applied once. Per D-022, this is also the structural backstop for the CompositeResolver consolidation gap above: because it checks final `selected`/`discarded` content directly, it catches unique-fact/idea loss regardless of which of the ~14 upstream hybrid_* authorities caused it. |
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

## D-023 — CompositeResolver consolidation: classification + canonical composition
**Status: CANONICAL**

Follow-up to D-022's honest gap ("full consolidation... remains open"). The user's
explicit directive: classify every relevant hybrid_*/post_selection_* hook before
touching anything, then consolidate into one directly-callable CompositeResolver,
with no legacy hook remaining an independent downstream semantic-membership authority.

### Classification (7 questions each, per the directive)

The initial audit, keyed on `install_hybrid_*`/`install_post_selection_*` naming,
found 15 hooks (14 take-level + 1 downstream). **That audit was incomplete.** A
differential test written against this module's own "the base call is pure" claim
failed during verification, which is how a further, broader grep (every file in
`cutsell_worker/` that textually references `apply_hybrid_session_cleanup`, not
just files matching a naming convention) found **five more actively-installed
hooks** doing the identical thing under different names: `semantic_fragment_guard`,
`incomplete_bridge_retry_authority`, `failed_prefix_completion_rescue`,
`final_delivery_integrity`, `terminal_delivery_reconciliation`. All five are real,
narrow, individually-tested delivery-restoration/deletion authorities in
CompositeResolver's exact domain -- not the broader out-of-scope Boundary sprawl --
and critically, they were **interleaved** with the original 15 in
`cutsell_worker/__init__.py`'s historical order, not merely appended before or
after. (Three further files referencing the same identifier --
`cross_group_truncated_winner_authority.py`, `hybrid_performance_retry_restore_
guard.py`, `incomplete_unique_bridge_completion_rescue.py` -- were confirmed dead:
their own `install_*()` is never called anywhere in the codebase.)

The corrected full set is 19 take-level hooks plus the 1 downstream extension (20
total), all sharing the same execution point (monkeypatched onto
`hybrid_session_cleanup.apply_hybrid_session_cleanup`, running inside `pipeline.py`
Pass 2, before IdeaClusterer/grouping) and the same mutable state
(`HybridSessionCleanupResult.kept/deleted/diagnostics`):

| Hook | Responsibility | Can KEEP/restore/delete/suppress? | Overlaps with | Still required under V1? | Canonical placement |
|---|---|---|---|---|---|
| `semantic_fragment_guard` | Deletes structurally-obvious failed speech debris (tiny/open fragments, filler BTS, severe repetition) already labeled failed/BTS at medium-high confidence | delete | All later delete-only hooks | Yes -- adds textual-structure corroboration no other hook checks | CompositeResolver step 2 |
| `hybrid_retry_completion_integrity` | Removes cross-group retries proven covered by a peer; rolls back a completed clause's parallel failed tail | delete, trim-restore | `hybrid_cross_group_retry_integrity` (narrower, later, different evidence) | Yes -- distinct evidence (reset-backed full-alternate retry) other hooks don't cover | CompositeResolver step 3 |
| `hybrid_story_guard` | Restores a unique story paragraph deleted on non-authoritative (semantic-only) evidence | restore | All later delete-only hooks (acts as their common floor) | Yes -- the one hook enforcing "semantic confidence alone is not physical proof" | CompositeResolver step 4 |
| `hybrid_alternate_integrity` | Suppresses a stranded short alternate beside a clear winner | delete | `hybrid_cross_group_retry_integrity` (broader) | Yes -- narrow, cheap, catches short-debris case others miss | CompositeResolver step 5 |
| `hybrid_cross_group_retry_integrity` | Collapses a semantically-proven retry stranded across deterministic groups | delete | `hybrid_alternate_integrity`, `hybrid_retry_winner_authority` | Yes -- feeds `hybrid_failed_soft_restore` by diagnostics name (order-dependent) | CompositeResolver step 6 |
| `incomplete_bridge_retry_authority` | Protects a completed clause's bridge continuation from being amputated by a later guard | protect/restore | `hybrid_failed_continuation_integrity` (adjacent structural case) | Yes -- a distinct completed-clause protection | CompositeResolver step 7 |
| `hybrid_failed_continuation_integrity` | Repairs a failed split retry (both directions: failed-prefix+continuation, and selected-prefix+failed-suffix) | delete (two-part) | none direct | Yes -- the "Video 00 Round 5 Gold" case, still a real failure shape | CompositeResolver step 8 |
| `hybrid_retry_winner_authority` | Drops a proven failed attempt superseded by a later high-confidence clean winner | delete | `hybrid_gold_reconciliation` (similar spirit, different threshold/evidence) | Yes -- lower threshold than the generic delete gate, closes a real gap | CompositeResolver step 9 |
| `hybrid_gold_reconciliation` | Two narrow Human-Gold-exposed repairs (restore clean retake + remove failed prior; remove orphan continuation of a deleted incomplete alternate) | restore + delete | `hybrid_retry_winner_authority` | Yes -- distinct structural cases | CompositeResolver step 10 |
| `failed_prefix_completion_rescue` | Rescues a clean completion prefix from an immediately-following high-confidence failed tail | restore (prefix only) | `final_delivery_integrity`, `terminal_delivery_reconciliation` (related prefix/suffix repairs) | Yes -- the "Video 03 Gold" case | CompositeResolver step 11 |
| `final_delivery_integrity` | Three global repairs: retry_setup-separated duplicate deliveries; open retry prefix before already-failed suffixes; incomplete delivery + tiny discarded completion fragment | restore + delete | `terminal_delivery_reconciliation` (narrower version of the same two structures) | Yes -- broader-window version of the terminal reconciliation cases | CompositeResolver step 12 |
| `terminal_delivery_reconciliation` | Two terminal-boundary repairs: open retry prefix yields to an earlier complete delivery when its continuation is already failed; incomplete delivery reclaims one tiny immediate fragment | restore + delete | `final_delivery_integrity` (broader) | Yes -- narrower, source-word-only version still catches cases the broader pass's stricter gates miss | CompositeResolver step 13 |
| `hybrid_failed_soft_restore` | Undoes a weak (<0.90 confidence) cross-group "failed" delete lacking destructive authority | restore | Reads `hybrid_cross_group_retry_integrity`'s diagnostics directly | Yes -- the correction for cross-group integrity's own occasional over-reach | CompositeResolver step 14 |
| `hybrid_unavailable_retry_fallback` | Deletes an undecided incomplete retry only when Hybrid windows were unavailable and a later complete delivery strongly covers it | delete | `hybrid_complementary_delivery_guard`'s second half (same trigger condition, different evidence direction) | Yes -- the fail-open case when Hybrid itself couldn't run | CompositeResolver step 15 |
| `hybrid_complementary_delivery_guard` | Restores a complementary tail cut by cross-group collapse; deletes an unavailable-window prior restart (earlier-complete-delivery direction) | restore + delete | `hybrid_unavailable_retry_fallback` (later-delivery direction) | Yes -- the earlier-delivery-direction complement | CompositeResolver step 16 |
| `hybrid_semantic_complementary_rescue` | Restores a complete alternate that retry-completion removed, when it carries material unique content vs. its named winner; marks it for composite split | restore + composite-mark | `hybrid_semantic_composite_bridge` (consumes its output) | Yes -- first stage of the composite pipeline | CompositeResolver step 17 |
| `hybrid_semantic_composite_bridge` | Revokes a rescue that is actually a same-opening retry, not complementary; normalizes valid rescues into Composite Best Take's shape | revoke + normalize | `hybrid_semantic_complementary_rescue` (upstream), `hybrid_composite_best_take` (downstream) | Yes -- the correction/bridge between the two | CompositeResolver step 18 |
| `hybrid_composite_best_take` | Restores performance-only-condemned complete deliveries with unique content; deletes strong-prefix unavailable restarts; builds two-piece composites; marks singleton split | restore + delete + composite-mark | `hybrid_semantic_composite_bridge` (upstream evidence) | Yes -- the actual composite-construction authority | CompositeResolver step 19 |
| `hybrid_semantic_conflict_arbitration` | Restores a complete delivery whose strongest winner/keep evidence is >= its conflicting failed/bts evidence (overlapping-window label conflicts) | restore | none direct (reads only `semantic_decisions` + diagnostics) | Yes -- the final label-conflict correction, must run last | CompositeResolver step 20 |
| `post_selection_complementary_family_stabilizer` | Replaces a redundant selected monolith with a concise discarded delivery + later winner when they jointly preserve its critical facts and cover most of its content | restore + suppress | none direct; operates on the built `DraftTimeline`, not raw takes -- genuinely downstream | Yes -- catches a case only visible after grouping/ranking has already built a draft | CompositeResolver step 21 (the one true downstream extension) |

None of the 19 take-level hooks was found REDUNDANT/SUPERSEDED or SAFETY/ROLLBACK-ONLY
in isolation -- each has a distinct trigger condition or evidence source pinned by its
own existing tests. The violation was never "these do nothing" -- it was "no single
place says what CompositeResolver does, in what order, or why," and two of the
nineteen (`hybrid_composite_best_take`, `hybrid_semantic_complementary_rescue`) each
independently monkeypatched `session_boundaries.safe_group_takes_by_sessions` too,
via two SEPARATE `ContextVar`s, one of which (`hybrid_semantic_composite_bridge`)
reached directly into the other module's private `_SPLIT_IDS` variable by name.

### Canonical composition implemented

An earlier version of this consolidation hand-transcribed each of the (then believed
complete) 15 hooks' glue code directly into composed step functions -- this is
exactly what missed the five extra hooks above, and would independently have gotten
their relative order wrong even once found, since they are interleaved with the
original 15, not appended around them. Hand-transcribing 19 interacting closures by
reading and re-typing each one carries real, demonstrated transcription risk, so the
final design does not do that.

`composite_resolver.py` instead builds the chain by calling each hook's own real,
already-tested `install_*()` function -- completely unmodified -- exactly once, in
the exact historical `cutsell_worker/__init__.py` order, against the shared module
attributes, then restores those attributes to what they already were. This reuses
every hook's real closure directly (zero risk of a transcription error changing any
threshold, condition, or diagnostics key) while turning 20 scattered import-time
side effects into one composed, private, directly-callable reference this module
owns (`apply_composite_resolution`) that `pipeline.py` calls explicitly.
`apply_composite_group_split` replaces the two `ContextVar`-based group-splitting
monkeypatches with one explicit call `pipeline.py` makes right after grouping
(reading both hooks' ContextVars once, right after the chain runs, rather than
letting either hook's own grouping monkeypatch actually apply). `apply_composite_
family_stabilization` (step 21) is called explicitly at the end of
`build_flow_b_draft` instead of via a monkeypatch on that function.

`cutsell_worker/__init__.py` no longer calls any of the 19 take-level hooks' own
`install_*()` functions (nor `post_selection_complementary_family_stabilizer`'s).
Each hook's own file, its own pure/glue functions, and its own monkeypatch-based
tests are completely unchanged -- `composite_resolver.py` consumes them by calling
their real installers, not by reimplementing anything. Two hooks whose logic
previously lived only inside an install-time closure (`hybrid_failed_soft_restore`,
`hybrid_unavailable_retry_fallback`) had that logic extracted to a named function
first, non-breaking, purely so a test could call it directly; their `install_*()`
still delegates to it unchanged and works exactly as before.

`hybrid_session_cleanup.apply_hybrid_session_cleanup` and `session_boundaries.
safe_group_takes_by_sessions` are therefore guaranteed to remain their pure,
unwrapped selves for the life of the process, except for the brief window inside
`composite_resolver._build_take_level_chain()` itself (which runs once, lazily, on
first use, and restores both attributes before returning). This closes the literal
"multiple independent semantic authorities sequentially mutate final membership"
violation for CompositeResolver's domain: there is now one callable, one order, one
decision path, and it is provably immune to a stray future call to any of the 19
hooks' own `install_*()` (pinned by
`test_stray_legacy_install_call_after_chain_is_built_cannot_affect_it`).

**Verification (behavior-preservation, not just code inspection):** the full test
suite was run repeatedly through this consolidation and its correction, ending at
1313 passed, 1 skipped (up from the pre-consolidation 1294, due to new tests added
along the way; no pre-existing test's outcome changed). Fixes needed along the way:
one fixture-text recalibration (D-022's "loser" fixture, unrelated to this specific
consolidation), a rewrite of the two wiring-mechanism tests that asserted the
now-removed monkeypatch chain itself, and -- after the five-hook discovery -- a
fixture-isolation fix to `tests/test_cutsell_semantic_fragment_guard.py` (which had
been relying on `cutsell_worker/__init__.py`'s auto-install rather than wrapping the
hook itself the way every other hook's test file already does) plus a test-isolation
bug in this session's own new differential test (an install-time monkeypatch of
`session_boundaries.safe_group_takes_by_sessions` that a `try/finally` did not
restore, leaking into a later test). A differential-style test
(`test_cross_group_retry_integrity_feeds_failed_soft_restore_by_diagnostics_name`)
pins the one cross-module diagnostics coupling most likely to break silently under
any future change to this composition.

**Known remaining gap, stated honestly:** the broader legacy sprawl of other
`install_*` hooks in `cutsell_worker/__init__.py` that do NOT touch
`apply_hybrid_session_cleanup` or `safe_group_takes_by_sessions`
(post_selection_edge_only_boundary, round8/9/11_*, selection_integrity,
temporal_word_boundary_integrity, etc.) is explicitly OUT of this cycle's scope --
a separate, larger question about the Boundary/post-selection-integrity layer, not
CompositeResolver's domain. Confirmed (by the same broad grep that found the five
missing hooks above) that none of those remaining hooks reference
`apply_hybrid_session_cleanup`, so this scope boundary is now evidence-based rather
than assumed.

## D-025 — RAW 33366538992 follow-up: composite persistence, freeze/review consistency, plan versioning
**Status: CANONICAL**

Independent inspection of RAW 33366538992's actual result JSON found two architectural
inconsistencies. Both were root-caused to specific code, fixed generally (no
Video00-specific logic), and pinned with tests. Building the fixes surfaced two more
real bugs in the same week-old D-024 code, found by writing rigorous tests rather than
by further human inspection -- recorded honestly below, not smoothed over.

### Issue 1 — accepted composite did not survive to final KEEP membership

**Root cause:** `reconcile_semantic_idea_equivalence` (`take_grouping_provider.py`,
Pass 3) runs its OWN, separate Gemini arbiter call on the lexical grouping's output,
including composite-marked clips that `apply_composite_group_split` (Pass 3, run
immediately before) had already forced into protected singleton groups. That
protection stops the LEXICAL grouper from re-merging them, but
`reconcile_semantic_idea_equivalence` has no knowledge of it at all: its own arbiter
call confirmed the two accepted composite pieces were "the same idea" (true -- that is
exactly why they were accepted as a composite) and re-merged them into one ordinary
retry contest. A third, unrelated clip then won that contest outright, and neither
composite piece survived to KEEP -- exactly what `lost_semantic_atoms` correctly
flagged (`freeze_blocked=true`), but the underlying composite decision was still
silently discarded.

**Fix:** `reconcile_semantic_idea_equivalence` gained a `protected_ids` parameter that
filters composite-marked clips out of candidate-pair generation entirely -- not
"unlikely to merge", structurally unable to be proposed as a candidate pair.
`pipeline.py` passes the same `composite_split_ids` `apply_composite_group_split`
already receives. Tests: `tests/test_cutsell_semantic_idea_equivalence_grouping.py`
pins that a protected pair is never re-merged with each other AND never merged into an
unrelated group, and that the arbiter is never even called for a fully-protected
candidate set.

### Issue 2 — Freeze state contradicted Review state

**Root cause:** `install_selection_freeze()`/`install_boundary_selection_invariant()`
(`selection_boundary_contract.py`) monkeypatch `pipeline.build_flow_b_draft` to
unconditionally freeze-then-verify at the end of Pass 3+legacy-Selection-phase hooks
-- a holdover from the pre-V1 architecture where `build_flow_b_draft`'s own output was
the final answer. Clean Cut Core V1 added `deterministic_best_take_authority`,
`final_story_coherence_validation`, `build_canonical_edit_plan`, and
`final_edit_reviewer.review` as MORE semantic authorities that run AFTER
`process_local_sources` (and therefore after that entire legacy chain) returns to
`universal_clean_cut.py`. When this module's own gate later determines
`freeze_blocked=true` and correctly skips its OWN freeze call, the diagnostics key
`selection_boundary_contract` is left at whatever the premature, pre-StoryValidator
legacy freeze already wrote ("frozen" or "verified") -- a direct, misleading
contradiction with `freeze_blocked=true` in the same result JSON.

**Fix:** `universal_clean_cut.py`'s freeze-blocked branch now explicitly overwrites
`diagnostics["selection_boundary_contract"]` with an honest
`"not_frozen_freeze_blocked_by_coherence_review"` status (recording the superseded
premature status for observability, not silently dropping it), rather than leaving the
stale legacy value in place. The much larger legacy Selection-phase/Boundary-phase
monkeypatch chain this exposed (~13 more `install_*` hooks wrapping
`build_flow_b_draft` for post_selection/round8-9-11/final_selection_retry_arbiter
work) is NOT touched -- restructuring it to run StoryValidator/FinalEditReviewer
BEFORE that whole chain, rather than working around its premature freeze after the
fact, is a real, larger, separate re-architecture and out of this cycle's scope; this
fix makes the CURRENT two-phase-freeze reality self-consistent rather than attempting
to eliminate it. Test: `tests/test_cutsell_universal_clean_cut.py`'s
`test_freeze_blocked_never_leaves_a_stale_frozen_or_verified_contract` simulates the
exact premature-freeze state and pins that it can never coexist with
`freeze_blocked=true` afterward.

### Two more real bugs found while building the above (D-025, same cycle)

1. **`CanonicalEditPlan.keep_sequence` was re-sorted by timestamp instead of
   preserving `draft.selected`'s actual order.** `render_plan.py` renders
   `for clip in draft.selected` directly -- that tuple's order IS the true final
   composed order (Composer is explicitly allowed to reorder independent ideas for
   pacing/sales logic). Re-sorting by `(source_order, start, end, clip_id)` when
   building the "single semantic handoff to physical editing" made every
   order-sensitive check silently unable to ever detect a real reordering. Fixed to
   iterate `draft.selected` verbatim.
2. **Every accepted composite was misclassified as `unresolved_ambiguous`
   (DUPLICATE_IDEA + UNRESOLVED_RETRY findings).** `build_canonical_edit_plan`'s idea
   construction treated ANY idea with 2+ surviving members as ambiguous, with no
   exception for the case where those 2+ members are exactly an accepted composite's
   own components -- which is the CORRECT, intended outcome, not an unresolved retry
   contest. Fixed: `coverage_status` is `"complete"` (not `"unresolved_ambiguous"`)
   when every surviving member of a 2+-winner idea is a composite-protected clip.
   Without this fix, FinalEditReviewer would have FAILed on every correctly-resolved
   composite, permanently.

Both were caught by writing the STORY_ORDER_BREAK detector's own tests, not by manual
inspection -- direct evidence for why the tests were worth writing carefully rather
than assuming the D-024 code from the prior cycle was already correct.

### Plan identity: `plan_id` / `plan_version` / `semantic_hash`

`CanonicalEditPlan` gained `plan_id` (derived from `project_id` + `semantic_hash`, so
a materially different KEEP timeline gets a different id), `plan_version` (always `1`
this cycle -- no automatic repair loop exists yet to produce v2/v3, see below), and
`semantic_hash` (the same token-stream digest `selection_boundary_contract.py` already
computes, reused via its `semantic_token_stream` helper rather than a second,
possibly-divergent implementation). `freeze_selection_contract` gained an optional
`plan` parameter that records `plan_id`/`plan_version`/`plan_semantic_hash` and a
`matches_reviewed_plan` boolean onto its own diagnostics -- observability, not a hard
equality gate, because `enforce_complete_idea_boundaries` legitimately runs between
FinalEditReviewer's PASS and the freeze call and can restore source-proven leading/
trailing words, which changes the token stream without changing meaning.

### STORY_ORDER_BREAK: a real, narrow detector (not the general causal-order case)

`final_edit_reviewer.py` now actually detects `STORY_ORDER_BREAK`, scoped narrowly to
one accepted composite's own components: they must appear in the final KEEP sequence
in the same relative order they were recorded in ("natural continuation" is exactly
what `hybrid_composite_best_take.py` already requires to accept a composite in the
first place). This deliberately does NOT attempt general cross-idea narrative-order
checking (added to the vocabulary as `CAUSAL_ORDER_BREAK`, left in `_UNIMPLEMENTED_
KINDS`) -- Composer is explicitly allowed to reorder independent ideas, so a blanket
order check would false-positive on legitimate behavior. This narrow version directly
addresses RAW 33366538992's own regression harness findings
(`pimples_micro_order`, `sonography_good_before_diagnosis`), which are exactly this
failure shape.

### PostRenderWatchListenQC: one real structural check, perceptual checks still a gap

`check_render_plan_covers_edit_plan` (`post_render_watch_listen_qc.py`) is a real,
tested function -- not a stub -- but deliberately scoped to the deterministic render
PLAN artifact (`render_plan.RenderSegment`), not decoded MP4 bytes (still unreachable
from this sandbox). It verifies every `keep_sequence` clip is fully time-covered by
some render segment in the same source, correctly tolerating `render_plan.py`'s own
segment-coalescing (which merges touching segments and drops the second's clip_id --
a naive per-clip_id-segment match would false-positive on that legitimate behavior).
This catches a distinct failure mode from the existing text-token-hash freeze check: a
render-plan bug that silently truncates or drops a segment's actual time range without
touching its text. All genuinely perceptual checks (clipped phonemes, fumble frames,
framing, A/V drift, real decode/export integrity) remain unimplemented -- not
fabricated without real signal-processing capability against a reachable file.

### Explicitly NOT built this cycle, stated honestly

- **The automatic review/repair loop** (CanonicalEditPlan v1 -> FinalEditReviewer FAIL
  -> route affected Idea to CompositeResolver/BestTakeResolver -> v2 -> re-review ->
  Freeze v2). This requires re-running Pass 2/3 of `pipeline.py` on a SUBSET of takes
  while provably preserving every other already-valid group's decision untouched --
  a real pipeline restructuring, not a bolt-on, and exactly the kind of change that
  produced this cycle's own two-more-bugs-found-while-building experience above.
  `plan_version` is wired to support it (always `1` until it exists) but the loop
  itself is not built. Today's actual repair path is a human reviewing a FAIL and
  triggering a fresh run -- one bounded "repair attempt", performed by a person.
- **General cross-idea `CAUSAL_ORDER_BREAK` detection.** See STORY_ORDER_BREAK above
  for why the narrow, composite-scoped version was built instead.
- **`INCOMPLETE_DELIVERY`, `ORPHAN_FRAGMENT`, `INCOMPATIBLE_COMPOSITE`.** No existing
  deterministic detector to draw from; unchanged from D-024.
- **PostRenderWatchListenQC's perceptual half** (everything except the one structural
  check above).

### Verification

Full suite: 1328 passed, 1 skipped (same 2 pre-existing unrelated failures excluded as
throughout this session), run repeatedly through this cycle's fixes with consistent
results. 17 new tests across `take_grouping_provider`, `universal_clean_cut`,
`canonical_edit_plan`/`final_edit_reviewer`, and the new post-render structural check.

## D-026 — Automatic targeted repair loop

**Status: CANONICAL**

D-025 left the automatic review/repair loop explicitly unbuilt ("today's actual repair
path is a human reviewing a FAIL"). This cycle builds it, scoped to what can be fixed
without guessing content.

**Mechanism (`repair_loop.py`):** `run_repair_loop(draft)` builds CanonicalEditPlan v1,
reviews it, and -- only for a finding kind with a registered repair strategy -- applies
a targeted repair and rebuilds/re-reviews, bounded at `max_attempts=3`. A repair
mutates only the specific clips a finding names, at their existing positions in
`draft.selected`; nothing else moves, nothing else is discarded or restored. Because
Final Story Coherence Validation's own checks (`lost_semantic_atoms`,
`contradiction_findings`, `missing_idea_coverage`) read `selected`/`discarded` as sets,
not sequences, a pure reorder repair never needs to re-run that whole pass -- only
CanonicalEditPlan (order-sensitive) and FinalEditReviewer are rebuilt, which is what
keeps a repair "targeted" rather than a global re-run.

**Honest scope -- only `STORY_ORDER_BREAK` has a real repair:** reordering an accepted
composite's own components back into recording order is the one repair this
architecture can perform without guessing at content (no other clip's membership,
text, or position is touched). `DUPLICATE_IDEA`, `UNRESOLVED_RETRY`,
`IDEA_COVERAGE_LOST`, `CONTRADICTION`, `UNIQUE_FACT_LOST`, and (D-027, below)
`CAUSAL_ORDER_BREAK` have NO automatic repair, by design: an automatic "fix" for any of
them means guessing which take wins a still-ambiguous contest, which discarded clip to
blindly restore, which side of a contradiction is true, or how to reorder across
independent ideas without undoing an intentional Composer pacing choice. CLAUDE.md's
"WHEN UNCERTAIN, KEEP" and this whole session's established conservatism (deterministic
best-take authority already declines a thin score-gap decision; CompositeResolver's
restore functions already require strong, specific evidence) both say guessing here is
a regression in editorial judgment, not a repair. The loop still records an attempt for
these (audit trail, bounded termination) and always routes straight to
`NEEDS_HUMAN_REVIEW` -- never `PASS`.

**Audit trail:** every `RepairAttempt` records `plan_id`, previous/new `plan_version`,
`finding_kind`, `idea_id`, `owning_authority`, previous/replacement realization,
coverage before/after, `reason`, and `unaffected_ideas_changed` (proven false for every
passing repair test). `universal_clean_cut.py` writes the full attempt list to
`diagnostics["repair_loop"]`, and treats `repair_result.status == "NEEDS_HUMAN_REVIEW"`
exactly like the existing `freeze_blocked` gate -- Selection Freeze never runs on an
unresolved plan.

**Tests (`tests/test_cutsell_repair_loop.py`, + one end-to-end test in
`test_cutsell_universal_clean_cut.py`):** a disordered composite is repaired without
touching an unrelated idea; a valid composite survives repair of a different disordered
composite; `plan_version` increments across a repair; `semantic_hash` is unchanged by a
pure-reorder repair (proving the repair changed ordering only, not semantic content); a
finding with no repair strategy terminates safely as `NEEDS_HUMAN_REVIEW` after exactly
one recorded attempt (never spins to `max_attempts`); a clean plan with no findings
passes with zero repair attempts.

## D-027 — General causal/story order validation

**Status: CANONICAL**

D-025's `STORY_ORDER_BREAK` is deliberately narrow (one composite's own components
only). This cycle adds the general, cross-idea complement the canonical directive
requires: detecting a dependent consequence/continuation/CTA/explanation placed before
(or with) its required context missing from KEEP -- diagnosis before its test,
consequence before cause, continuation before its parent, a dependent explanation
detached from the fact it explains -- without hardcoding any Video00 fact, disease,
phrase, or timestamp.

**Mechanism (`causal_order_validator.py`):** two general, deterministic evidence
sources, exactly as the canonical directive specifies: (1) **source chronology** --
`source_asset_id` + `start`/`end`, scoped to same-source pairs within a
continuous-take gap tolerance (45s, the same kind of adjacency evidence
`take_grouping_provider.py` already uses for retry-family grouping); (2) **connector
language** -- a small, general English+Spanish lexicon of phrases that mark a clause as
a dependent consequence/continuation of whatever preceded it ("therefore", "that's
why", "and that confirmed", "por lo tanto", "como resultado", ...), matched only as a
text PREFIX, never a substring search. A STRONG connector match is sufficient
deterministic evidence on its own. A WEAK/generic connector match ("so", "entonces",
"which means") is treated as insufficient evidence by itself -- exactly the same
"WHEN UNCERTAIN" posture as the rest of this architecture -- and is escalated to the
bounded `CausalOrderArbiter` Protocol; with no arbiter configured, or on any arbiter
exception, the weak hit is dropped silently rather than flagged (false-positive
prevention takes priority, since there is no repair path either way -- see D-026's
scope decision). The required-context search runs over kept AND discarded clips
together (`_clip_pool`), so a required clip that was discarded entirely -- not just
misordered -- is still caught as a detached explanation.

**Wiring:** `final_edit_reviewer.review()` gained an optional `causal_order_arbiter`
parameter and now emits blocking `CAUSAL_ORDER_BREAK` findings
(`owning_authority="StoryValidator"`); `repair_loop.run_repair_loop()` and
`universal_clean_cut.process_universal_clean_cut_sources()` both forward the same
parameter through, defaulting to `None` everywhere -- the already-established fail-open
behavior for an absent arbiter elsewhere in this codebase, not a degraded mode.
`CAUSAL_ORDER_BREAK` was moved out of `_UNIMPLEMENTED_KINDS`.

**Honest gap:** `CausalOrderArbiter` (mirrors `semantic_idea_equivalence.
SemanticEquivalenceArbiter`'s shape/fail-open contract exactly) is a real, usable
Protocol and `find_causal_order_breaks` already calls it when supplied, but no live
Gemini-backed implementation exists yet -- that is a new provider/prompt module (same
shape of work as `semantic_idea_equivalence_google.py`) and is explicitly not built
this cycle. Every caller defaults the parameter to `None` today.

**Tests (`tests/test_cutsell_causal_order_validator.py`, 14 tests, + 2 in
`test_cutsell_canonical_edit_plan_and_reviewer.py`):** valid chronology (no break);
inverted cause/effect (strong connector, blocks); continuation before its parent
(blocks); diagnosis before discovery (generic non-medical fixture, blocks);
independent ideas with no connector language safely reorder (no false positive on
legitimate Composer pacing); a correctly-placed CTA is not flagged; a weak connector
alone is never flagged without a confirming arbiter (false-positive prevention); a
confirming arbiter resolves an otherwise-ambiguous weak hit into a blocking finding; a
denying arbiter drops it; a strong connector hit is never second-guessed by a denying
arbiter; a required clip discarded entirely is caught as a detached explanation; an
arbiter exception is treated as "not available"; clips in the same source far beyond
the gap tolerance are never treated as dependent.

## D-028 — Real PostRenderWatchListenQC media checks (ffmpeg/ffprobe)

**Status: CANONICAL**

D-025 scoped `check_render_plan_covers_edit_plan` to the deterministic render PLAN
artifact only, explicitly not decoded media, because Azure Blob egress remained
blocked (the real Video00 rendered MP4 was unreachable). This cycle confirmed
`ffmpeg`/`ffprobe` install cleanly in this sandbox (`apt-get update -qq && apt-get
install -y --no-install-recommends ffmpeg`) and `numpy` installs via pip -- unlocking
real signal-processing work against SYNTHETIC media fixtures, per the canonical
directive's explicit instruction: "do not fake this with transcript-only checks... do
not wait for Video00 to test basic media-QC behavior."

**What is real and built (`post_render_media_qc.py`), against actually decoded/probed
media, not transcripts:**

- **DECODE_EXPORT_INTEGRITY** -- a full null-decode (`ffmpeg -v error -xerror -i
  <file> -f null -`); any nonzero exit or stderr output is real corruption/truncation.
- **LINGERING_ACCIDENTAL_SILENCE** -- ffmpeg's own `silencedetect` audio filter
  against real decoded audio; a silence longer than a threshold is only flagged when
  it falls outside caller-supplied `protected_pause_windows` (an ordinary parameter,
  never a Video00-specific constant -- whichever upstream authority knows a pause was
  editorially intentional supplies it).
- **FROZEN_OR_REPEATED_FRAME** / **DEAD_BLACK_FRAME** -- ffmpeg's own `freezedetect` /
  `blackdetect` video filters, real frame-to-frame comparison.
- **ABRUPT_AUDIO_DISCONTINUITY** -- at each caller-supplied edit-point timestamp,
  decode a short raw-PCM window via ffmpeg and compare the largest sample-to-sample
  jump against the window's own median jump (numpy); a ratio far above baseline is the
  actual acoustic signature of a hard "click"/step discontinuity at a bad cut, measured
  from the waveform, not inferred from text.
- **STRUCTURAL_DUPLICATE_SEGMENT** (`post_render_watch_listen_qc.py`, deterministic,
  no ffmpeg needed) -- a clip_id appearing in more than one render segment; legitimate
  coalescing merges adjacent segments and keeps only the first's clip_id, so this is
  always a real render-plan bug, never a coalescing artifact.

**Honest gap, explicitly not built:** phoneme-level word-boundary truncation and
unnatural-breath-cut detection (require ASR-phoneme alignment, not an ffmpeg
capability); body/mic/camera reset debris, awkward post-line facial expression, and
face/body jump (require computer-vision/pose estimation -- no cv2/mediapipe-class
library is installed for this purpose, and adding one is a separate, larger decision);
fine-grained (sub-frame) A/V sync drift and framing integrity beyond a gross
resolution mismatch (no motivating fixture or real failure case yet). None of these
are faked with a stand-in heuristic; they are recorded here as unbuilt, exactly like
every other honest gap this session has tracked.

**Authority rule, enforced in code, not just prose:** every finding kind this module
can emit is physical (`is_physical_finding_kind`, `post_render_watch_listen_qc.py`).
`run_post_render_media_qc` asserts this of its own output. `run_bounded_physical_repair
_loop(render_attempt, max_attempts=3)` calls a caller-supplied re-render function
(BoundaryEngine/Renderer's concern, never this loop's) and re-checks, bounded at
`max_attempts`; it raises immediately if ever handed a non-physical finding kind
rather than attempting to "fix" a semantic mismatch by re-rendering -- that must
invalidate the candidate and route upstream instead, exactly as the canonical
directive requires. Exhausting attempts without a clean pass reports
`NEEDS_HUMAN_REVIEW`, never `PASS`.

**Tests (`tests/test_cutsell_post_render_media_qc.py`, 20 tests, all against ffmpeg-
generated synthetic fixtures -- sine tones, testsrc/color frames, constant-amplitude
`aevalsrc` splices):** clean media passes decode integrity; a truncated file fails it;
excessive accidental silence is detected; an intentional protected pause is preserved;
continuous audio with no gap passes; a frozen stretch and a dead-black stretch are each
detected, and a continuously-changing/normal video passes both; a genuine hard
amplitude-jump splice is detected as an audio discontinuity, and a clean boundary
inside continuous audio passes; the orchestrator passes on clean media and
short-circuits to decode-integrity failure on corrupt media; every finding the
orchestrator can emit is confirmed physical; duplicate vs. distinct clip_ids in render
segments; the bounded physical repair loop passes when a later re-render attempt is
clean, terminates as `NEEDS_HUMAN_REVIEW` after exactly `max_attempts` (never spins
further), never claims `PASS` while a finding remains, and refuses (raises) if ever
handed a non-physical finding kind. Skipped, not failed, if `ffmpeg` is absent from
the runner (a robustness guard; confirmed present and used for real on the standard
CI ubuntu-latest image and in this sandbox).

## D-029 — RAW gate assessment after D-026/D-027/D-028 (repair loop, causal order, real media QC)

**Status: CANONICAL, conditions 6/7 superseded by D-030**

Conditions 6 and 7 below were reported NOT MET at this checkpoint (real
PostRenderWatchListenQC / bounded physical repair existed and were tested, but nothing
in the live pipeline called them). D-030 (next entry) wires both into the real export
job and reports them MET. This entry is kept as the honest historical record of that
gap, not rewritten.

Closing checkpoint for the "three remaining architectural gaps" directive (repair loop,
general causal/story order validation, real PostRenderWatchListenQC), assessed against
the directive's own 12-condition RAW gate. Full suite: 1094 passed (`pytest -q
tests/test_cutsell_*.py`, the exact CI glob), `compileall cutsell_worker cutsell_app`
clean. Reported honestly, condition by condition, rather than a generic "done":

1. Targeted repair loop green -- **MET** (D-026, `tests/test_cutsell_repair_loop.py`).
2. Plan versioning across repairs green -- **MET** (`plan_version` increments,
   `semantic_hash` stable across a pure reorder -- tested).
3. Unaffected Idea stability proven -- **MET** (`unaffected_ideas_changed` asserted
   false in both a valid-composite-survives and an unrelated-idea-untouched test).
4. General STORY_ORDER_BREAK green -- **MET** (D-025, unchanged, still passing).
5. General CAUSAL_ORDER_BREAK green -- **MET** (D-027,
   `tests/test_cutsell_causal_order_validator.py`, 14 tests + 2 end-to-end).
6. Real PostRenderWatchListenQC active on rendered media -- **NOT MET as an active
   pipeline stage.** D-028 built and tested real ffmpeg/ffprobe checks against actual
   decoded media (not transcript-only, not faked) and validated them against synthetic
   fixtures without waiting for Video00, exactly as the directive required for
   buildability/testability. What is honestly still missing: `post_render_media_qc.py`
   is not called from anywhere in the live pipeline today (confirmed by search -- no
   reference outside its own module and docstrings elsewhere). Wiring it in means
   invoking it from wherever the real render actually happens (`render.py`'s export
   step is the natural integration point) against the REAL RunPod output file, which
   this sandbox cannot reach today (Azure Blob egress remains blocked) and was not
   attempted this cycle -- a real pipeline-wiring change deserves its own careful
   review, not a rushed addition immediately before a paid RAW.
7. Bounded physical repair loop green -- **NOT MET as an active pipeline stage**, same
   root cause as #6: `run_bounded_physical_repair_loop` is real and tested (with a
   caller-supplied `render_attempt` function standing in for BoundaryEngine/Renderer,
   since no real re-render is invoked from this sandbox either), but nothing in the
   live pipeline calls it yet.
8. Invalid semantic plan still cannot Freeze -- **MET** (unchanged; `freeze_blocked`
   now also fires on `repair_result.status == "NEEDS_HUMAN_REVIEW"`, D-026).
9. Composite persistence remains green -- **MET** (D-025's `protected_ids` fix,
   unchanged, still passing).
10. CoverageLedger remains green -- **MET** (`lost_semantic_atoms` checks, unchanged,
    still passing).
11. CleanCutBench materially green -- **MET** (1094 passed, including the 2 new D-026/
    D-027 end-to-end integration fixtures added this cycle).
12. CI green -- **MET** (exact CI glob green; `compileall` clean).

**Honest conclusion: 10 of 12 conditions are met; 6 and 7 are not**, for the same
reason (real post-render media QC exists and is proven correct against synthetic
media, but is not yet an active stage of the real render pipeline that would run
against the RAW's actual output). Per the directive's own framing ("push once and
allow ONE controlled Video00 RAW... when the gate is truly met"), this is reported as
NOT fully met rather than rounded up -- consistent with this whole session's standing
practice of surfacing an honest gap instead of overclaiming readiness for a paid,
hard-to-verify action. No push and no RAW triggered this cycle; the standing HOLD
remains in effect pending the user's decision on how to proceed (wire #6/#7 into the
live pipeline first, or accept the built-and-tested state and proceed with the RAW
with #6/#7 as a known follow-up).

## D-030 — Live wiring: PostRenderWatchListenQC + bounded physical repair active in the real export pipeline

**Status: CANONICAL**

D-029 reported RAW-gate conditions 6/7 as NOT met: D-028's real ffmpeg/ffprobe media
checks existed and were tested against synthetic fixtures, but nothing in the live
pipeline called them. This cycle closes that gap by wiring them into the actual
execution path, per the required live order:

```
Validated CanonicalEditPlan -> Selection Freeze -> BoundaryEngine
-> Render actual MP4 -> PostRenderWatchListenQC on that actual file
-> PASS -> final output
   NO physical issue -> targeted Boundary repair -> re-render -> QC again
   NO semantic mismatch -> invalidate candidate / route upstream (never "fixed" by Boundary)
```

**Where it runs:** `export_job.py`'s `run_export_job` -- the actual RQ job that
downloads sources, builds the render plan, and used to call `render.render_preview(...)`
directly -- now calls `live_render_qc.render_with_post_render_qc(...)` instead. QC runs
against the REAL LOCAL rendered file the RunPod worker already has on disk, strictly
BEFORE `store_export(...)` uploads anything -- never a downloaded-back artifact (Azure/S3
egress is irrelevant to this check).

**New modules:**
- `live_boundary_repair.py` -- `repair_segment_for_finding`: a targeted, single-segment,
  Boundary-only physical repair. Trims ONE `RenderSegment`'s leading or trailing edge by
  a physical defect's own duration, using `segment_output_windows` (built on `render.
  tighten_trailing_silence`, the SAME per-segment trim the real renderer already applies,
  so this mapping from output-timeline offsets back to segments never drifts from what
  actually gets rendered). Refuses (`None`) a mid-segment defect a boundary trim cannot
  reach, or a trim that would eat too much real content -- never guesses.
- `live_render_qc.py` -- `render_with_post_render_qc`: the orchestrator. Renders, runs
  the deterministic structural checks (segment coverage/order/duplication vs. the frozen
  CanonicalEditPlan) ONCE against the untouched attempt-0 segments, then D-028's real
  media checks; on a physical finding, calls `repair_segment_for_finding`, re-renders,
  and re-checks, bounded at `max_attempts`; on ANY non-physical (structural/semantic)
  finding, invalidates the candidate immediately and never calls the repair function at
  all. `_resolve_edit_plan` preserves a repair-loop-derived `plan_version` from
  `draft.diagnostics["canonical_edit_plan"]` when present, rather than resetting it to 1.
- `post_render_watch_listen_qc.py` gained `check_render_sequence_matches_edit_plan`
  (`STRUCTURAL_SEQUENCE_MISMATCH`): the render segments' clip_id order must be consistent
  with `keep_sequence`'s order (accounting for coalescing dropping a merged segment's
  second clip_id) -- catches "rendered spoken sequence matches frozen CanonicalEditPlan" /
  "semantic order unchanged" for real, which the existing coverage-only check did not.
- `render.py`'s `_tighten_trailing_silence` is now public (`tighten_trailing_silence`) so
  `live_boundary_repair.py` shares exactly one implementation of the real per-segment
  render duration, not a second guess.

**A real bug found and fixed via this cycle's own integration tests, not by inspection:**
the structural coverage check was initially re-run after every physical repair attempt --
and a legitimate, Boundary-authorized edge trim (shrinking a segment's `end` to remove a
dead-air tail) always makes that segment's covered time-range smaller than the frozen
`keep_sequence` clip's full original range, so the coverage check immediately (and
wrongly) reported it as `STRUCTURAL_SEGMENT_TRUNCATED` -- a semantic mismatch that would
have made the physical repair loop invalidate its own repair on every single attempt,
never actually reaching a second, corrected render. Fixed: structural checks run ONCE,
against attempt 0's own untouched segments, before any physical repair -- a physical
repair is Boundary's own authorized territory (identical in kind to `tighten_trailing_
silence`'s existing, already-accepted silence trim), not something a coverage check
should re-litigate on every retry.

**Authority rule enforced in code:** `is_physical_finding_kind` is the single source of
truth for what may ever reach `repair_segment_for_finding`; every `STRUCTURAL_*` kind and
any future non-physical media finding is treated as a semantic mismatch, invalidates the
candidate immediately, and is never retried or "fixed" by Boundary.

**Tests (`tests/test_cutsell_live_render_qc.py`, 9 tests covering the directive's 10
required proof points -- items 3/4 share one test):** real ffmpeg rendering and the real
deterministic structural checks are exercised unmocked throughout; the physical-defect
DETECTION layer is scripted for the repair-loop tests (D-028's own suite already
exhaustively covers real detector correctness against synthetic fixtures -- this file's
job is the orchestration/wiring risk this cycle actually introduced). Covers: render
actually invokes PostRenderWatchListenQC; a clean render passes directly; a physical
failure triggers a real targeted Boundary repair AND a real re-render, re-QC'd; the
bounded loop never spins past `max_attempts`; a semantic/structural mismatch never
reaches the repair function and is never re-rendered; a semantic mismatch blocks
delivery (`output_path=None`); a missing composite/KEEP segment fails QC; a reversed
render order fails the new sequence check; a final PASS's `plan_id`/`semantic_hash`
match `build_canonical_edit_plan(draft)` exactly, and a carried-over `plan_version` (from
an earlier D-026 repair) is preserved rather than reset. Existing
`tests/test_cutsell_clean_worker_export.py` unit test updated to mock the new
orchestrator instead of the now-internal `render_preview` call.

**RAW-gate update (supersedes D-029's #6/#7):** condition 6 (real PostRenderWatchListenQC
active on rendered media) and condition 7 (bounded physical repair loop active) are now
**MET** -- both are exercised for real, against the actual local rendered file, inside the
real export job, with integration tests proving the full render -> QC -> repair -> re-
render -> re-QC cycle and the semantic-mismatch-never-touched-by-Boundary invariant.

**Full suite:** 1103 passed (`pytest -q tests/test_cutsell_*.py`, exact CI glob),
`compileall cutsell_worker cutsell_app` clean.

## D-031 — Semantic-atom importance classification: CRITICAL/CONTEXTUAL/UNCERTAIN

**Status: CANONICAL**

RAW 33402023395 (D-029/D-030's own RAW) exposed a real CoverageLedger over-
conservatism, not a Selection regression: a discarded clip's incidental year
("...en 2023.") blocked Selection Freeze under the old unconditional rule --
"any missing number/negation atom blocks" -- even though the Human Gold
oracle itself does not preserve that year in its own equivalent delivery.
The audience-facing idea (endoscopy -> diagnosis -> medication) was fully
intact in the winning take. This generalizes that rule.

**Mechanism (`semantic_atom_importance.py`, new module):** every missing
number/negation atom `final_story_coherence_validation._lost_semantic_atoms`
finds is now classified CRITICAL, CONTEXTUAL, or UNCERTAIN before deciding
whether it blocks Freeze:

- A negation is always CRITICAL (flips a claim's truth value outright).
- A number is CRITICAL when its own clip's text carries a general
  percentage/price/measurement/dose marker, correction language
  ("instead of", "actually", "corrijo", ...), or a chronology-relation
  phrase ("before that", "since then", "antes de eso", ...).
- A number is CONTEXTUAL only when it is a bare, plausible-year-shaped
  value (1900-2099) in an ordinary temporal-aside clause with NONE of the
  above markers present -- the canonical directive's own "during one
  period in 2023 I had stomach problems" example.
- Anything else is UNCERTAIN. `blocks_freeze(importance)` treats UNCERTAIN
  exactly like CRITICAL -- "WHEN UNCERTAIN, KEEP" means an atom this
  deterministic layer cannot confidently clear as safe-to-lose stays
  blocking, never silently downgraded to a warning. No Video00 fact,
  phrase, or literal value is hardcoded anywhere in the marker vocabulary.

A bounded `SemanticAtomImportanceArbiter` Protocol exists for resolving a
genuinely ambiguous UNCERTAIN atom with minimal context (mirrors this
codebase's other bounded arbiters' fail-open contract exactly: no arbiter,
an arbiter exception, or a malformed verdict all leave the atom UNCERTAIN).
No live implementation exists yet -- same honest-gap pattern as
`CausalOrderArbiter` (D-027) -- every caller defaults it to `None`.

**Policy change, scoped narrowly:** `_lost_semantic_atoms` now returns an
`atom_classifications` list and a `blocking` boolean per finding.
`final_story_coherence_validation`'s own `freeze_blocked` and
`final_edit_reviewer.review()`'s `UNIQUE_FACT_LOST` finding both now respect
`blocking` -- a CONTEXTUAL-only atom loss is a non-blocking warning (same
shape as `REQUIRED_CONTINUATION_LOST`); a critical/uncertain atom, or the
BROADER content-vocabulary-loss signal (unchanged, still unconditionally
blocking), still blocks exactly as before. `contradiction_findings` and
`missing_idea_coverage` are completely unaffected. Protection is NOT
weakened for papillary-cancer-shaped diagnoses, sonography/nodule-shaped
measurements, pimples/rash-shaped ideas, or any number/negation that
actually changes meaning -- all of those hit a CRITICAL rule (measurement,
dose, correction, chronology, or plain negation) or fall through to
UNCERTAIN, which still blocks.

**Tests:** 15 unit tests in `tests/test_cutsell_semantic_atom_importance.py`
(negation always critical; incidental-year contextual; correction-language
critical -- the directive's own "instead of" example; chronology-relation
critical; percentage/price/measurement/dose critical; bare ambiguous
quantity uncertain-and-blocks; arbiter confirms/never-second-guesses-a-
deterministic-critical-verdict/exception-handling/malformed-verdict). 2 new
tests in `tests/test_cutsell_final_story_coherence_validation.py` (the exact
RAW 33402023395 shape: contextual year does not block; a critical
measurement alongside an incidental year still blocks). 1 new test in
`tests/test_cutsell_canonical_edit_plan_and_reviewer.py` (a CONTEXTUAL-only
row surfaces as a non-blocking `UNIQUE_FACT_LOST` warning, not a FAIL). 7
new CleanCutBench fixtures in `tests/test_cutsell_clean_cut_core_evaluation_
suite.py`, reached through the REAL take-grouping/idea-equivalence/take-
judge/coherence chain, covering all ten of the canonical directive's named
categories (several collapse onto the same CRITICAL/CONTEXTUAL split):
incidental year safely omitted; year required for chronology; numeric
measurement/percentage/dose must survive; redundant date repeated in two
attempts (never even flagged as missing); an ambiguous atom with no
deterministic signal stays blocking without an arbiter.

Full suite: 1128 passed (`pytest -q tests/test_cutsell_*.py`, exact CI glob).

## D-032 — Human Gold regression aligner: ordered semantic alignment replaces positional diffing

**Status: CANONICAL**

RAW 33402023395's CI-reported "23 -> 20, content missing" alarm was investigated by
hand against the actual KEEP text (see the RAW's own diagnosis report) and found to be
a false positive: `benchmarks/validate_video00_selection_lock.py` compared the
candidate against the Human Gold baseline INDEX BY INDEX, so one benign re-chunking
(the baseline's "...funcionando" / "perfectamente." split into two array entries where
this run produced one merged sentence) shifted every later index and cascaded into a
wall of false `missing_segment`/`text_changed` errors. `benchmarks/validate_video00_
regression_qa.py`'s `required_exact`/`required_order` checks were already position-
independent but required byte-for-byte text equality, so the same re-chunking (and
ordinary ASR wording variance) still made genuinely-present facts register as missing.
The Human Gold oracle itself was never wrong; the comparison mechanism was too brittle.

**Mechanism (`benchmarks/video00_semantic_alignment.py`, new module, zero import from
`cutsell_worker` -- Human Gold stays QA/oracle only):** `align(gold_segments,
candidate_segments)` walks both lists together, in order, using CONTENT-TOKEN overlap
(not exact text equality) with a growable window on either side, recognizing:

- `EXACT` -- one gold segment, one candidate segment.
- `RECHUNKED` -- 2+ gold segments merged into one candidate segment (or vice versa on
  the candidate side never happening simultaneously with the gold side -- see below).
- `COMPOSITE` -- one gold segment realized across 2+ candidate segments.
- `MISSING` -- no candidate window anywhere ahead explains a gold segment.
- `EXTRA` -- candidate content nothing in gold asked for.
- `DUPLICATE` -- extra candidate content that closely repeats an already-matched gold
  segment (the same idea rendered twice).

Because the walk only ever moves forward on both sides, a candidate realization
belonging to a LATER gold idea appearing before an EARLIER one's realization cannot be
"found" by looking backward -- the earlier gold segment correctly reports `MISSING`
(a real ordering break), never a silently-accepted out-of-order match.

**Three real bugs found and fixed via this module's own tests, not by inspection:**
1. Greedy smallest-window-first matching could consume a candidate segment via a
   narrow 1:1 match before checking whether it actually belonged to a wider merge,
   stranding the true second half of a rechunk as falsely `MISSING`. Fixed: matches
   must be BIDIRECTIONAL (gold's content covered by the candidate window AND the
   candidate window's content explained by gold), and a window may only grow on ONE
   side at a time (gold OR candidate, never both) -- growing both simultaneously is
   bag-of-words matching two independent multi-segment spans against each other,
   which is order-blind and cannot tell `gold=[A,B]` from `candidate=[B,A]` apart.
2. A `skip` (tolerating an unrelated EXTRA candidate segment in the way) could mask
   genuine reordering: skipping ahead to find idea A's content might jump straight
   over idea B's content sitting where A should have been. Fixed: a skip is refused
   when the skipped content itself looks like it belongs to any remaining gold idea.
3. Aggregate coverage across a multi-segment window can mathematically "average out"
   a completely absent segment behind a strongly-covered neighbor it happens to be
   windowed with. Fixed: every individual segment inside a 2+-segment window must
   also individually clear a lower floor on its own -- except a near-content-free
   trailing fragment (<=2 content tokens, e.g. "perfectamente."), which is exempt
   since it cannot independently prove or disprove anything and exists only to be
   merged with its real neighbor (the legitimate rechunk case, not a bug).

**Validated against the actual RAW 33402023395 data**, not just synthetic fixtures:
reconstructing that run's real 20-clip KEEP sequence and running the rewritten
`validate_video00_selection_lock.py` against the real 23-segment `video00_selection_
lock.json` baseline now reports `selection_locked: true`, `historical_regression_qa_
pass: true`, all 18 named checks passing (including every one RAW 33402023395's run
had reported failing: `papillary_cancer_preserved`, both `sonography_good_take_*`,
all three `pimples_micro_*`, `pimples_later_winner_present`, `family_context_
preserved`, both `required_order` checks), `error_count: 0` -- confirming that
candidate's Selection was correct all along.

**Count is no longer a hard gate, per the canonical directive's explicit "do not force
exactly N segments":** both scripts now record a changed segment count as a
non-blocking warning, not a failure -- `selection_locked`/`qa_pass` are driven by
whether the alignment/named-fact checks actually pass, never by count equality alone.

**Tests:** 12 unit tests in `tests/test_video00_semantic_alignment.py` (exact,
rechunk, composite, missing, extra, duplicate, reordering-reports-missing, ASR wording
variance, extra-segment-does-not-break-alignment, tiny-fragment-exempt-from-floor,
substantial-content-never-masked, empty input). `tests/test_video00_regression_qa.py`
updated (4 tests, richer non-degenerate fixture text, count-drift-is-a-warning).
`tests/test_video00_selection_lock.py` (new, 4 tests: identical locks, benign rechunk
does not break the lock, genuinely missing content does, duplicate rendered segment
does). Full suite (including these, outside the `test_cutsell_*.py` CI glob): 1422
passed (`pytest -q tests/`, excluding the two long-pre-existing unrelated broken
files this session has documented throughout).

## D-033 — CI diagnostic printing for the D-026/D-030/D-031 architecture

**Status: CANONICAL**

The RAW workflow's "Print unified Selection reasoner diagnostics" step never surfaced
`diagnostics.canonical_edit_plan`, `diagnostics.repair_loop`, or `diagnostics.final_
edit_reviewer` -- diagnosing RAW 33402023395 required inferring their state from
`stage_status`'s flat `"final_edit_reviewer": "FAIL"` string alone, cross-referenced
against `final_story_coherence_validation`'s own detail by hand. `.github/workflows/
cutsell-video00-raw-v5-auto-microtrim.yml` now also echoes: a curated `canonical_edit_
plan` summary (`plan_id`/`plan_version`/`semantic_hash`/`validation_state`/
`freeze_blocked`/per-Idea `coverage_status`+`is_composite`+winning/discarded clip ids);
the full `final_edit_reviewer` findings+warnings; the full `repair_loop` attempt
history; `selection_boundary_contract` (the frozen plan's own id/version/hash, when
freeze happened); and a `lost_semantic_atoms` atom-classification extract (D-031's
`blocking`/`atom_classifications` per row) pulled out for readability. Every new jq
expression was validated against a synthetic result JSON (both present and `absent`
shapes) before landing, since a jq syntax error here would only otherwise surface
mid-paid-RAW. No production code changed; this is CI-log observability only.

## D-034 — RAW 33409169518 findings: D-031/D-032 confirmed live; two open gaps found, neither fixed yet

**Status: DIAGNOSIS RECORDED — CURRENT LIVE BLOCKER, awaiting explicit direction before further code changes or another RAW**

RAW 33409169518 (source: `9fad0788e...` lineage plus commits `5ce26eb`, `cdb55ae`,
`2a03880` -- D-031/D-032/D-033) was launched per the D-031/D-032/D-033 RAW gate.
Semantic-side result, confirmed from the run's own printed diagnostics:

- `final_story_coherence_validation.freeze_blocked: false` (RAW 33402023395 had this
  `true`, blocked solely by the "2023" atom). The same atom is present again in this
  run's data (`clip_aeb69adfb81e1d9c2296`, "Tuve problemas de estómago en una
  temporada en 2023.") and is now correctly classified
  `{"atom": "2023", "atom_type": "number", "evidence":
  "incidental_year_in_ordinary_temporal_aside", "importance": "CONTEXTUAL",
  "resolved_by": "deterministic"}` with `"blocking": false` -- D-031 works live, not
  just in unit tests.
- `final_edit_reviewer`: `status: PASS`, `findings: []`, one non-blocking `warnings`
  entry (the same atom, `"blocking": false`) -- D-031's blocking-vs-warning routing
  is confirmed live.
- `repair_loop`: `attempt_count: 0`, `status: PASS` -- no targeted semantic repair was
  needed pre-freeze this run.
- `canonical_edit_plan`: `validation_state: frozen_ready`, `freeze_blocked: false`,
  3 retry-family Ideas recorded (matching this run's 3 `take_judge_groups`), each
  `coverage_status: complete`.
- `selection_boundary_contract`: `status: verified`, `matches_reviewed_plan: false`.
  This is the documented, non-enforced observability case from
  `selection_boundary_contract.py` (`enforce_complete_idea_boundaries` legitimately
  runs between FinalEditReviewer's PASS and freeze and can restore source-proven
  leading/trailing words) -- not a new problem.
- The RAW workflow's own "Verify frozen Selection lock" step (D-032's aligner)
  **passed for the first time** on a real RAW.

Despite all of the above, the workflow's overall `conclusion` was `failure`, from a
later step, "Verify unified Selection architecture." Root-causing that step (not yet
fixed -- diagnosis only) surfaced two separate, real findings:

1. **Step 18 asserts legacy `unified_selection_reasoner`-era fields that are
   structurally never true under the active Clean Cut Core V1 path.** It checks
   `selection_reasoner_enabled == true`, `selection_reasoner_status == "applied"`,
   `external_brain_calls_enabled == true`, `hybrid_requested_group_count == 0`, etc.
   This run's own diagnostics show `diagnostics.unified_selection_reasoner:
   {"status": "absent"}`, and `universal_clean_cut_validation.
   run_single_universal_clean_cut_validation` derives `selection_reasoner_status`
   from that same absent dict (`unified_diag.get("status")` -> `None`). Clean Cut
   Core V1 deliberately deactivates the whole-video Unified Selection reasoner (see
   Current mission, CLAUDE.md) -- so this check can never pass while V1 is active by
   design, regardless of edit quality. This looks like a pre-existing CI-script bug
   that checks the wrong (legacy) architecture's success markers, not a regression
   from D-026 through D-033: step 18 was never reached to completion in an earlier
   RAW this session observed (step 17 used to fail first), so this is the first time
   it has been exercised at all. Not yet fixed; the workflow's own verification
   script needs updating to assert Clean Cut Core V1's actual success markers
   instead, but that change has not been made without explicit direction.

2. **New, more significant finding: the Video00 RAW benchmark harness never
   exercises D-030's live PostRenderWatchListenQC / bounded physical repair wiring
   at all.** `serverless_handler._focused` (the RunPod op this workflow submits,
   `op: "focused"`) calls `universal_clean_cut_validation.
   run_single_universal_clean_cut_validation`, whose preview render path is
   `_render_validation_preview` -> `render.render_preview` -- a bare ffmpeg
   concat+overlay function with no QC and no repair loop. D-030's actual live
   wiring (`live_render_qc.render_with_post_render_qc`, the bounded physical repair
   loop, and every `render_attempt`/`post_render_qc_status`/`plan_id`/`plan_version`
   /`semantic_hash` diagnostic D-030 added) lives only in `export_job.run_export_job`
   -- the separate, real mobile-app export RQ job path, never invoked by the Video00
   RAW harness. Consequently: no RAW run to date (including 33409169518) has ever
   produced a `PostRenderWatchListenQC` verdict, a `render_attempt` record, or
   exercised the bounded physical repair loop against a real Video00 render, even
   though D-030's live wiring is code-complete, unit-tested (10 integration tests),
   and separately proven correct in isolation. This is a structural gap between the
   RAW benchmark harness and the production export pipeline that predates this
   session's D-026/D-030 work (the harness function was never updated to route
   through the new live-wired path) -- not something this session's changes broke,
   but also not something this session's changes have yet closed. Closing it would
   mean routing `run_single_universal_clean_cut_validation`'s preview render through
   `render_with_post_render_qc` (or an equivalent path) so a Video00 RAW actually
   proves PostRenderWatchListenQC/repair-loop behavior end-to-end, per Directive A's
   original mandate. Not yet done; no code changed for this finding.

**No further code changes and no further RAW were made after this diagnosis.** Per
the standing acceptance rule (do not call a run successful merely because CI passed
or a RAW completed; do not launch another RAW off a failing run without root-causing
it first), both findings above were reported instead of silently patched or
retried around.

## D-035 — Closing the two D-034 gaps: single-path live render/QC for Video00 RAW, and a Clean-Cut-V1-shaped architecture verifier

**Status: CANONICAL**

Both gaps D-034 found are closed. Neither changed retry/idea/composite/coverage
semantics; the semantic Clean Cut Core baseline validated live in RAW 33409169518
is unchanged and preserved as-is.

**Fix 1 -- the stale architecture verifier is replaced, not deleted.**
`benchmarks/validate_video00_architecture.py` (new) asserts evidence the CURRENT
Clean Cut Core V1 architecture actually ran, instead of the old whole-video Unified
Selection reasoner's success markers (`selection_reasoner_status == "applied"`,
`external_brain_calls_enabled == true`, `hybrid_requested_group_count == 0`, which
can never be true while V1 -- which deliberately deactivates that reasoner -- is
active). It checks: Clean Cut Core V1 is the active semantic authority; the
whole-video reasoner being absent/disabled is asserted as the EXPECTED state, not an
error; CanonicalEditPlan was built (`plan_id`/`plan_version`/`semantic_hash`
present); FinalEditReviewer executed (status PASS/FAIL); the repair loop reported a
status; CoverageLedger/StoryValidator (Final Story Coherence Validation) executed;
SWAP is absent (`alternate_count == 0`, D-019) asserted as EXPECTED, not an error;
CompositeResolver diagnostics are present; and, when the candidate was not
freeze-blocked, that the plan was `frozen_ready`, that Selection Freeze references
the exact validated plan id/version/hash, that Boundary only ran after that freeze,
and that the live render/QC service (Fix 2, below) actually ran against the real
render. When the candidate WAS freeze-blocked, it instead asserts the
semantic-failure-blocks-freeze gate actually engaged (Boundary/render never ran on
an unfrozen draft) rather than requiring stages that legitimately never happened.
The workflow's "Verify unified Selection architecture" step now runs this script
(keeping a few small architecture-agnostic sanity checks -- source duration,
speech-lock, non-empty selection -- inline) instead of the old `jq -e` assertion.
20 targeted tests (`tests/test_video00_architecture.py`) cover a valid V1 run, the
absent-reasoner/absent-SWAP EXPECTED cases, missing-CanonicalEditPlan/
FinalEditReviewer/freeze-contract failures, and the freeze-blocked shape (including
a case where Boundary incorrectly ran despite a block -- caught). Validated against
a reconstruction of RAW 33409169518's own real diagnostics: passes on every check
except `live_render_qc_ran_against_the_real_render`, which correctly fails because
Fix 2 (below) had not landed yet at that run -- proof the verifier is not a rubber
stamp and would have caught Fix 2's own gap unassisted.

**Fix 2 -- the Video00 RAW harness now renders through the exact same live
render/QC service the real export job uses; there is no second implementation.**
`universal_clean_cut_validation._render_validation_preview` (previously a bare
`render.render_preview()` call) now calls `live_render_qc.render_with_post_render_qc`
-- the identical function object `export_job.run_export_job` already uses (D-030),
imported from `live_render_qc.py` in both places; `tests/test_cutsell_universal_
clean_cut_validation_live_render_qc.py`'s
`test_export_job_and_validation_harness_share_one_render_qc_implementation` asserts
object identity directly, not just behavioral similarity. The harness gained a
`freeze_blocked` parameter (from `result.stage_status["freeze_blocked_pending_
coherence_review"]`): "if semantic validation fails, no render" is enforced before
Boundary/Render/QC is ever attempted, matching the canonical live order exactly.
`run_single_universal_clean_cut_validation`'s return value gained a `live_render_qc`
key (`status`, `output_path`, `plan_id`, `plan_version`, `semantic_hash`,
`render_attempt_count`, full per-attempt `attempts`) alongside the existing
`preview_path`/`preview_skipped_reason`, and `serverless_handler._focused`'s compact
RunPod output surfaces the same fields for at-a-glance visibility without
downloading the full result JSON. The workflow's diagnostic-printing step gained a
`live_render_qc` echo block. 7 targeted tests (real, unmocked ffmpeg throughout,
skipped if ffmpeg is absent) prove: the harness actually invokes the shared service;
a clean candidate renders and passes; a physical failure triggers bounded Boundary
repair and the repaired candidate is re-rendered/re-QC'd; a structural/semantic
mismatch is never routed to Boundary; a freeze-blocked draft never reaches
render/QC at all; and the final artifact's `plan_id`/`semantic_hash` match the
draft's own `CanonicalEditPlan` exactly. No new render/QC implementation was
written -- this is entirely reuse of the D-030 service, per the single-path rule.

**Validation**: full targeted suite green (`tests/test_video00_architecture.py`,
`tests/test_cutsell_universal_clean_cut_validation_live_render_qc.py`, updated
`tests/test_cutsell_universal_clean_cut_validation_empty.py`), the full
`tests/test_cutsell_*.py` CI glob green (1136 passed), `python -m compileall`
clean, and the new architecture verifier independently validated end-to-end against
a reconstruction of RAW 33409169518's real diagnostics.

## D-037 — Physical fragment identity + authoritative delivery gate

**Status: CANONICAL**

Root-caused and fixed the exact defect that invalidated RAW 33415661351's render.
The semantic Selection result from that run (`selection_locked: true`, `error_count:
0`, all 19 named regression checks passing) is unchanged and preserved as-is; this
work is entirely downstream, physical/render infrastructure.

**Root cause, confirmed by tracing the real data flow** (Selection Freeze -> Boundary
-> render segment construction -> render plan -> PostRenderWatchListenQC):
`human_boundary_polish_v5._remove_micro_visual_reset_word_gaps` splits an
already-frozen semantic clip into physical left/right pieces at a micro visual-reset
word gap via `dataclasses.replace(clip, start=..., end=...)` -- correct and intended
Boundary behavior (`"semantic_membership_changed": False`) -- but kept the exact same
`clip_id` for every resulting piece, and can split repeatedly (each detected gap
re-splits the growing piece list). `check_no_duplicate_render_segments` (D-030) had
no way to distinguish that from a real duplicate, since it keyed purely on raw
`clip_id` equality. This is the ONE semantic clip -> multiple legitimate PHYSICAL
CHILD fragments the user's hypothesis named; `post_selection_interior_gap_trim.py`'s
own splitter already mints a fresh id per child (via `_child_id`, a
content-and-boundary hash) and was never the problem -- only `human_boundary_polish_
v5` reused identity across siblings.

**1. Explicit identity contract (semantic vs. physical), on the repo's existing
types -- no parallel provenance system.** `contracts.DraftClip` gained five optional
fields, following the exact precedent the `signals` field already set (all default
`None`, every existing construction site unaffected): `render_fragment_id` (unique
per physical piece), `parent_semantic_clip_id` (the semantic clip_id every sibling
reconstructs together -- deliberately requires POSITIVE evidence, never inferred
from bare `clip_id` equality), `fragment_index`/`fragment_count` (position among
siblings), `boundary_reason` (which Boundary operation produced it). Two helpers,
`effective_render_fragment_id`/`effective_parent_semantic_clip_id`, fall back to
`clip_id`/`None` for anything nobody has ever split. `render_plan.RenderSegment`
(the existing PhysicalRenderPlan representation -- `CanonicalEditPlan` remains the
untouched semantic source of truth, never rewritten into physical terms) carries the
identical fields through from `build_render_plan`.

**2. `human_boundary_polish_v5` now mints real identity.** Every split piece gets a
`render_fragment_id` derived from `sha256(clip_id|human_boundary_polish_v5|start|end)`
(mirroring `_child_id`'s pattern) and a `parent_semantic_clip_id` resolved to the
TRUE root even when re-splitting an already-split fragment (`clip.parent_semantic_
clip_id or clip.clip_id`, computed once per input clip) -- so every physical sibling
of one frozen delivery stays discoverable under one shared key regardless of how many
Boundary passes touched it. `clip_id` itself is never mutated.

**3/4. `check_no_duplicate_render_segments` rewritten to reason from provenance, not
raw `clip_id` equality.** Three tiers: (A) a repeated `render_fragment_id` is always
a bug, full stop, regardless of parent bookkeeping; (B) segments sharing an explicit
`parent_semantic_clip_id` must reconstruct that parent as one contiguous
(render-order-adjacent), non-overlapping, correctly time-ordered run -- a violation
(scattered position, reordering, or overlap) still fails, exactly as a real
audience-perceptible duplicate/repetition should; (C) segments with NO fragment
provenance at all are judged exactly as this check always has (bare `clip_id`
collision = fail) -- legitimacy requires positive evidence, so this never silently
waves through an unrelated bug from code that hasn't been updated to set the new
fields. 7 targeted tests cover both fixture-level checks and, via `render_with_post_
render_qc` end-to-end with real ffmpeg, the exact RAW 33415661351 shape now passing
plus a deliberately-reordered variant that still correctly fails.

**5. `LiveRenderQCResult` gained the one authoritative delivery gate**: a `deliverable`
property (`True` iff `status == "PASS"`) and a `delivery_status` string
(`"DELIVERABLE"` / `"NOT_DELIVERABLE_<status>"`) -- read by every caller (the
Video00 RAW harness; `export_job.run_export_job` already enforced the equivalent via
`PostRenderQCFailure`) rather than each re-deriving it.

**6. Unconditional artifact upload fixed.** `serverless_handler._focused()`
previously uploaded `preview.mp4` regardless of QC outcome -- meaning a run's
"deliverable" artifact could actually be the QC-invalidated candidate (exactly what
happened on RAW 33415661351: the duplicate-segment MP4 was uploaded and downloadable
as if it were the final output). Now: a `deliverable` candidate uploads under the
unchanged `preview.mp4` name; anything else uploads ONLY as
`diagnostic-invalidated-preview.mp4`, with `preview_uri` left null in both the full
result and the compact RunPod output, so nothing downstream can mistake a diagnostic
render for a deliverable one. Auto-speech-visual-microtrim (a separate, pre-existing
feature) is skipped entirely on a non-deliverable candidate -- no reason to spend ASR
work refining a render that will not ship. The workflow's "Download unified Selection
artifact" step was updated to match: it now downloads `result.json` independently of
whether a preview exists at all, then fetches whichever of `preview_uri`/
`diagnostic_preview_uri` the run actually produced (clearly named either way) rather
than requiring both non-empty as one combined guard, which would have silently
skipped downloading `result.json` itself on every non-deliverable run.

**Validation**: 22 new/changed targeted tests across
`tests/test_cutsell_human_boundary_polish_v5.py` (new),
`tests/test_cutsell_post_render_media_qc.py`,
`tests/test_cutsell_live_render_qc.py`, and
`tests/test_cutsell_serverless_focused_contract.py`, covering every scenario D-037's
directive specified (2- and 3-fragment legitimate splits, exact-range and
overlapping-range real duplicates, scattered/non-contiguous real duplicates,
Composite-piece non-interference, fragment-id survival through render plan and QC
finding diagnostics, reconstructed-sequence-equals-frozen-plan, upload gating in
both directions, exact plan id/version/hash in delivery metadata, Boundary-cannot-
touch-semantic-membership). Full `tests/test_cutsell_*.py` CI glob green (1158
passed), CleanCutBench green, `compileall` clean, and the pre-existing 23/23 Human
Gold semantic-alignment fixture (`tests/test_video00_selection_lock.py` and
neighbors) confirmed unaffected -- this work never touches Selection.

## D-038 — Per-Idea semantic claim coverage (semantic fact preservation)

**Status: CANONICAL**

RAW 33423953391 confirmed D-037's physical-fragment/delivery-gate fix working correctly
(genuine physical QC findings correctly reported NEEDS_HUMAN_REVIEW rather than a
fabricated PASS), but exposed a separate, serious semantic regression: for idea
`tg_99d2c57b5472dc615a`, BestTake picked a cleaner-but-incomplete candidate over one
that stated a diagnosis-confirmation fact, dropping it from the winning realization.
CoverageLedger's existing `_lost_semantic_atoms` check (D-021/D-031) did not catch
this because it compares a discarded clip's vocabulary against the ENTIRE final KEEP
timeline's bag of words -- the lost fact's own words ("cáncer"/"tiroides"/"biopsia")
happened to also appear in unrelated selected clips elsewhere in the video (an earlier
screening discussion), which falsely satisfied whole-video vocabulary coverage.
Root cause, confirmed by reading `_lost_semantic_atoms` directly: `kept_content`/
`kept_critical` are built from the whole `draft.selected`, never scoped to one idea's
own winning realization.

    WHOLE-VIDEO WORD PRESENCE  !=  PER-IDEA CLAIM PRESERVATION

**1. New module `semantic_claims.py`**: general (no Video00 vocabulary), deterministic,
marker-based `classify_claim(sentence) -> (claim_type, importance, evidence)` covering
all 10 canonical claim types (ENTITY_RELATION, STATE_RESULT, DIAGNOSIS_IDENTIFICATION,
CAUSE_EFFECT, ACTION_EVENT, MEASUREMENT_QUANTITY, NEGATION, CORRECTION,
TEMPORAL_RELATION, UNIQUE_CONCLUSION) and 4 importance levels (CRITICAL/SUPPORTING/
CONTEXTUAL/REDUNDANT) -- importance is deliberately per-instance, not fixed per type
(a CAUSE_EFFECT clause is always SUPPORTING by design; a DIAGNOSIS_IDENTIFICATION or a
NEGATION is always CRITICAL). `extract_claims`/`dedupe_claims` build one `Claim` per
sentence (reusing `final_sibling_grouping`'s existing `_content`/`_negations`/
`_numbers` tokenizers and `semantic_atom_importance`'s existing marker vocabularies --
no parallel text-processing stack). `claim_coverage`/`claim_is_covered` score whether a
claim's content survives in a candidate's text, guarded against a real false-coverage
trap found while building this module's own CleanCutBench fixtures: a negation flip
("confirmed X" vs. "did NOT confirm X") shares almost every noun while asserting the
opposite, so a mismatch between the claim's and the candidate's own negation markers
caps coverage below the ambiguous floor rather than scoring it as near-total overlap.
`resolve_ambiguous_coverage` escalates only the genuinely ambiguous coverage band
(0.3-0.6) to a bounded `ClaimEquivalenceArbiter` (fails open to LOST, same posture as
`SemanticAtomImportanceArbiter`/`CausalOrderArbiter` -- no implementation wired in this
codebase, same honest-gap pattern).

**2. New module `claim_coverage_best_take.py`**: runs immediately after
`deterministic_best_take_authority`, before Final Story Coherence Validation. For every
genuine retry-family contest with exactly one current winner, checks whether that
winner covers every CRITICAL claim found across the GROUP'S OWN members (never
outside it). Bounded resolution, ambiguity fails open throughout (same posture as
`deterministic_best_take_authority.py`): (a) if exactly one other member covers every
critical claim, it becomes the new winner (KEEP/DISCARD-only move, D-019: no SWAP);
(b) if no single member does, but a time-compatible pair together covers everything,
both are kept as a narrow 2-piece composite (ordered by recording time); (c) anything
broader is left exactly as upstream decided, flagged in this module's own diagnostics
for observability -- the real backstop is item 3 below. The composite path is guarded
against a real regression found via full-chain testing: two members whose UNIQUE
contributions share a claim_type (e.g. both NEGATION, as in a paraphrased retry family
where each side negates the same fact in different words) are NOT composited -- that
shape is far more likely one idea's coarse-classifier-split paraphrase than genuinely
complementary facts, and forcing it into a composite silently defeated the existing,
correct arbiter-based retry-family collapse (`final_story_coherence_validation`'s own
residual-family resolution). Disjoint claim_types stay compositable.

**3. Per-idea backstop**: `final_story_coherence_validation._lost_critical_claims`
compares each retry-family group's own critical claims directly against ONLY that
group's own winning realization text -- never the whole-KEEP-timeline vocabulary
`_lost_semantic_atoms` uses -- so a claim can never be falsely satisfied merely because
its words also appear in a different, unrelated selected clip elsewhere in the video.
Runs even when `claim_coverage_best_take.py` already tried and could not safely
resolve a family (the 3+-way/no-compatible-pair case). A finding here always blocks
Freeze (`freeze_blocked`), by construction only ever CRITICAL-importance.

**4. Integration**: `CanonicalEditPlan` gained `lost_critical_claims` (populated from
StoryValidator's diagnostics, defaulted `()` so the one existing construction site and
any external payload deserializer stays valid unchanged); `_composite_piece_ids` also
recognizes `claim_coverage_best_take`'s own composites so an idea it resolved this way
correctly reports `is_composite: true`. `final_edit_reviewer.py` gained the
`CRITICAL_CLAIM_LOST` finding kind: always blocking (every row is CRITICAL by
construction, so there is no warning-split like `UNIQUE_FACT_LOST`'s), carrying
plan_id/plan_version/idea_id/claim_id/claim text/source clip/winning clips/coverage
evidence, routing to BestTakeResolver or CompositeResolver -- the reviewer never edits
membership itself. `process_universal_clean_cut_sources` threads an optional
`claim_equivalence_arbiter` (defaults `None`, same unwired-by-default pattern as every
other bounded arbiter here) to both the Best-Take override and StoryValidator.
`repair_loop.py`'s `_REPAIR_STRATEGIES` has no entry for `CRITICAL_CLAIM_LOST` by
design (same as `IDEA_COVERAGE_LOST`/`CONTRADICTION`): no safe automated repair exists,
so it correctly falls to NEEDS_HUMAN_REVIEW rather than an invented auto-fix.

**A second regression found and fixed via full-chain testing, not unit isolation**:
wiring `claim_coverage_best_take` into the real chain (all existing 37 CleanCutBench
fixtures, not just new ones) surfaced that `final_story_coherence_validation`'s own
residual-multi-select-group resolution had no awareness a group it saw with 2+ still-
selected members might already be a LEGITIMATE composite `claim_coverage_best_take`
just created, and used the same `semantic_equivalence_arbiter` that grouped them in
the first place to collapse the composite straight back down to one winner -- silently
undoing the fix. `_residual_multi_select_groups` now excludes any group_id present in
`diagnostics["claim_coverage_best_take"]["composites"]` (mirrors
`canonical_edit_plan._composite_piece_ids`'s own existing pattern for the same class of
trap). Caught by running the full existing eval-suite fixture set against the newly-
wired chain, per this directive's own explicit item 11 ("protect current semantic
success") -- not by the new fixtures alone.

**A pre-existing false-positive fixed in the same pass, found via direct classifier
testing before it ever reached a fixture**: `_IDENTIFICATION_COPULA_MARKERS`' bare "was
a"/"was an" (and Spanish "era un(a)"/"fue un(a)") entries matched as a substring PREFIX
of ordinary words with no word-boundary check ("it was ALREADY late" contained "was a"
literally) and, even fixed to a proper boundary, remained too generic on their own
("it was a good day" is not a diagnosis). Split into `_STRONG_IDENTIFICATION_MARKERS`
(unambiguous phrases -- "diagnosed with", "turned out to be", "se trataba de", etc. --
safe to treat as identity evidence standalone) and `_WEAK_COPULA_MARKERS` (only counted
as identity evidence when the same sentence ALSO carries explicit result-reporting
language, e.g. "the biopsy CONFIRMED it WAS A tumor") -- preserves every existing
positive case (importance stays CRITICAL either way; only the specific
DIAGNOSIS_IDENTIFICATION-vs-ENTITY_RELATION type label can shift) while eliminating the
false-positive class entirely.

**Protects, unchanged**: D-037 physical fragment identity, PostRenderWatchListenQC
behavior, the delivery gate, the Boundary repair loop, render provenance, and the
event-driven RAW-monitoring protocol -- none of that code was touched in this cycle.
Also confirmed unregressed: retry cleanup, no incomplete+complete duplicate,
hereditary contradiction resolution, sonography/pimples micro-ordering, composite
persistence, and the Human Gold semantic-alignment framework.

**Validation**: `semantic_claims.py` (35 tests), `claim_coverage_best_take.py` (15
tests), `_lost_critical_claims`/`CRITICAL_CLAIM_LOST` (9 tests), and 12 new
CleanCutBench fixtures mapping every category the directive named (cleaner take losing
a diagnosis claim must not win; a SUPPORTING cause/effect claim's loss must NOT force
an override -- the deliberate importance boundary, not an oversight; all-critical-
claims-covered wins despite worse performance; complementary claims require a
composite; a claim missing from its own idea still fails despite similar words
elsewhere in the video; same nouns present in the wrong winner produce no false
coverage; a CONTEXTUAL claim is safely omitted; a SUPPORTING claim is safely omitted
with the core idea intact; a critical correction is preserved; a critical claim split
across a genuinely independent continuation is left alone; a duplicate retry with no
unique claim is a safe plain discard; the bounded arbiter is consulted only for
ambiguous coverage and can change the outcome) -- reached through the REAL take-
grouping/idea-equivalence/take-judge/coherence chain, not hand-built drafts alone.
Full `tests/test_cutsell_*.py` CI glob green (1228 passed), `compileall` clean.

**RAW gate result (run 33432104336, commit 3f7122b)**: the target regression is fixed
-- the papillary-cancer diagnosis claim from RAW 33423953391's own failure now survives
(`papillary_cancer_preserved` passed the historical Human-Gold regression-QA check,
where it had failed on the pre-D-038 run). A second, real defect was found and fixed
from this run's own diagnostics, not from a fixture: `claim_coverage`'s negation-flip
guard (above) checked negation presence over the WHOLE candidate text, so a candidate
clip's or joined winning-realization's OTHER, unrelated sentence carrying a negation
("no creo ... son hereditarios" several sentences before an unrelated "solo un 5-10%
son de hereditario" claim in the very same clip) falsely capped coverage for a claim
whose own sentence was present, uncontradicted, verbatim -- producing a spurious
CRITICAL_CLAIM_LOST finding. Fixed by scoping the negation check to only the
sentence(s) that share a substantive (>=2, or the claim's full token count if fewer)
portion of the claim's own content tokens with the candidate, not one incidental
shared word (a first, too-broad attempt -- adding "son"/"fue"/"fueron"/"era"/"eran" to
`final_sibling_grouping._STOP` -- was reverted after it changed unrelated session-
boundary-reconciliation test outcomes; that stopword list is shared, foundational
infrastructure, and the correct fix belongs local to `semantic_claims.py`'s own
negation-scoping instead). Two new regression tests pin both directions: an unrelated
negation elsewhere no longer deflates coverage, and a genuine same-sentence negation
still caps it. This fix is committed to the branch but, per the "one RAW then STOP"
gate, has NOT been pushed -- pushing on this branch auto-triggers another paid RAW,
which needs explicit authorization first. The regression-QA's separate `pimples_micro_*`
failures on this same run were audited offline before any further RAW -- see D-039 below
for the confirmed root cause (not D-038 code, not simple non-determinism: a real,
general arbiter-confirmation gap in the pre-existing grouping stage) and its fix.

## D-039 — Offline audit of RAW 33432104336's pimples_micro_* regression + general grouping fix

**Status: CANONICAL**

Per explicit instruction, this audit ran BEFORE spending a second RAW and before touching
any production logic on assumption. Traced the full evidence path for the five clips
`take_judge_groups` merged into idea `tg_886d4543ce1fe7f21e` (winner
`clip_3d2f0f5c4d7a15cc7054`; discarded `clip_39cfe70af53d9dfe9cc5`,
`clip_8a345f3048dd41ffc5a2`, `clip_cbc4beb40b406dbdb068`, `clip_aa6d498957c527147344`)
against the immediately preceding RAW (33423953391, pre-D-038), using the CI job's own
printed diagnostics (`asr`/`attempt_reconstruction` stage_status, `semantic_idea_
equivalence`'s `candidate_pair_count`/`checked_pair_count`/`merged_pair_count`/`merges`,
`take_judge_groups`, `canonical_edit_plan.ideas`) -- no production code was modified
until this traced to a confirmed cause.

**ASR / attempt reconstruction**: `asr.segment_count` (57), `attempt_reconstruction.
attempt_count` (30), `.boundary_count` (29), `.input_candidate_count` (52), and
`.merged_fragment_count` (22) are IDENTICAL between the two runs -- the segmentation
structure did not change. `clip_id` is `stable_clip_id(source_asset_id, start, end,
text)` (`source_identity.py`), a SHA256 hash sensitive to millisecond-level timing and
exact wording; every clip_id differs between the two runs regardless, meaning the live
ASR pass produced slightly different word-level transcription and/or sub-millisecond
timing this run despite an identical overall structure -- ordinary GPU-inference
run-to-run jitter, not a different number or boundary of attempts.

**Grouping / candidate generation**: `semantic_idea_equivalence.candidate_pair_count`
was 52 this run vs. 51 the prior run -- proof the cross-group candidate-pair universe
itself differs (a direct consequence of the ASR/timing jitter above cascading into
`take_grouping_provider._cross_group_candidate_pairs`, since eligibility and lexical-
overlap scoring both read clip text/timing). `checked_pair_count` was 14 in BOTH runs --
confirming a hard, fixed per-request arbiter budget (`SemanticEquivalenceGatePolicy.
max_pairs_per_request`) that `_rank_candidate_pairs` fills highest-priority-first (an
earlier, already-documented fix for a different pair-starvation failure mode -- see
that function's own docstring). With a different, larger candidate pool this run, the
ranking spent its fixed 14-slot budget on a different subset of pairs than last run.

**Semantic-equivalence arbiter (Gemini `gemini-3.5-flash-lite`)**: of the 7 pairs it
confirmed this run (vs. 5 last run, and NONE of the prior run's 5 touched pimples
content at all), 3 involved the five pimples-related clips. Their own stated reasons
name the actual defect: `clip_8a345f3048dd41ffc5a2`'s text is
"Otro síntoma era que me salían espinillas ... Me salía por temporadas." -- matching
`benchmarks/video00_regression_qa.json`'s own `pimples_later_winner_present` fixture
almost verbatim, and opening with "Otro síntoma" ("ANOTHER symptom"), the speaker's own
explicit signal that this is a DIFFERENT, additional point, not a restatement. The
arbiter nonetheless confirmed it "same idea" as `clip_39cfe70af53d9dfe9cc5` (confidence
0.95, matching the fixture's own forbidden `pimples_bad_monolith_absent` text) and as
`clip_cbc4beb40b406dbdb068` (confidence 0.9, matching `pimples_micro_1/2/3_present`'s
three fragments concatenated) purely on topical similarity ("same hormonal acne
symptoms" / "acne interpreted as an allergy rash"), missing that discourse marker
entirely. `clip_3d2f0f5c4d7a15cc7054` (the eventual winner) also merged with
`clip_cbc4beb40b406dbdb068` ("both discuss experiencing seasonal acne outbreaks").
`clip_aa6d498957c527147344` is not in any printed arbiter merge -- it joined via the
baseline (pre-arbiter) lexical/temporal tier, `safe_group_takes`, itself.

**Root cause classification: G (combination)** -- B (ASR/timing jitter, same structure,
different exact clip_ids) -> C (different candidate-pair pool, +1 pair) -> D (fixed
14-pair budget now spends its ranked slots on a different subset) -> **E (the arbiter's
own verdict on the newly-reachable pair is a real misjudgment)**, not a code-path change
in D-038 (root cause A is ruled out: neither `reconcile_semantic_idea_equivalence` nor
`_resolve_residual_family`, the two mechanisms that actually merged and then collapsed
this family, were touched by D-038 -- `claim_coverage_best_take` never even acted on
this group, since all 5 members were still selected when it ran, `current_winners != 1`,
its own explicit "not this module's job" branch). This is genuine E, not mere run-to-run
noise dressed up as inevitable: the SAME arbiter, given this exact pair again, would very
likely misjudge it the same way -- it is a real, general gap in the grouping stage's
defenses, not an unrepeatable fluke, and D-025's own docstring in `take_grouping_
provider.py` already documents an earlier RAW (33366538992) hitting this identical
"pimples/otro sintoma" shape from a different angle (there, CompositeResolver had
already accepted these as composite pieces and `reconcile_semantic_idea_equivalence`
re-merged them anyway; `protected_ids` fixed that specific path but only once
CompositeResolver has already acted -- it does not help when, as here, nothing upstream
ever flagged the clips as composite pieces in the first place).

**What the five clips actually are**: reading the three visible discarded texts against
`benchmarks/video00_regression_qa.json`'s own fixture definitions, they are **mixed
ideas incorrectly collapsed**, not one retry family and not a valid multi-piece
composite/continuation of ONE idea: `clip_cbc4beb40b406dbdb068` is the concatenated
micro-fragment idea (`pimples_micro_1/2/3`), `clip_39cfe70af53d9dfe9cc5` is a verbose,
correctly-discardable retry of that same idea (`pimples_bad_monolith_absent`), and
`clip_8a345f3048dd41ffc5a2` is a DIFFERENT, LATER beat the speaker explicitly marks as
additional (`pimples_later_winner_present`) that the Human Gold keeps as its own
delivery, not a restatement to be collapsed into the first idea's retry contest.

**General fix (grouping capability, not a D-038-layer patch)**: `take_grouping_
provider.reconcile_semantic_idea_equivalence` gained a deterministic override,
independent of and evaluated after the arbiter's own verdict: when exactly one side of
an arbiter-confirmed "same idea" pair carries a general "this is a new/additional item"
discourse marker (`_DISTINCT_ADDITION_MARKERS` -- "otro sintoma", "otra cosa", "another
symptom", "an additional", etc.; general connector vocabulary, no Video00-specific
phrase, deliberately excluding "otra vez"/"again"-style repetition markers) and the
other side does not, the merge is blocked and recorded in a new
`distinct_addition_blocked` diagnostics list -- the speaker's own explicit signal of
distinctness outranks a same-idea verdict from a topical-similarity judgment. Both sides
carrying the marker (two attempts at introducing the same transition) is not blocked --
only an IMBALANCE is evidence of an actual difference. This complements, and does not
replace, the existing `protected_ids` defense (which requires CompositeResolver to have
already acted); this guard fires independently, before that ever needs to happen.

**Validation**: 3 new unit tests in `tests/test_cutsell_semantic_idea_equivalence_
grouping.py` (blocks the imbalanced-marker case reproducing this exact audit finding;
still merges an ordinary marker-free paraphrase; still merges when both sides carry the
marker) plus one new CleanCutBench fixture (`test_distinct_addition_marker_prevents_a_
real_chain_false_merge`) reached through the real `safe_group_takes` ->
`reconcile_semantic_idea_equivalence` chain, proving the guard fires with a real
arbiter call and both deliveries survive independently rather than being collapsed to
one winner. Full `tests/test_cutsell_*.py` CI glob green, `compileall` clean. Pushed as
`5c264af`/`796b0dc` after explicit authorization and preconditions re-verification
(no RAW queued/in-progress, PR #25 open/draft/unmerged, `main` untouched, teardown
step unchanged).

**Addendum -- the RAW trigger itself had a coverage gap.** Neither commit above
auto-triggered a RAW: the workflow's push `paths:` filter did not include `semantic_
claims.py` (D-038's own fix file) or `take_grouping_provider.py` (this decision's own
fix file) -- a real, general gap, not specific to these two files. Audited the full
D-021 canonical component map against the filter and found it missing most of
IdeaClusterer/SemanticArbiter, ALL of StoryValidator/CanonicalEditPlan/
FinalEditReviewer/the repair loop/causal-order validator, SelectionFreeze, most of
BoundaryEngine, all of Renderer/live-render-QC, the D-038 claim-coverage layer, and the
shared `contracts.py`/`source_identity.py` identity model underneath all of it -- every
one of those can materially change Video00's semantic or physical output. Extended the
`paths:` filter to cover the complete active canonical set (grouped and commented by
component, mirroring D-021's own table), added `tests/test_video00_raw_trigger_
coverage.py` (pins the required file set, asserts the workflow's YAML covers it, and
that every required path actually exists on disk) so a future canonical-map change and
a workflow-filter change can no longer silently drift apart, and left the already-
deprecated `unified_selection_reasoner.py`/`unified_selection_google.py` entries in
place (rollback target, not removed) rather than pruning anything. Deliberately
excludes docs-only files, tests-only files, and the dormant Sales/TikTok Shop extension
points. No editorial logic touched. Full `tests/test_cutsell_*.py` CI glob green,
`compileall` clean, YAML validated.

RAW `33448261223` (commit `ddfdb6a`) then ran: the cleanest Selection of the session
(Human Gold regression-QA 18/18 checks passed, `selection_locked: true`) but blocked at
Freeze by one `CRITICAL_CLAIM_LOST` finding -- see D-040 below for the root cause found
by cross-checking that finding against the Human Gold alignment evidence.

## D-040 — Claim granularity: core proposition vs. supporting clause

**Status: CANONICAL**

RAW `33448261223`'s one blocker, root-caused: `final_story_coherence_validation.
_lost_critical_claims` flagged a NEGATION claim at `coverage_against_winning_
realization: 0.5` (ambiguous, no arbiter, fails open to LOST). Cross-checked against
the Human Gold alignment (QA evidence, never runtime logic) for that exact span:
`content_coverage: 1.0`, `relation: EXACT` -- the winning clip's text matches the Gold
reference verbatim. The Gold editor's own authoritative edit made and endorses the
same content the finding called "lost." Root cause: `extract_claims` (D-038) treated
an entire multi-clause sentence as one atomic claim --
"Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides, pues porque cada
año que me hacía mínimo dos estados." bundles a CORE negation ("we never thought to
check") with a merely-SUPPORTING reason clause ("...because ... two exams a year") --
so a winning realization that kept the core (matching Gold) but phrased the reason
differently scored as if the whole sentence, core included, were lost.

**Fix**: `extract_claims` now splits each sentence into clauses BEFORE classifying
(`_split_into_clauses`), on the same general connector vocabulary `classify_claim`'s
own CAUSE_EFFECT/TEMPORAL_RELATION rules already draw on
(porque/because, pero/but/aunque/although, cuando/when, después de/after,
entonces/so, lo que/lo cual/which -- bare Spanish "que" deliberately excluded, too
generic a subordinator to split on safely), plus a new `_CONTRASTIVE_MARKERS` set
(pero/but/aunque/although/sin embargo/however/though). A split is accepted only when
BOTH sides clear the existing >=2-content-token floor, and splitting recurses into the
remainder so a chain of connectors yields every piece. **No new importance axis was
needed**: each clause is classified independently by the SAME deterministic
`classify_claim` already used for whole sentences, so a clause that itself carries a
real marker (negation/reporting/correction/etc.) keeps CRITICAL on its own, and a
clause introduced by a mere connector with no such marker of its own falls through to
`classify_claim`'s existing SUPPORTING/CONTEXTUAL fallbacks -- exactly the CORE vs.
SUPPORTING vs. CONTEXTUAL distinction the directive asked for, without inventing a
second severity scale to keep in sync with the first. Coverage is claim-LOCAL by
construction once each clause is its own `Claim`: a winner can legitimately score 1.0
on the core clause and 0.0 on a dropped supporting one and still be valid, because
only the core clause is ever CRITICAL and only CRITICAL claims are checked for
blocking loss.

**Bounded `ClauseRoleArbiter` (new, honest-gap, unwired by default)**: for the one
genuinely ambiguous case -- `classify_claim`'s weakest, marker-less `general_statement`
fallback -- an optional arbiter can be asked "would removing this clause materially
change the audience-facing factual meaning of this Idea?", returning CORE_CRITICAL/
SUPPORTING/CONTEXTUAL/UNCERTAIN. With no arbiter configured (every RAW to date), the
deterministic fallback is left exactly as `classify_claim` decided -- forcing a blanket
escalation for every marker-less clause in a video would reintroduce the very
over-blocking this fix exists to remove. Once wired: a confirmed CORE_CRITICAL upgrades
the clause; an explicit UNCERTAIN verdict or an arbiter exception also upgrades to
CRITICAL ("WHEN UNCERTAIN, KEEP") rather than trusting the weak fallback silently.
Threaded through `claim_coverage_best_take.apply_claim_coverage_best_take`,
`final_story_coherence_validation.apply_final_story_coherence_validation`, and
`universal_clean_cut.process_universal_clean_cut_sources` alongside the existing
`claim_equivalence_arbiter`, defaulting `None` throughout.

**Explicitly not touched**: grouping (`take_grouping_provider.py`), BestTake's own
override mechanics, CompositeResolver, the D-039 distinct-addition guard, physical
fragment identity, the delivery gate, monitoring behavior, and the Human Gold alignment
harness itself (used only as QA evidence to diagnose the false positive, never encoded
into runtime logic).

**Validation**: 15 new unit tests in `tests/test_cutsell_semantic_claims.py` -- clause
splitting itself (no-connector sentences unsplit; a connector with too-thin a remainder
left unsplit) plus the directive's own 10 false-positive-protection cases (core
preserved/supporting dropped -> not a critical loss; core dropped -> still blocks even
with a supporting clause preserved; a cause/effect-INTRODUCED clause that is itself
critical -> still blocks if dropped; a redundant/explanatory clause -> never tracked as
critical; an incidental temporal clause -> CONTEXTUAL, never critical; a critical
numeric-measurement clause -> still blocks if dropped, survives splitting; two
independently critical claims in one sentence -> both tracked, each independently
checkable; a subordinate correction clause -> still critical, splitting never demotes
real critical content; a paraphrased core claim -> still counts as covered; same
vocabulary/different proposition at clause granularity -> still no false coverage) --
plus 5 tests for `resolve_ambiguous_clause_role`'s own escalation contract. One new
CleanCutBench fixture reproduces RAW `33448261223`'s exact false positive through the
real take-grouping/idea-equivalence/take-judge/coherence chain and confirms it no
longer blocks Freeze. One pre-existing test (`test_dedupe_claims_keeps_distinct_
propositions`) needed its fixture text updated -- "I felt tired because I skipped
breakfast." now correctly splits into two claims, which is the fix working as intended,
not a regression; a new test pins that exact split. Full `tests/test_cutsell_*.py` CI
glob green, `compileall` clean.

## D-041 — RunPod health/capacity reliability hardening (infrastructure only)

**Status: CANONICAL**

RAW `33453836301` (the D-040 gate run, commit `19f6612`) failed at the "CUDA health"
step before any application code -- including the D-040 fix -- ever ran. Diagnosis
(`mcp__github__get_job_logs` on job `99689435894`): the health job was submitted at
`00:18:15Z`, accepted by RunPod (`id=f89a0a98-...-u1`), and polled every 5s against one
undifferentiated 1200s (20-minute) deadline. It never left `IN_QUEUE` -- never
`COMPLETED`/`FAILED`/`TIMED_OUT`/`CANCELLED` -- for the entire window; at `00:38:19Z` the
loop simply timed out and printed whatever the last poll happened to show. No worker on
endpoint `xxu7autt8mv2rn` ever picked the job up. Because the health gate never passed,
"Submit original six-minute Video00" and "Wait for unified Selection result" were
correctly SKIPPED. This is not evidence against D-040: nothing downstream of health ever
ran. It is a distinct RunPod-infra failure shape from the earlier `33414001062` incident
this session (a fast, explicit `409` on submission) -- this one was accepted into queue
and then starved, a capacity/provisioning-style stall rather than a rejection.

**Fix (`runpod_orchestration.py`, new, infrastructure/orchestration only -- no
`cutsell_worker` editorial code touched)**: replaces the workflow's old inline-bash
"roll endpoint" + one-blind-20-minute-wait "CUDA health" pair with a small, fully
unit-testable state machine (`Transport` protocol injected for both production
(`UrllibTransport`, stdlib `urllib.request` only -- no new pip dependency, matching this
workflow's existing Python steps) and tests (a scripted fake, no network/GPU/credentials
required)):

- `wait_for_endpoint_ready` -- polls `GET /v1/endpoints/{id}` until it reflects the roll
  just PATCHed (matching `templateId` + `workersMax`) or a bounded readiness timeout
  elapses. A `409` on this read (endpoint still mid-transition) is classified
  `ENDPOINT_TRANSITION_RACE`; a stale config that never converges within the timeout
  (no `409`, just never matches) is classified `CAPACITY_UNAVAILABLE`. Records
  `template_id`/`workers_min`/`workers_max`/`gpu_type`/elapsed readiness time.
- `submit_and_poll_health` -- submits the health job, then tracks time spent
  specifically in `IN_QUEUE` against a bounded stall threshold (`queue_stall_s`,
  default 300s) instead of one undifferentiated deadline. Still `IN_QUEUE` at the stall
  threshold on a first attempt -> `WORKER_PROVISIONING_STALLED` (treated as possibly
  transient, eligible for one retry); still stuck on a retry attempt ->
  `CAPACITY_UNAVAILABLE` (persistent, no further retries -- see honesty note below). A
  genuine terminal RunPod status is never reclassified as a queue problem even if slow
  to arrive: `COMPLETED` with `output.ok`/`output.cuda_available` not both `true`, or any
  of `FAILED`/`TIMED_OUT`/`CANCELLED`, is `HEALTH_APP_FAILURE` -- a real application/CUDA
  problem, not a flake. `HEALTH_PASSED` only on `COMPLETED` with both flags `true`.
- `run_with_bounded_retry` -- retries only infrastructure-class failures
  (`ENDPOINT_TRANSITION_RACE`/`CAPACITY_UNAVAILABLE`/`WORKER_PROVISIONING_STALLED`/
  `RUNPOD_API_ERROR`), exactly once (`max_infra_retries=1`, a hard ceiling -- never an
  unbounded loop), re-rolling the endpoint and backing off before the retry.
  `HEALTH_APP_FAILURE` is never retried. `cancel_job_if_active` tears down a stalled
  job's RunPod-side state (cancels if still `IN_QUEUE`/`IN_PROGRESS`) before that retry.
- Exit code contract: `main()` returns `0` only when `OrchestrationResult.passed` is
  `True` (a genuine `HEALTH_PASSED`), `1` otherwise -- unchanged from the old step's
  contract, so the workflow's existing "Submit original six-minute Video00" step (no
  `if: always()`) is still automatically skipped by GitHub Actions on any failure
  classification, exactly as it correctly was for RAW `33453836301`.
- Observability (structured `[runpod_orchestration] <event> {...}` log lines, printed
  unconditionally): `endpoint_roll_started_at`, `endpoint_ready`/`endpoint_not_ready`
  (with elapsed readiness time + classification), `health_submitted`, `health_status`
  (per-poll, with `job_id`/`status`/elapsed/`worker_id` when RunPod reports one),
  `health_queue_stalled` (with `time_in_queue_s` + classification),
  `infra_retry_backoff`, `job_cancelled`. `main()` also writes `HEALTH_JOB_ID` (the last
  attempt's) and `RUNPOD_INFRA_CLASSIFICATION` to `GITHUB_ENV` for the unchanged teardown
  step and for any future CI-log grep.

**Honesty note on the `WORKER_PROVISIONING_STALLED` vs. `CAPACITY_UNAVAILABLE` split**:
RunPod's plain endpoint/job-status APIs do not expose a worker-count or scheduler-queue-
depth field this session could verify, so the two labels are NOT read off a real RunPod
signal -- they are a policy distinction keyed on retry-attempt-number only (first stall =
labeled as possibly-transient and retried once; still stuck on that one retry = labeled
persistent and given up on). This is stated plainly rather than implying a deeper signal
that does not exist, per the same "never accept silent provider fallback" / "preserve
observability" engineering rules that govern the rest of this file.

**Item 4 (GPU/capacity fallback audit) -- explicitly incomplete, and said so rather than
guessed**: this session has no live RunPod API credentials (`RUNPOD_API_KEY` is a GitHub
Actions secret, not available to local development), and the repository's own code
contains no GPU-type pin to read (the endpoint patch payload never sets `gpuIds`; whatever
GPU pool endpoint `xxu7autt8mv2rn` uses was configured outside this codebase, in RunPod's
own dashboard). No prior CI run ever printed the endpoint's GPU type either -- until this
fix, nothing in the pipeline captured it. Rather than fabricate a GPU-class comparison or
a cost/performance estimate this session cannot verify, this fix adds the *capture*
(`wait_for_endpoint_ready` now reads back and logs `gpuIds`/`gpuType` when RunPod's
response includes it) so the next real RAW's CI log answers this question with actual
data instead of a guess. **No GPU/capacity/cost configuration was changed.** A genuine
fallback-GPU audit (current GPU vs. compatible alternatives vs. cost/perf/CUDA-stack
compatibility) is deferred until a run has actually surfaced the current GPU type via
this new observability, or until someone with live RunPod dashboard/API access supplies
it directly.

**Explicitly not touched**: `cutsell_worker/` editorial code (grouping, BestTake,
CompositeResolver, StoryValidator, D-038/D-039/D-040's fixes, physical fragment identity,
the delivery gate, `PostRenderWatchListenQC`, Boundary repair), the RAW-completion
event-driven monitoring behavior this session uses to watch a run, Sales Funnel/TikTok
Shop styling, SWAP, `main`/PR merge state, production/TestFlight. The workflow's
push-`paths:` filter (D-039/D-039-addendum) is unchanged for every canonical Clean Cut
V1 file; only the workflow file's own self-reference and (deliberately never added)
`runpod_orchestration.py` are excluded from it -- see the item below.

**Trigger-path side effect (intentional)**: the workflow file itself was previously in
its own `push.paths` filter, meaning any edit to this workflow -- including a pure
CI-orchestration fix like this one -- would have auto-fired a paid RAW on push. Removed:
neither the workflow file nor `runpod_orchestration.py` can change Video00's semantic or
physical output, so neither should silently burn a paid RAW on every CI-orchestration
edit (the same "keep the trigger narrow" goal the D-039 addendum stated for the opposite
direction -- ensuring editorial files DO trigger). `tests/test_video00_raw_trigger_
coverage.py::test_trigger_set_excludes_ci_orchestration_infra` pins the exclusion.
Consequence: this fix's own push does NOT auto-trigger a RAW; the one controlled retry
against the existing D-040 head is fired deliberately via `workflow_dispatch`.

**Validation**: 18 new unit tests in `tests/test_runpod_orchestration.py` covering all 9
required scenarios (endpoint ready normally; `409` recovers vs. persists past timeout;
health accepted then stalls `IN_QUEUE` and fails fast; worker eventually starts and
health completes; a real terminal app failure is never reclassified as a queue problem;
capacity stall on a retry attempt; bounded retry succeeds; bounded retry exhausted after
the ceiling with no third attempt; `HEALTH_APP_FAILURE` never retried; teardown cancels a
stalled job and a full attempt-flow test proves teardown happens before a successful
retry; the `passed` contract that gates Video00 submission) -- all against a scripted
fake `Transport` and a fake clock, no network/GPU/credentials. One new trigger-coverage
test pins the path-filter exclusion above. Full `tests/test_cutsell_*.py` CI glob plus
the two new test files green, `compileall` clean.

### D-041 follow-up — GPU fallback audit: endpoint is already a 4-GPU pool, not narrowly pinned

**Status: CANONICAL**

After D-041 landed, RAW `33457835750` (the controlled retry) proved the hardened
orchestration layer itself works correctly end to end -- readiness detection, IN_QUEUE
stall tracking, the one bounded infra retry, teardown-before-retry, and failure
classification all behaved exactly as designed (`WORKER_PROVISIONING_STALLED` on the
first attempt, `CAPACITY_UNAVAILABLE` on the retry, `~626.5s` total vs. the old blind
`1200s`) -- but the underlying RunPod worker placement still failed: two independent
health jobs, on two independently fresh endpoint rolls, each sat `IN_QUEUE` a full 5
minutes with no worker ever assigned.

Before proposing any GPU fallback policy change, the live endpoint/template
configuration was read directly rather than guessed, via a new read-only tool
(`runpod_endpoint_inspect.py` + `.github/workflows/runpod-endpoint-inspect.yml`,
`workflow_dispatch`-only with a self-scoped `push` trigger on just its own two files so
it can register for dispatch and stay runnable -- GETs only, no GPU job, no Video00, zero
cost). Real, live result for endpoint `xxu7autt8mv2rn`:

```
gpuTypeIds:      ["NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA A40", "NVIDIA RTX A6000"]
gpuCount:        1
minCudaVersion:  "12.0"
flashboot:       true
workersMin/Max:  0 / 0  (at rest -- teardown between runs sets this)
workersStandby:  1
scalerType:      QUEUE_DELAY, scalerValue: 2, idleTimeout: 5, executionTimeoutMs: 1800000
networkVolumeId: "" (none)
templateId:      07g9dovc17 ("EditDNA-Worker-2")
  imageName:         madiator2011/better-pytorch:cuda12.4-torch2.6.0
  containerDiskInGb: 80, volumeInGb: 60, volumeMountPath: /workspace
```

No `locations`/`dataCenterIds` field is present at all -- this endpoint has no region/
datacenter restriction configured; it is free to place a worker anywhere RunPod has
matching capacity.

**Finding: the premise that the endpoint might be "pinned too narrowly" is false.** The
repository's own PATCH payload (`{templateId, workersMin, workersMax, scalerType,
scalerValue, idleTimeout, executionTimeoutMs}`) never sets `gpuTypeIds` on either roll
this session performed, so both of RAW `33457835750`'s stalled health jobs were already
eligible for placement on any of these four GPU classes -- a reasonably broad,
non-exotic pool of consumer/workstation cards (24-48GB VRAM), all CUDA-12.0+-compatible
with the template's CUDA 12.4 / torch 2.6.0 stack, all with VRAM far exceeding this
workload's footprint (an 80GB container disk, no large-model VRAM requirement implied
anywhere in the repo). There is no narrower single-GPU-class pin to broaden, so items 2/3
of the follow-up directive (build a compatibility matrix, propose a priority-ordered
fallback pool) do not apply -- there is nothing to widen. Building a compatibility
matrix or proposing a fallback priority order for a pool that is already 4 GPU classes
wide, and inventing GPU-class comparisons for classes not actually in play, would not be
grounded in evidence and was avoided rather than fabricated.

**No GPU/cost/capacity configuration was changed** -- correctly so per item 6's own
gate: this is exactly the "capacity still fails even with a safe compatible fallback
pool" scenario the directive itself names as the stop condition. A pool already spanning
four distinct GPU classes across (per the absence of any location restriction) every
RunPod datacenter, still producing two consecutive 5-minute placement failures, points
at an account-level or provider-side capacity/quota/billing issue rather than a
GPU-policy misconfiguration this session can fix from code. Per instruction: **STOPPING
and recommending RunPod support/account-level investigation rather than another RAW or
health-only retry.** `workersStandby: 1` is worth a human's attention too -- its
interaction with this workflow's own transient `workersMax` 0/1 cycling was not
something this session had grounds to evaluate confidently; flagged rather than guessed
at.

**Explicitly not touched**: any `cutsell_worker/` editorial code, D-040, D-041's
orchestration state machine itself, grouping, physical fragment identity, delivery gate,
monitoring behavior, Sales Funnel/TikTok Shop styling, SWAP, `main`/PR merge state,
production/TestFlight, and the live RunPod endpoint's actual GPU policy (read, never
written).

**Validation**: 4 unit tests in `tests/test_runpod_endpoint_inspect.py` pin the
allowlist-filter's only two real risks (a template's `env` dict, which carries real
secrets, must never reach stdout even though nothing asked for it to be excluded; an
unrecognized future field is dropped by default rather than guessed at). The tool itself
was run three times live via its self-registering push trigger (zero GPU cost, seconds
each), iterating from an initial too-narrow allowlist (guessed `gpuIds`, absent on this
account's API) to the real field names (`gpuTypeIds` et al.) via a safe values-never-shown
key-name-only diagnostic step -- the actual live data above is the third run's real
output, not inferred. Full `tests/test_cutsell_*.py` CI glob plus all new test files
green, `compileall` clean.

## D-042 — CutSell QA GPU execution fallback: RunPod Pod On-Demand automation (infrastructure only)

**Status: CANONICAL (code built + tested; first live health-only Pod test not yet run)**

D-041's own escalation concluded with two independent, hours-apart, `workersMin=1`
warm-worker isolation tests both confirming a **persistent** RunPod Serverless
worker-provisioning/account-placement failure (see D-041 and
`ops/runpod-support-report-gpu-allowlist-discrepancy.md`), now held for RunPod Support's
response. Rather than block all CutSell QA GPU work on that response, this decision adds
a second, interchangeable execution backend so paid QA benchmarking can continue on
RunPod Pods (persistent GPU instances) while Serverless recovery is pending -- **without
replacing, deleting, or degrading Serverless**, which remains the production backend and
is reactivated the moment RunPod resolves the provisioning issue.

**Architecture**: `gpu_execution_provider.py` defines the `GPUExecutionProvider`
Protocol (`health_check() -> HealthCheckResult`, `teardown() -> None`) plus
`RunPodServerlessExecutionProvider`, a thin wrapper around D-041's already-tested
`wait_for_endpoint_ready`/`submit_and_poll_health` -- not a reimplementation.
`runpod_pod_provider.py` adds `RunPodPodExecutionProvider`, the new RunPod Pod on-demand
implementation, using the exact same injectable `Transport` pattern
`runpod_orchestration.py` established (fakes in tests, `UrllibTransport` in production).
Both providers return the identical `HealthCheckResult` dataclass shape
(`execution_provider`, `passed`, `classification`, `elapsed_s`, `detail`) -- output-format
parity, never a forked result schema.

**One canonical benchmark contract, preserved by construction**: `cutsell_worker/
serverless_handler.py`'s op-dispatch table (`health`/`focused`/`locked_selection`) was
already a set of plain dict-in/dict-out functions with no RunPod object touched inside --
the ONLY transport-specific code in that file was `handler()`'s job-envelope unwrapping.
This decision extracts that dispatch into `run_op(op, payload) -> dict`, and `handler()`
now just unwraps `job["input"]` and calls it. `cutsell_worker/pod_job_server.py` (a
stdlib-only `http.server`, matching this repo's no-new-dependency policy for
orchestration/transport code) is the Pod-side counterpart: it exposes `GET /health` and
`POST /run` and calls that exact same `run_op`. There is one canonical CutSell job
runner; the two backends only differ in how a job reaches it (RunPod's async
Serverless job-queue envelope vs. a direct synchronous HTTP call to a running Pod).

**GPU pool + cost safety**: `runpod_pod_provider.APPROVED_POD_GPU_TYPE_IDS` is the
conceptual preference order RTX 4090 -> A40 -> RTX A6000 -> L4 (current image: PyTorch
2.6 / CUDA 12.4). `EXCLUDED_POD_GPU_TYPE_IDS` explicitly and permanently excludes RTX PRO
6000 Blackwell Server Edition (the exact D-041 GPU-fallback-audit incompatibility) and
H100/H200/A100, regardless of price or catalog-reported availability -- assertions in
`rank_gpu_candidates` and dedicated tests guard against either ever entering the approved
pool. GPU availability is never assumed: `fetch_pod_gpu_catalog` reads RunPod's live
`/v1/gpuTypes` catalog for price/availability (failing safely to an empty catalog, never
an exception, on any read failure), and actual Pod creation is the authoritative
availability signal -- `RunPodPodExecutionProvider._select_and_create_fresh` attempts
creation in ranked, under-ceiling order and only advances to the next candidate on a
capacity-shaped error (`looks_like_capacity_error`); a non-capacity error (auth, quota,
malformed request) is fatal and is never retried across GPU types. A configurable QA
hourly cost ceiling (`DEFAULT_COST_CEILING_USD_PER_HR = 1.50`, overridable via the new
workflow's `qa_pod_cost_ceiling_usd_per_hr` input) is enforced before any creation
attempt; if every available approved GPU is priced above it, the provider reports
`POD_COST_CEILING_EXCEEDED` and provisions nothing.

**Lifecycle**: `RunPodPodExecutionProvider.ensure_ready()` implements TEST REQUESTED ->
inspect existing Pod (by an optional caller-supplied `existing_pod_id`) -> reuse if
already `RUNNING` with a matching image -> one bounded restart attempt if `STOPPED`/
`EXITED` -> on restart failure/timeout or any other unexpected state (wrong image,
`ERROR`, disappeared), delete the stale Pod and create exactly one fresh Pod via the GPU
search above -- never an unbounded restart/recreate loop (pinned by
`test_recreation_is_bounded_not_looped`). `teardown()` always STOPs (never deletes --
reuse-first) the Pod this instance is holding, retries the STOP call once on failure, and
is a safe no-op if no Pod was ever acquired.

**Concurrency**: rather than invent a second, divergent distributed-lock primitive, the
new workflow reuses the exact mechanism the Serverless RAW workflow already relies on for
the same purpose -- a GitHub Actions `concurrency: group:` block (`cutsell-video00-pod-qa`,
`cancel-in-progress: false` so a running test finishes and stops its Pod rather than being
killed mid-flight).

**Workflow**: `.github/workflows/cutsell-video00-pod-raw.yml` is a new, dedicated,
`workflow_dispatch`-only workflow (no `push:` trigger of any kind -- ordinary commits to
`cutsell/mobile-v1-clean` can never provision an on-demand Pod). It builds the same
`Dockerfile.cutsell.serverless` image RunPod Serverless already uses, runs
`runpod_pod_health_gate.py` (provision/reuse -> HEALTH ONLY -> guaranteed STOP in
`finally`), uploads the health summary as an artifact, and issues an independent
belt-and-suspenders force-STOP as a second safety net in case the Python process itself
never reached its `finally` (e.g. the runner was killed) -- mirroring the Serverless RAW
workflow's own redundant `if: always()` teardown step. Pod reuse across manual runs is by
an `existing_pod_id` workflow input (the prior run's summary names the Pod id to pass
back in) rather than auto-persisted state -- adequate for QA's low-frequency, human-
initiated usage; not claimed to be more than that.

**Selector**: `gpu_execution_provider.EXECUTION_BACKEND_SERVERLESS`/
`EXECUTION_BACKEND_POD` (`"serverless"`/`"pod"`) is the two-and-only-two valid backend
selector. Serverless remains the default/production backend everywhere except this new,
manually-dispatched QA workflow, which is `pod`-only for now, by construction (no
`execution_backend` input exists yet -- there is exactly one workflow for each backend).
When RunPod Support resolves the Serverless provisioning issue, Serverless is used again
for QA without reverting any of this work.

**Testing**: 26 tests in `tests/test_runpod_pod_provider.py` (reuse, restart-succeeds,
restart-unavailable-recreate, stale-wrong-image-recreate, fresh-create, the full
RTX4090->A40->A6000->L4 fallback chain, no-compatible-GPU, cost-ceiling rejection
(both the "cheaper approved GPU chosen over an expensive one" and "all approved GPUs over
ceiling" shapes), Blackwell-never-attempted-even-if-cheapest, health pass/fail, health
never attempted when the lifecycle itself fails, one-bounded-recreation-succeeds,
recreation-is-bounded-not-looped, STOP-after-success/failure/exception via `finally`,
STOP-API-failure-retried-once, teardown-is-a-noop-with-no-pod, GPU-catalog parsing/
failure-safety, and non-capacity errors never retried across GPU types), 4 tests in
`tests/test_gpu_execution_provider.py` (backend-selector constants, Serverless-wrapper
call-shape parity with calling the D-041 primitives directly, Serverless teardown is a
documented no-op, and `HealthCheckResult` schema identity across both providers), 9 tests
in `tests/test_cutsell_pod_job_server.py` (real HTTP against the stdlib server on an
ephemeral port: health/run dispatch, default-op, 404s, invalid/non-object JSON bodies,
and an exception inside `run_op` becoming a 500 rather than crashing the server or
leaving it unable to answer the next request), 4 tests in `tests/
test_runpod_pod_health_gate.py` (pass/fail/raise all still call `teardown()`,
`existing_pod_id` env var threaded through), and 5 tests in `tests/
test_cutsell_serverless_run_op_dispatch.py` locking that `handler()` delegates to
`run_op()` rather than re-implementing dispatch. Full `tests/test_cutsell_*`/D-041/D-042
suite green; `compileall` clean. (Two pre-existing, unrelated failures --
`tests/test_semantic_stitch.py`'s module-level collection error and one
`tests/test_hybrid_story_guard_incomplete_retry.py` assertion -- were confirmed identical
on the unmodified base commit before this work began and are therefore out of this
decision's scope; not touched, per the standing "infrastructure only, no editorial
changes" constraint.)

**Not yet done / explicitly gated**: this decision covers the provider abstraction,
lifecycle automation, GPU search, cost safety, and the health-only workflow --
**no real GPU or Pod has been touched yet.** The full canonical Video00 benchmark on a
Pod (the same D-021 pipeline stages Serverless already runs, via the same `run_op`) is
deliberately not wired into the new workflow yet and stays gated on: (1) the first live
health-only Pod test passing, and (2) separate, explicit authorization for the first full
Pod-backed Video00 benchmark. Nothing in this decision changes D-040, any
`cutsell_worker/` editorial module, `main`, or PR #25's draft/open state.

**Live health-only Pod test checkpoint (2026-09-01, on `feature/runpod-pod-on-demand`,
not `cutsell/mobile-v1-clean`)**: five real `workflow_dispatch` runs of
`cutsell-video00-pod-raw.yml`, each root-caused from live evidence and fixed before the
next attempt -- never a blind retry:

1. Run `33551894420` -- `POST /v1/pods` 400: RunPod's REST v1 schema requires `ports`
   and `dockerStartCmd` as JSON arrays, not strings. Fixed (`create_pod` now sends
   arrays; `shlex.split`s the caller's shell-style start-command string). Zero GPU
   touched (failed before any create attempt could succeed).
2. Run `33553415542` -- past the schema fix, hit a genuine RunPod capacity response
   (`"no instances currently available"`) on all 4 approved GPUs, but only under
   `cloudType: COMMUNITY` (hardcoded). Fixed: sweep COMMUNITY fully across the ranked
   pool first, then fall back to a full SECURE sweep before concluding
   `POD_CAPACITY_UNAVAILABLE`.
3. Run `33554748857` -- the SECURE sweep succeeded: **first real Pod created**
   (`wmohn5wxu8q9il`, RTX 4090, SECURE). Health GET immediately after creation hit a
   RunPod proxy 403 in ~1.5s -- accepting a create call is not the same as the Pod being
   RUNNING. **Guaranteed teardown fired correctly regardless** (confirmed via API:
   `pod_stopped`, `final_status: EXITED`) -- a false health failure, not a safety-guarantee
   failure. Fixed: wait for `RUNNING` (bounded, `create_wait_timeout_s`) before touching
   the health endpoint.
4. Run `33556235100` -- **second real Pod created** (`chr1zet8f6sr3r`, RTX 4090,
   COMMUNITY this time). Reached `RUNNING` almost instantly (`elapsed_s ~0` -- RunPod's
   status field reflects scheduling, not "the container is actually listening"), then the
   health endpoint 403'd for the **entire** 180s poll window, never once answering. Added
   a bounded retry loop around the health GET itself (already built for #3, exercised
   here) -- still no pass. Rather than guess a fourth cause with a fourth paid Pod, added
   a **zero-cost, read-only diagnostic path** (`fetch_pod_logs`, `DIAGNOSE_POD_LOGS_ID`)
   to inspect the already-stopped Pod's own state and container logs without provisioning
   anything.
5. Diagnostic reads (runs `33557956110`, `33558239960`, no Pod created, no GPU cost)
   against the stopped `chr1zet8f6sr3r`: its own record confirms `dockerStartCmd`
   (`["python3","-m","cutsell_worker.pod_job_server"]`) and `ports` (`["8080/http"]`)
   were both recorded exactly as sent -- ruling out a payload-shape problem as the cause
   of the 403s. `machine: {}` came back empty despite `desiredStatus`/`lastStatusChange`
   confirming the Pod ran from 20:43:06 to 20:46:08 -- suggestive of the Pod never
   actually landing on real host/GPU compute, but not conclusive from this field alone.
   RunPod's REST v1 has no `/pods/{id}/logs` route (confirmed: 400, "does not exist in
   the specification"); REST v2's `/pods/{id}/logs` exists but returned 403 with no body
   under this account's API key -- container-level logs are not reachable with the
   access this integration currently has, so the exact mechanism behind the empty
   `machine` record remains unconfirmed.

**Net effect validated end-to-end against real RunPod Pods (not fakes)**: GPU search
(ranked pool, cost ceiling, COMMUNITY-then-SECURE cloud sweep, capacity-error detection)
all matched real API responses exactly as designed; guaranteed STOP fired correctly and
was verified via API on both real Pod creations, at every failure mode exercised so far.
The **open question is whether this account's RunPod Pod product delivers real compute
behind an accepted create call** -- the same "accepted but nothing real happens" shape
already open with RunPod Support for Serverless workers (see D-041's support report).
Total real GPU time across both live Pod creations: a few seconds each; no paid GPU was
left running at any point (independently confirmed via API state, not assumed).

**Holding here rather than continuing to iterate blindly**: the two available zero-cost
diagnostic avenues (REST v1 and v2 Pod logs) are now exhausted without a conclusive
answer, and further diagnosis would require either RunPod documentation/support access
this session doesn't have, or provisioning another paid Pod on a guess. Per the standing
"diagnose before retrying" instruction, this checkpoint is reported for a decision on
next steps (e.g., add this Pod evidence to the existing RunPod support ticket; authorize
one more live test with a longer health-poll bound in case it is genuinely a slow image
pull; or pause Pod work pending Support's response on the parallel Serverless issue) --
not resolved by further unsupervised live attempts.

### D-042 follow-up — CutSell-Pod-QA template cloned from EditDNA-Worker-2 (Steps 1-6)

**Status: CANONICAL (template created + validated; first live Pod test from it not yet
run -- see Step 7 / task #92)**

Per the separate "create a CutSell QA template from known-working EditDNA-Worker-2"
directive, rather than keep hand-building ad-hoc inline Pod configs, `CutSell-Pod-QA`
(RunPod template id `5moabglc4m`) was created as an evidence-grounded clone of the
already-proven `EditDNA-Worker-2` template (id `07g9dovc17`) -- **the base template was
read live and never mutated** (`runpod_pod_template.py`'s `find_template_by_name`/
`create_pod_template` only ever POST to the generic `/v1/templates` create endpoint,
confirmed by `test_create_pod_template_never_calls_base_mutation_endpoint`). Both the
live base fetch (run `33563258690`, `template_action=fetch_base`) and the clone creation
(run `33564506657`, `template_action=create_qa_template`, HTTP 201) are real, GitHub
Actions-verifiable evidence -- not reconstructed from memory.

**Step 3 -- environment/startup parity table** (env values never shown, per the standing
redaction policy -- only names, and only whether each is present/required):

| SETTING | EditDNA-Worker-2 (base) | CutSell-Pod-QA (new) | STATUS | WHY CHANGED |
|---|---|---|---|---|
| id | `07g9dovc17` | `5moabglc4m` | CHANGED | new template gets its own RunPod-assigned id |
| name | `EditDNA-Worker-2` | `CutSell-Pod-QA` | CHANGED | directive requirement |
| imageName | `madiator2011/better-pytorch:cuda12.4-torch2.6.0` | `ghcr.io/automatedretailservices/cutsell-serverless@sha256:2240ec43fc4e1f7203658842a66ec00e5069666b8a668e57d870230fff842433` | CHANGED | Step 5: current canonical CutSell image, immutable digest of the exact commit built; base image carried no CutSell code at all |
| dockerStartCmd | none (relies on the base image's own default entrypoint) | `["python3","-m","cutsell_worker.pod_job_server"]` | ADDED | Step 4 (startup parity): confirmed live there was no historical start command to preserve or chain after -- this is a clean addition, not a replacement of anything |
| ports | none configured on base | `["8080/http"]` | ADDED | `pod_job_server`'s `GET /health` / `POST /run` HTTP surface (Step 6 requirement) |
| category | `NVIDIA` | `NVIDIA` | PRESERVED | no proven need to change |
| containerDiskInGb | `80` | `80` | PRESERVED | Step 2: "preserve every known-working Pod-level config item unless proven need to change" |
| volumeInGb | `60` | `60` | PRESERVED | same |
| volumeMountPath | `/workspace` | `/workspace` | PRESERVED | same |
| containerRegistryAuthId | `""` (none) | `""` (none) | PRESERVED | GHCR image is public on both; no registry auth configured either template |
| isServerless | `false` (Pod template) | `false` | PRESERVED | both are Pod templates, not Serverless endpoints |
| isPublic | `false` | `false` | PRESERVED | QA-only, not published |
| startSsh | `true` | `true` | PRESERVED (not by choice) | confirmed live: `POST /v1/templates` rejects `startSsh`/`startJupyter` as input fields ("Extra input keys..."); whatever the new template gets is RunPod's own creation default, not something this session's code can set either way |
| startJupyter | `true` | `true` | PRESERVED (not by choice) | same as above |
| env -- 48 base vars | present, real values | present, values carried over unchanged | PRESERVED | Step 2: "prefer exact inheritance/parity over manually inventing configuration" |
| env -- 6 `CUTSELL_*` additions | absent | added (see table below) | ADDED | current CutSell Video00 harness config, mirrored from the existing Serverless RAW workflow's own env block, for full-benchmark parity once Video00 is authorized on Pod |
| readme | `""` | `""` | PRESERVED | neither template sets one |

**Env var table** (evidence: live redacted fetch, runs `33563258690`/`33564506657`, cross-
referenced against every `cutsell_worker/*.py` module via `grep` -- never guessed):

*Required for the full Video00 benchmark path (`_focused`/`_locked_selection` in
`serverless_handler.py`), present on both templates, NOT required for `health`* (`_health()`
only imports `torch` and reads no env var at all -- confirmed by reading its body):

| NAME | present old? | present new? | req. startup? | req. health? | req. full Video00? |
|---|---|---|---|---|---|
| `S3_BUCKET` | yes | yes | no | no | yes -- artifact upload, `S3_BUCKET is required` in 8 modules |
| `AWS_REGION` | yes | yes | no | no | yes -- S3 client region (`serverless_handler._upload_artifact`) |
| `AWS_ACCESS_KEY_ID` | yes | yes | no | no | yes -- boto3 S3 credentials (`config.py` presence check) |
| `AWS_SECRET_ACCESS_KEY` | yes | yes | no | no | yes -- boto3 S3 credentials (`config.py` presence check) |
| `GEMINI_API_KEY` | yes | yes | no | no | yes, only when `CUTSELL_HYBRID_LLM_ENABLED=1` (`brain_runtime.py`) |
| `FFMPEG_BIN` | yes | yes | no | no | yes -- audio boundary completion during render (`audio_boundary_completion.py`) |

*Present on both, referenced somewhere in `cutsell_worker/`, but NOT by `health` or by
either Video00 op (`_focused`/`_locked_selection`) -- other, unrelated code paths (API-
server-side persistence, or a presence-only check with no call site)*:

| NAME | present old? | present new? | req. startup? | req. health? | req. full Video00? |
|---|---|---|---|---|---|
| `REDIS_URL` | yes | yes | no | no | no -- gates project/draft/auth/notifications/batch/render-history/account-lifecycle persistence modules, not the GPU render path |
| `OPENAI_API_KEY` | yes | yes | no | no | no -- `config.py` presence-check only, no call site found |

*Present on both, zero references anywhere in current `cutsell_worker/*.py`* (preserved
from the base template for parity per Step 2, not required for startup, health, or
Video00 under the current codebase -- 40 vars):

`ASR_DEVICE`, `ASR_ENABLED`, `BAD_TAKES_ENABLED`, `BENCHMARK_INTERNAL_API_KEY`,
`BOUNDARY_REFINER_ENABLED`, `BOUNDARY_REFINER_HEAD_STEP_SEC`,
`BOUNDARY_REFINER_MIN_DURATION_SEC`, `BOUNDARY_REFINER_TAIL_STEP_SEC`,
`COMPOSER_MAX_PER_SLOT`, `COMPOSER_MIN_SEMANTIC`, `EDITDNA_CTA_MIN_SCORE`,
`EDITDNA_HOOK_MIN_SCORE`, `EDITDNA_LLM_MODEL`, `EDITDNA_MIN_CLIP_SCORE`,
`EDITDNA_USE_LLM`, `FFPROBE_BIN`, `HEAD_TRIM_SEC`, `PRESIGN_EXPIRES`, `PYTHONPATH`,
`S3_ACL`, `S3_PREFIX`, `TAIL_TRIM_SEC`, `TAKEJUDGE_FRAMES_PER_CLIP`,
`TAKEJUDGE_MAX_CLIPS`, `TAKEJUDGE_MIN_SCORE`, `TAKE_JUDGE_ENABLED`,
`TAKE_JUDGE_FRAMES`, `TAKE_JUDGE_MAX_GROUPS`, `TAKE_JUDGE_MAX_TAKES`,
`TAKE_JUDGE_MODEL`, `VISION_ENABLED`, `VISION_INTERVAL_SEC`, `VISION_MAX_SAMPLES`,
`VISUAL_BAD_THRESHOLD`, `WHISPER_DEVICE`, `WHISPER_MODEL`, `W_FACE`, `W_SEMANTIC`,
`W_VISION`, `W_VISUAL`. (Likely historical EditDNA-Worker-2 config for features not
present, renamed, or not yet wired in the current CutSell codebase -- not deleted from
the clone, per "prefer exact inheritance over manually inventing configuration.")

*Present ONLY on `CutSell-Pod-QA` (new additions, absent from the base template)* --
mirrors the existing Serverless RAW workflow's own env block (`cutsell-video00-raw-v5-
auto-microtrim.yml`) so a future Pod-backed Video00 run is configured identically to
Serverless, not required for `health`:

| NAME | present old? | present new? | req. startup? | req. health? | req. full Video00? |
|---|---|---|---|---|---|
| `CUTSELL_BRAIN_BACKEND` | no | yes | no | no | yes -- selects `runpod_local` brain backend |
| `CUTSELL_EDITORIAL_MODE` | no | yes | no | no | yes -- selects `clean_cut` |
| `CUTSELL_ASR_MODEL` | no | yes | no | no | yes -- ASR model size (`medium`) |
| `CUTSELL_HYBRID_LLM_ENABLED` | no | yes | no | no | yes -- gates the Gemini-backed hybrid path |
| `CUTSELL_HYBRID_PROVIDER` | no | yes | no | no | yes -- selects `google` |
| `CUTSELL_UNIFIED_SELECTION_REASONER` | no | yes | no | no | yes -- matches Serverless's current flag (note: dormant in the active Clean Cut Core V1 path per D-019/D-020; carried for harness parity, not reactivating it) |

**Step 6 -- validation before Pod creation (all confirmed against the actual created
template, run `33564506657`, before any Pod is created)**:
- required env names: all 48 base + 6 `CUTSELL_*` present (54 total) -- confirmed in the
  created template's own echoed `env` dict.
- correct image: `ghcr.io/automatedretailservices/cutsell-serverless@sha256:2240ec43...`
  (immutable digest of the commit that created the template, per Step 5).
- correct startup/bootstrap: `dockerStartCmd` == `["python3","-m","cutsell_worker.pod_job_server"]`,
  confirmed in the created template's own record.
- correct ports: `["8080/http"]`, confirmed in the created template's own record.
- correct disk/volume settings: `containerDiskInGb=80`, `volumeInGb=60`,
  `volumeMountPath=/workspace` -- all inherited unchanged from the base template.
- correct registry/auth configuration: `containerRegistryAuthId=""` on both (GHCR image
  is public); no auth needed.
- no Blackwell-specific assumptions: the template itself carries no GPU pin at all (GPU
  selection happens at Pod-creation time via `create_pod(..., template_id=...)`'s
  `gpuTypeIds`, which still goes through the existing D-042 approved pool / Blackwell
  exclusion / cost ceiling -- unchanged by this work).
- no missing historical Pod settings: every base field not explicitly listed as CHANGED/
  ADDED above is preserved verbatim (see parity table).
- tests: 19 tests in `tests/test_runpod_pod_template.py` (payload construction, env
  redaction/merge, base-never-mutated, `startSsh`/`startJupyter` regression), 5 template-
  related tests in `tests/test_runpod_pod_health_gate.py`, plus 2 new tests in `tests/
  test_runpod_pod_provider.py` locking `create_pod(..., template_id=...)`'s minimal
  payload shape (sends only `name`/`templateId`/`gpuTypeIds`/`gpuCount`/`cloudType`;
  never duplicates `imageName`/`ports`/`env`/`dockerStartCmd`/`containerDiskInGb`, which
  the template itself supplies) and that omitting `template_id` leaves the existing
  inline-config path completely unaffected. Full targeted suite (`tests/
  test_runpod_pod_provider.py` -- 40 tests, `tests/test_runpod_pod_template.py` -- 19,
  `tests/test_runpod_pod_health_gate.py` -- 10, `tests/test_runpod_orchestration.py`) all
  green; `compileall` clean.

**Not yet done**: Step 7 -- the first live Pod created FROM `CutSell-Pod-QA` (rather than
an inline ad-hoc config matching it), health-only, then guaranteed STOP. Gated on
explicit authorization per the standing directive ("Do NOT create another Pod yet").

### D-042 Step 7 -- first live test from CutSell-Pod-QA (2026-09-01, authorized): health FAILED, STOP confirmed

**Status: CANONICAL (health-only test run to completion; Video00 remains gated on
separate authorization, not granted and not run)**

Authorized, ran on `feature/runpod-pod-on-demand` via `create_pod(template_id=
"5moabglc4m")` (the `POD_TEMPLATE_ID` workflow input added this cycle, see below).

**Result: health FAILED -- `POD_HEALTH_APP_FAILURE`, identical pattern to the earlier
inline-config live tests.** Pod `l368986gtg5ijn` (RTX 4090, COMMUNITY cloud, created via
the template, `templateId` field independently confirmed as `5moabglc4m` on the live Pod
record) reached `RUNNING` almost instantly (`elapsed_s ~1.3e-7`) but its health endpoint
403'd for the entire 180s poll window, never once answering -- the same failure mode as
D-042's very first live Pod tests on the hand-built inline config. **This directly answers
the question Directive B was testing: the known-working EditDNA-Worker-2-derived template
does NOT by itself fix the Pod startup/readiness problem.** Whatever prevents
`pod_job_server.py` from answering `/health` is not a template-configuration difference
(image, ports, dockerStartCmd, env now all inherited from a template cloned from a
proven-working base) -- it points at something else in the RunPod Pod product itself
(see the still-open `machine: {}` question from the earlier checkpoint) or at the
container's actual startup on this specific host/image, neither of which template parity
can fix. **Per the standing "do not blindly retry" instruction, no further live Pod test
was run after this result.**

**Guaranteed STOP -- confirmed, twice over, independently**: the code's own `finally`
block called `teardown()` (`pod_stopped`, `final_status: EXITED`) immediately after the
180s poll timed out, and the workflow's independent force-stop safety net ran immediately
after as a no-op backup. A separate, later, read-only `diagnose_pod_id` check against
`l368986gtg5ijn` independently re-confirmed via RunPod's own API: `lastStatusChange:
"Exited by user: ... 23:22:51 UTC"` -- the exact same timestamp as the code's own
`pod_stopped` event. No paid GPU was left running.

**A genuinely useful secondary finding, not asked for but worth recording**: the
*previous* live Pod (`z7sgsafyvlto5p`, the one from the earlier checkpoint) was found
already `EXITED` by RunPod itself when this run inspected it -- despite no explicit STOP
call from this session's code having reached it (the session's own attempt to do so was
interrupted before it ran; see the incident below). This is the first direct evidence
that RunPod's Community Cloud can apparently reclaim/terminate a Pod on its own when its
health endpoint never responds, independent of any client-side stop call -- consistent
with, though not proof of, the standing suspicion (parallel to D-041's own open Serverless
support ticket) that this account's Pod compute may not always be real, delivered
capacity. Not investigated further this cycle (would require RunPod support access this
session doesn't have); noted for the existing support thread.

**Full evidence, as required**:

| Field | Value |
|---|---|
| Template used | `CutSell-Pod-QA` (`5moabglc4m`), confirmed via live Pod's own `templateId` field |
| Pod ID | `l368986gtg5ijn` |
| GPU | NVIDIA GeForce RTX 4090 |
| Cloud type | COMMUNITY |
| Hourly rate | not exposed this run (GPU catalog fetch itself failed -- `pod_gpu_catalog_unavailable` -- same as several earlier live runs; ranked-pool fallback still worked correctly) |
| Image digest | `ghcr.io/automatedretailservices/cutsell-serverless@sha256:2240ec43fc4e1f7203658842a66ec00e5069666b8a668e57d870230fff842433` (confirmed on the live Pod record) |
| Pod creation timestamp | 2026-09-01 23:19:49 UTC |
| RunPod RUNNING timestamp | 2026-09-01 23:19:50 UTC (`elapsed_s` ~0) |
| First successful HTTP/app response | never -- health endpoint answered 403 for the full 180.4s poll |
| Total startup-to-health-verdict latency | 180.4s (poll timeout), health never passed |
| Health response | none (403, no parseable app body) |
| Device name / compute capability / torch version / compiled CUDA / CUDA available | not obtainable -- the app never answered `/health` to report them |
| ffmpeg/ffprobe status | not exposed by `_health()` even when it does answer (not part of its payload) |
| Application/job-server readiness | not confirmed -- no evidence `pod_job_server.py` ever started listening |
| STOP request | issued by `teardown()` in `finally` immediately on poll timeout, confirmed `pod_stopped`/`EXITED` via API; independent workflow force-stop step ran as a no-op backup |
| Final Pod state | `EXITED` (`lastStatusChange: "Exited by user"`), independently re-confirmed by a later read-only check |
| Approx. GPU-active time / cost | ~3 minutes on this Pod (23:19:49-23:22:51); the earlier `z7sgsafyvlto5p` was active at most ~7.5 minutes before being found already exited. At the previously-observed $0.34/hr RTX-4090-COMMUNITY rate, total cost across both this cycle's Pods is on the order of a few cents |

**Not run**: Video00. Per the explicit instruction, health FAILED means STOP + diagnose +
report, not retry -- done exactly that way.

### Security incident during Step 7 (2026-09-01) -- live secrets leaked, fixed same cycle

While chasing this result, a genuine operational mistake compounded into a real security
exposure, both now fixed and both recorded here honestly rather than glossed over:

1. **Premature cancellation from a sandbox timing artifact.** This session's own elapsed-
   time tracking (summing nominal background sleep durations) proved unreliable in this
   environment -- background-task completion notifications were arriving far later than
   the commands' real durations, making a live run that was actually healthy and only
   ~3 minutes old look, by this session's own miscount, like it had been stuck for 30+
   minutes. Acting on that false read, this session cancelled a live, in-bounds workflow
   run 5 seconds before its own 180s health-poll timeout would have concluded it
   naturally. GitHub Actions' cancellation hard-killed the Python process without
   letting its `finally` block run, so the created Pod (`z7sgsafyvlto5p`) was left
   without an explicit STOP from this session's code (RunPod itself was later found to
   have already exited it independently -- see above -- so no orphaned billing resulted,
   but that was not something this session could have known at the time). **Fixed
   procedurally, not by touching Pod-lifecycle code**: all further waits in this session
   used `date -u`/a `Monitor`-based real-time-verified loop instead of counting nominal
   sleep durations, and no further premature cancellation occurred.
2. **A real, if lower-severity, cleanup misstep while responding to (1).** A follow-up
   dispatch to check/reuse the stale Pod omitted pinning `POD_TEMPLATE_ID`/the image,
   which would have triggered a needless rebuild; caught and cancelled before it reached
   the Pod-lifecycle code (confirmed via job-step timestamps), no Pod was touched by it.
3. **A genuine secret leak.** The zero-cost, read-only `diagnose_pod_id` path
   (`_diagnose_pod_logs` in `runpod_pod_health_gate.py`) printed RunPod's raw
   `GET /v1/pods/{id}` response without redaction. Unlike a template GET (already
   redacted via `redact_template_env` since the template-management work earlier this
   cycle), a **Pod** GET embeds the Pod's real, live env values -- this put real
   credentials (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `GEMINI_API_KEY`,
   `OPENAI_API_KEY`, `BENCHMARK_INTERNAL_API_KEY`, `REDIS_URL`, an SSH public key) into a
   GitHub Actions log in cleartext, and into this session's own conversation transcript.
   **Fixed same cycle** (commit on `feature/runpod-pod-on-demand`): `_diagnose_pod_logs`
   now passes both the fetched Pod state and any dict-shaped logs-fetch body through the
   same `redact_template_env` helper before ever printing them; 1 new regression test
   locks that real-shaped values never reach stdout while names and every other field
   still do. A follow-up commit fixing this was itself initially blocked by GitHub's own
   push protection for a different reason (the regression test's first draft used the
   actual leaked key values as "realistic" fixture data -- itself a mistake, replaced
   with clearly-fake placeholder strings before anything left this repository) and,
   separately, an unrelated authoring mistake (a literal backtick-wrapped `env` in a
   commit message run through a double-quoted shell string triggered real shell command
   substitution, embedding this sandbox's own environment variables into the commit
   object) -- caught before push via direct inspection of the git object, corrected by
   rewriting the commit message from a file, and pushed clean. Neither mistake reached
   the remote.

**Action required from the user, not resolvable by this session**: the exposed
credentials (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `GEMINI_API_KEY`,
`OPENAI_API_KEY`, `BENCHMARK_INTERNAL_API_KEY`, `REDIS_URL`) should be treated as
compromised and rotated. This session cannot rotate them (no credential-issuer access);
flagged here and to the user directly per the standing security rule that this runs in
parallel with editorial QA, not deferred to launch.

### D-042 follow-up -- direct-execution model (bypass HTTP/8080) built; same root cause found (2026-09-01/02)

**Status: CANONICAL (direct-execution QA path built and tested; root cause of the
underlying Pod failure now believed shared with the HTTP-transport path, not fixed by
bypassing HTTP; no further live Pod test run pending diagnosis/authorization)**

Per the user's explicit follow-up directive ("restore the known-working execution
model"), instead of continuing to debug `pod_job_server.py`'s HTTP `/health` transport,
a second, independent QA execution path was built that bypasses HTTP entirely, closer to
the historical EditDNA-Worker-2 workflow:

- **`cutsell_worker/pod_direct_benchmark_entrypoint.py`** (new): becomes the Pod's own
  `dockerStartCmd` (its one-shot main process, not a persistent server). Runs three tiny
  non-editorial sanity checks (CUDA via `torch`, `ffmpeg` present, `cutsell_worker`
  importable), uploads the result to S3 first (`sanity_check.json` -- this doubles as
  the "prove direct execution works" validation and the orchestrator's
  runtime-readiness signal), then calls `cutsell_worker.serverless_handler.run_op(op,
  payload)` directly -- the exact same dispatcher RunPod Serverless already uses, no
  second editor implementation. Any exception is caught and uploaded to a known S3 key
  rather than crashing silently.
- **`runpod_pod_direct_benchmark_gate.py`** (new): orchestration only. Fetches the live
  `CutSell-Pod-QA` template, builds an inline `PodExecutionConfig` from it (image/env/
  disk inherited verbatim, `CUTSELL_BENCHMARK_PAYLOAD_JSON` merged in), reuses
  `RunPodPodExecutionProvider`'s already-tested lifecycle unchanged (`ensure_ready()`,
  no HTTP at all), polls S3 (never HTTP, never container logs) for readiness and
  completion, downloads the result JSON + MP4, classifies the outcome, and always tears
  the Pod down in `finally`.
- **`.github/workflows/cutsell-video00-pod-direct-raw.yml`** (new): `workflow_dispatch`
  only, independent concurrency group, does not replace or touch the HTTP-based
  workflow or RunPod Serverless.
- **Real latent bug fixed along the way**: `PodExecutionConfig` never had an `env`
  field; `create_pod()`'s own `env` parameter was never threaded through the provider's
  fresh-create call site. Every earlier inline-mode live Pod test this cycle ran with an
  empty env (harmless for a bare HTTP health check, fatal for an entrypoint that needs
  `CUTSELL_BENCHMARK_PAYLOAD_JSON` plus AWS/S3/Gemini credentials). Fixed, with
  regression tests.
- 145 tests across the new/changed modules, plus the full `tests/test_cutsell_*.py`
  glob (1267 tests), green. `compileall` clean.

**First live test (2026-09-02, `feature/runpod-pod-on-demand`, Pod `aejb4hkhegwpk5`,
RTX 4090, COMMUNITY cloud): `SANITY_CHECK_TIMEOUT`.** The Pod itself was created and
reported `RUNNING` normally (near-instant, consistent with every earlier finding that
RunPod's Pod status flips before any container application is actually ready), but
`sanity_check.json` never appeared in S3 within the 300s bound -- meaning the
entrypoint's own tiny sanity checks never even ran, let alone the benchmark.

**Root-cause diagnosis, from a zero-cost read-only `diagnose_pod_id` check against the
stopped Pod (2026-09-02)**: the live Pod record's `dockerStartCmd`
(`["python3", "-m", "cutsell_worker.pod_direct_benchmark_entrypoint"]`), `templateId`
(`""`, confirming inline mode was used as designed), and `env` (contains
`CUTSELL_BENCHMARK_PAYLOAD_JSON` alongside every expected template variable, names
confirmed, values redacted) are all exactly correct -- **this cycle's own configuration
work is confirmed not at fault.** But the Pod record's `machine` field is `{}` (empty)
despite a populated `machineId` (`w8d456itlyzv`) -- **the same empty-machine-record
signature already seen on the HTTP-transport path's own live tests** (see the D-042
Step 7 checkpoint above). This is now the leading hypothesis for the shared root cause
of both failure modes: the Pod's control-plane status (`RUNNING`, later `EXITED by
user`) advances normally, but the underlying GPU host record is empty/never resolves,
consistent with (though not independently proven as) the container never actually
executing on real, delivered compute -- which would fully explain both symptoms without
needing two separate explanations (HTTP ingress unreachable vs. S3 egress never
attempted): in both cases, nothing inside the container ever ran at all. **This means
bypassing the HTTP transport did not bypass the true fault** -- it only changed which
downstream signal exposed the same upstream problem. Both `templateId` GPU-pool
(`l368986gtg5ijn`, Step 7) and inline-mode (`aejb4hkhegwpk5`, this test) live Pods have
now hit this same `machine: {}` pattern, on COMMUNITY cloud, ruling out
"config-shape-specific" as an explanation.

Guaranteed STOP confirmed the same way as every earlier test: the gate script's own
`finally`/`teardown()` call (`pod_stopped`, `final_status: EXITED`), independently
re-confirmed via the later read-only diagnostic (`lastStatusChange: "Exited by user"`),
plus the workflow's independent force-stop safety net as a no-op backup. No paid GPU
left running.

**Per the standing "do not blindly retry" instruction, Video00 was never run** (Step 4's
gate -- "if sanity checks pass, immediately proceed to Video00" -- was never reached).
**Not yet resolved at the time this checkpoint was first written / needed either RunPod
support engagement or a different Pod/host allocation to distinguish from a one-off bad
host** -- flagged to the user rather than re-attempted speculatively. **Fixed same
cycle**: the stdout-buffering gap in `runpod_pod_direct_benchmark_gate.py` (missing
`flush=True`, unlike the entrypoint script) that made one log gap look artificially
short -- all `print()` calls in that script now use `flush=True`, consistent with the
entrypoint.

### D-042 follow-up -- controlled SECURE-cloud direct-execution test authorized (2026-09-02)

**Status: INFRASTRUCTURE READY (cloud-type override built and tested; live SECURE-cloud
test pending dispatch)**

Per the user's explicit authorization to isolate cloud type as the one variable under
test -- rather than accept the default COMMUNITY-then-SECURE sweep, where a SECURE
attempt only ever happens as a capacity fallback -- `PodExecutionConfig` gains a new
`cloud_types: tuple[str, ...] = POD_CLOUD_TYPES` field (default unchanged: the existing
sweep, every other caller/workflow unaffected). `_select_and_create_fresh` now sweeps
`self._cfg.cloud_types` instead of the hardcoded `POD_CLOUD_TYPES` constant, so a caller
passing `("SECURE",)` gets a genuinely SECURE-only Pod (or a real
`POD_CAPACITY_UNAVAILABLE` if SECURE truly has none) on the approved GPU pool (RTX 4090
preferred, falling back to A40/RTX A6000/L4 -- no Blackwell, no COMMUNITY for this run).

`runpod_pod_direct_benchmark_gate.py`'s `build_direct_exec_config()` takes an optional
`cloud_types` parameter (defaults to `None`, meaning "don't override -- use
`PodExecutionConfig`'s own default sweep"); `main()` reads a new `QA_POD_CLOUD_TYPE` env
var (case-insensitive, validated against `POD_CLOUD_TYPES`, aborts before touching the
template/API on an invalid value) and records `cloud_types_requested` in the run summary
when set. `.github/workflows/cutsell-video00-pod-direct-raw.yml` gains a matching
`qa_pod_cloud_type` workflow_dispatch input (default `""` -- every future dispatch keeps
the existing sweep unless this input is explicitly set), threaded to `QA_POD_CLOUD_TYPE`.

No other variable changes from the prior `SANITY_CHECK_TIMEOUT` live test: same git head
lineage, same canonical image digest (read live from the `CutSell-Pod-QA` template, never
hardcoded), same inline direct-execution `PodExecutionConfig` construction, same
`python3 -m cutsell_worker.pod_direct_benchmark_entrypoint` start command, same sanity-
check-then-Video00 gate. No editorial code, D-040, PyTorch/CUDA, or Serverless touched.

11 new targeted tests (5 in `tests/test_runpod_pod_provider.py` locking the default sweep
order, SECURE-only creation on the first attempt with no COMMUNITY calls at all,
SECURE-only correctly returning `POD_CAPACITY_UNAVAILABLE` without ever falling back to
COMMUNITY, and `cloud_types` validation rejecting an invalid or empty tuple; 6 in
`tests/test_runpod_pod_direct_benchmark_gate.py` locking the pure-function default/
override behavior and the `QA_POD_CLOUD_TYPE` env-var wiring including its
case-insensitivity and invalid-value rejection). Full `tests/test_cutsell_*.py` glob
(1267 tests) green. `compileall` clean. Workflow YAML validated.

**Not yet run**: the live SECURE-cloud test itself. Per Directive B's exact sequence,
this is dispatched once this infrastructure lands, with sanity-check-first, Video00 only
on sanity pass, guaranteed teardown, and an explicit report on whether COMMUNITY->SECURE
changed the `machine: {}` failure signature observed on both prior COMMUNITY-cloud tests.

### D-042 -- controlled SECURE-cloud test result: identical failure, cloud type ruled out (2026-09-02)

**Status: CANONICAL (SECURE-cloud test run to completion; COMMUNITY vs. SECURE ruled out
as the variable behind the failure; root cause narrowed to something shared across both
cloud types and both execution transports)**

Dispatched on `feature/runpod-pod-on-demand` at `8601e71` via the new `qa_pod_cloud_type:
SECURE` workflow input -- every other input identical to the prior `SANITY_CHECK_TIMEOUT`
test (same source video, same op, same timeouts, same template). Run
[33595330866](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33595330866).

**Result: `SANITY_CHECK_TIMEOUT` again -- the exact same failure as the COMMUNITY-cloud
test.** Pod `u1nftzx1i1lrik` was created genuinely SECURE (`gpu_type_id: "NVIDIA GeForce
RTX 4090"`, `cloud_type: "SECURE"`, confirmed zero COMMUNITY attempts in the log --
`cloud_types_requested: ["SECURE"]` in the run summary directly proves the new
`cloud_types` override worked as designed, not silently falling back), reached `RUNNING`
near-instantly (`elapsed_s ~1.1e-7`, same pattern as every earlier live test), but
`sanity_check.json` never appeared in S3 within the 300s bound. This time the log
timestamps are trustworthy (the `flush=True` fix from the immediately preceding commit
worked): the S3 poll genuinely ran its full coded ~308s span (05:36:18.95 to
05:41:27.04), not a buffering artifact.

**A zero-cost, read-only `diagnose_pod_id` check against the stopped Pod
(`u1nftzx1i1lrik`) confirms the same `machine: {}` signature a third time**: `machineId:
"2qwkos95nrhz"` is populated but `machine: {}` resolves empty, `templateId: ""` (inline
mode, as designed), `imageName` matches the canonical digest exactly. **This is now the
same empty-machine-record pattern on all three live Pods tested across this D-042
follow-up cycle -- two on COMMUNITY cloud (`l368986gtg5ijn` via the HTTP transport,
`aejb4hkhegwpk5` via direct-exec) and one on SECURE cloud (`u1nftzx1i1lrik`, direct-exec)
-- ruling out COMMUNITY-vs-SECURE as the variable behind the failure.** Combined with the
earlier transport comparison (HTTP daemon vs. direct one-shot exec, also both hitting the
same signature), **two of the three most obvious candidate variables (cloud type,
execution transport) are now eliminated**; what remains in common across every failing
Pod is: this RunPod account, the `CutSell-Pod-QA`-derived template/image, and the
approved GPU pool search itself.

Guaranteed STOP confirmed the same way as every earlier test: the gate script's own
`finally`/`teardown()` call (`pod_stopped`, `final_status: "EXITED"`, logged at
05:41:28.44, ~1.4s after the timeout was declared), independently re-confirmed via the
read-only diagnostic (`lastStatusChange: "Exited by user: ... 05:41:27"`), plus the
workflow's force-stop safety net as a no-op backup. Total GPU-active time on this Pod:
~5m10s (05:36:18 to 05:41:28). No paid GPU left running.

**Direct answer to "did switching COMMUNITY -> SECURE change the container-execution
behavior": no.** The failure is identical in every observable respect (timing, the
`machine: {}` signature, the complete absence of any application-level output) on both
cloud types. **Per the standing "do not blindly retry" instruction, Video00 was never
run** (the sanity gate never passed) and no further live Pod test was dispatched pending
the user's direction. The two variables ruled out this cycle (cloud type, execution
transport) narrow, but do not yet identify, the remaining root cause -- RunPod support
engagement (citing all three Pod IDs and their identical `machine: {}` evidence) is now
the strongest lead, since every configuration variable this session controls has been
exhausted without resolving it.

### D-042 -- FINAL POD EXECUTION ISOLATION: minimal known-good image also fails, RunPod/account issue confirmed (2026-09-02)

**Status: CANONICAL (isolation test run to completion; DECISION A per the user's own
decision tree; RunPod/account/host execution problem effectively confirmed; CutSell
image/runtime/startup ruled out as the cause)**

Per the user's explicit "FINAL POD EXECUTION ISOLATION -- MINIMAL KNOWN-GOOD IMAGE"
directive, this isolates the one remaining shared variable across all three prior live
tests (COMMUNITY+HTTP, COMMUNITY+direct-exec, SECURE+direct-exec -- all running the
CutSell image): is the failure in the CutSell image/runtime itself, or in RunPod's
account/host execution layer, which would show up regardless of image?

**New, standalone infrastructure, deliberately independent of every CutSell-specific
dependency**: `runpod_pod_minimal_isolation_gate.py` creates a Pod from a minimal public,
non-CutSell image (`nvidia/cuda:12.4.1-base-ubuntu22.04`) running a trivial shell command
(`sh -c 'echo POD_EXECUTION_OK; sleep 60'`) -- no S3, no AWS/Gemini/Redis credentials, no
`pod_job_server`, no port 8080, no Video00, no CutSell code imported at all. Reuses only
the already-tested `RunPodPodExecutionProvider` lifecycle and the existing
`get_pod`/`fetch_pod_logs`/`delete_pod` primitives. Since there is no CutSell-side S3
marker to poll by design, readiness/completion are observed via bounded snapshots of the
Pod's own GET state (watching `machine` populate or not over a fixed window) plus a
best-effort container-log fetch. Classifies `CONTAINER_EXECUTION_CONFIRMED` (machine
populated, or logs contain the sentinel) vs. `CONTAINER_EXECUTION_NOT_CONFIRMED`. ALWAYS
stops AND deletes the Pod in `finally` (this ad hoc image has no reuse story, unlike the
`CutSell-Pod-QA` identity the other D-042 scripts keep alive). New, independent,
`workflow_dispatch`-only workflow (`cutsell-pod-minimal-isolation-raw.yml`) with no GHCR
login and no Docker build step -- the whole point is to never reference the CutSell
image. 22 targeted tests, full `tests/test_cutsell_*.py` glob (1267 tests) green,
`compileall` clean.

**Live test result (2026-09-02, `feature/runpod-pod-on-demand`, Pod `ca7r6mz7f960ga`, run
[33597545157](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33597545157)):
`CONTAINER_EXECUTION_NOT_CONFIRMED`.** RTX 4090, SECURE cloud (forced via
`POD_ISOLATION_CLOUD_TYPE=SECURE`, matching the standing directive), reached `RUNNING`
near-instantly (`elapsed_s ~1.5e-7`, the same pattern as every earlier live test). Six
bounded snapshots across the full 95-second observation window (0.7s, 16.4s, 32.1s,
47.8s, 63.5s, 95.0s) **all** show `"machine": {}` -- empty every single time, despite a
populated `machineId` (`i066evwrx2d6`), exactly like every prior live Pod this cycle.
The container-log fetch got the same 400/403 this account has always gotten regardless of
image, so its absence carries no new information either way. `machine_ever_populated:
false`, `log_confirms_execution: false` -- no evidence of execution by either channel
available to this session.

**This is the fourth live Pod across the entire D-042 cycle to show the identical
`machine: {}` signature -- and the first one running an image that is not CutSell's at
all.** Combined with the earlier two ruled-out variables (cloud type, execution
transport), **all three configuration axes this session can control -- image/runtime,
cloud type, and execution transport -- are now ruled out as the cause.** What remains
constant across every failing Pod is only this RunPod account and its GPU-pool
provisioning itself.

**Per the user's own decision tree: DECISION A.** "The minimal public image also never
executed. RunPod/account/host execution problem is effectively confirmed." Per the
standing directive, **no further Pod experiments are run.** The CutSell image, runtime,
and Pod startup sequence (`Dockerfile.cutsell.serverless`, `cutsell_worker.pod_job_server`,
`cutsell_worker.pod_direct_benchmark_entrypoint`) are cleared of suspicion by this
result -- none of them were even present in this test's image, yet the exact same failure
reproduced. **RunPod Support escalation, citing all four Pod IDs
(`l368986gtg5ijn`, `aejb4hkhegwpk5`, `u1nftzx1i1lrik`, `ca7r6mz7f960ga`) and their
identical `machine: {}` evidence across two cloud types, two execution transports, and
now two entirely different container images, is the necessary next step** -- this
session has exhausted every configuration variable it can control and cannot diagnose
further without RunPod-side account/infrastructure visibility this session does not have.

Guaranteed cleanup confirmed the same way as every earlier test, with an added layer
since this Pod had no reuse story: the gate script's own `finally` called
`provider.teardown()` (`pod_stopped`, `final_status: "EXITED"`) followed immediately by
`delete_pod()` (`pod_deleted: true`), and the workflow's independent force-stop +
force-delete safety net ran as a no-op backup. No paid GPU left running; total GPU-active
time on this Pod was under 2 minutes.

## D-043 — CutSell Modal GPU execution: first live validation (infrastructure only)

**Status: INFRASTRUCTURE READY (Modal added as a third GPU execution backend; live L4
smoke test pending dispatch)**

Per the user's explicit directive, this adds Modal as an ADDITIONAL GPU execution backend
for CutSell QA, alongside RunPod Serverless and RunPod Pod -- both fully preserved,
neither modified. All three backends share one provider-neutral interface
(`gpu_execution_provider.GPUExecutionProvider`); every backend ultimately invokes the
exact same canonical `cutsell_worker.serverless_handler.run_op` dispatcher -- no second
editor implementation, same discipline as D-042's RunPod Pod addition.

**Scope of this phase, explicitly**: a minimal, controlled Modal L4 GPU smoke test only.
No Video00 on Modal yet -- that integration is separately gated on this phase's PASS plus
explicit future authorization, exactly like D-042's own health-only-first sequencing.

### Architecture

- `gpu_execution_provider.py`: `EXECUTION_BACKEND_MODAL = "modal"` added alongside the
  existing `EXECUTION_BACKEND_SERVERLESS`/`EXECUTION_BACKEND_POD` (now three, not two,
  valid backends). `ModalExecutionProvider` implements the same `health_check()`/
  `teardown()` Protocol the other two backends already implement, returning the exact
  same `HealthCheckResult` dataclass (schema parity across all three backends, tested).
  `teardown()` is a documented no-op -- Modal's serverless GPU functions scale to zero
  automatically once a call returns; there is no persistent Pod/endpoint to stop or
  delete, unlike the two RunPod backends. `health_check()` takes an injected zero-arg
  `invoke` callable rather than hardcoding a specific Modal SDK method call
  (`Function.lookup` vs `Function.from_name` etc. have changed across Modal SDK
  versions) -- this keeps the class fully unit-testable without the `modal` package
  installed, and defers the actual in-process SDK-call-shape decision to whenever
  Video00-on-Modal integration is separately authorized and a Modal SDK version is
  pinned.
- `modal_gpu_config.py`: modal-package-free constants/validation, importable and
  testable in any environment (same "no runpod-package dependency for config" precedent
  as the existing RunPod modules). `APPROVED_MODAL_GPU_TYPES = ("L4",)` -- the only
  approved type this phase; `EXCLUDED_MODAL_GPU_TYPES` names A100/A100-80GB/H100/H200/
  L40S explicitly so a reader sees exactly what is deliberately excluded, not just
  "everything else." `require_modal_gpu_type`/`require_modal_timeout`/
  `require_modal_token_env` hard-reject anything outside the approved pool, a
  non-positive or excessive timeout, or missing/blank `MODAL_TOKEN_ID`/
  `MODAL_TOKEN_SECRET` -- named refusals, never silent substitution.
- `modal_gpu_diagnostics.py`: pure diagnostic logic (`collect_gpu_diagnostics()`) --
  GPU model, `torch.cuda.is_available()`, torch version, CUDA version, compute
  capability, ffmpeg/ffprobe availability+version, Python version, elapsed runtime,
  completion status. Fully independent of the `modal` package so it's directly
  unit-testable; the Modal-decorated function in `modal_gpu_minimal_test.py` is a thin
  wrapper that calls it, never a duplicated implementation.
- `modal_gpu_minimal_test.py`: the actual Modal App/Function. Invoked via Modal's own
  documented `modal run` CLI (not an in-process SDK call from
  `gpu_execution_provider.py` -- see that file's own module docstring for why), which
  is also how the ephemeral run/teardown lifecycle is handled: no persistent app stays
  deployed after the CLI invocation completes.

### Image/runtime strategy audit (per the standing "do not silently diverge" directive)

**Decision: reuse the exact same base image `Dockerfile.cutsell.serverless` builds
FROM (`madiator2011/better-pytorch:cuda12.4-torch2.6.0`), via `modal.Image.from_registry`,
plus the same `apt-get install ffmpeg` step the Dockerfile already runs.** This is
closest to the user's "Option A" framing (same dependencies/runtime as the Dockerfile)
implemented as a Modal Image chain rather than duplicating a Dockerfile — no separate
torch/CUDA/ffmpeg versions were introduced, and none needed to be: this base image
already has a working Python/pip (confirmed by the Dockerfile's own successful use of
it), so no `add_python` override was needed either. The full
`requirements.cutsell.worker.txt` dependency set and the `cutsell_worker` package
itself are deliberately NOT installed in this phase's image -- not needed for a
torch/CUDA/ffmpeg-only smoke test that runs no CutSell code, and adding them is
deferred to whenever full Video00-on-Modal integration is separately authorized. A
regression test (`test_cutsell_base_image_matches_dockerfile`) locks the base image
string against the Dockerfile's own first line, so any future edit to either one that
causes silent drift fails CI rather than going unnoticed. **No CUDA base change, no
torch/CUDA/ffmpeg version change of any kind was needed or made -- nothing to report
under the "report before changing" clause.**

### Cost safety

Exactly one approved GPU type (L4), enforced at import time by
`require_modal_gpu_type` inside `modal_gpu_minimal_test.py` itself (fails fast before
any Modal call, not a runtime check that could be bypassed) and independently re-tested
against the module's actual `@app.function(gpu=...)` call kwargs. Bounded timeout
(`DEFAULT_MODAL_TIMEOUT_S = 300`, hard-ceilinged at `MAX_MODAL_SMOKE_TEST_TIMEOUT_S =
600` for this phase). No explicit `scaledown_window`/`container_idle_timeout` override
-- deliberately: Modal's own default scale-to-zero behavior already satisfies "no idle
container remains" for a single ephemeral `modal run` invocation, and omitting the kwarg
avoids a possible SDK-version-specific naming mismatch (that parameter was renamed
across Modal SDK versions) causing an avoidable failure on this first live attempt.

### Tests

50 new tests across three new Modal-specific test files (20 in `tests/
test_modal_gpu_config.py`, 23 in `tests/test_modal_gpu_diagnostics.py`, 7 in `tests/
test_modal_gpu_minimal_test.py` [modal-stubbed, same technique `tests/
test_pod_direct_benchmark_entrypoint.py` already uses for `runpod`]), plus 10 new/2
updated cases in `tests/test_gpu_execution_provider.py` (14 total in that file now)
covering the three-backend constant set, `ModalExecutionProvider`'s L4-only
enforcement, health-check pass/fail/exception handling, teardown no-op, and
cross-backend `HealthCheckResult` schema parity. Running the full combined D-042+D-043
targeted suite (206 tests) confirms **RunPod Serverless and RunPod Pod are completely
untouched** -- no shared code path was modified, only additive. Full `tests/
test_cutsell_*.py` CI glob and `compileall` run before any live Modal call, per the
standing discipline.

### Live test result (2026-09-02): FAIL -- Modal token validation failed

**Status: FAILED, not retried (per the standing "do not blindly retry" directive)**

Dispatched on `feature/runpod-pod-on-demand` at `4db9849` via `workflow_dispatch`
(run [33601072592](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33601072592)).
Checkout, the `require_modal_token_env` presence check (both `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` were present/non-blank as GitHub Actions secrets), and
`pip install modal` (installed `modal-1.5.5` cleanly, ~3s) all succeeded. The actual
`modal run modal_gpu_minimal_test.py` invocation failed after ~3s with:

```
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ Token validation failed                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**This is a clean, well-formed rejection from Modal's own API/CLI, not a code or
infrastructure bug in this repo's own Modal integration** -- the auth-presence check
passed (both secrets are configured and non-blank in the repo), `modal` installed and
ran normally, and it reached Modal's servers before being rejected specifically on
token validity. The two most likely causes, neither resolvable by this session (no
access to the actual secret values or the Modal dashboard): (1) the configured
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` pair is stale/revoked/mistyped, or (2) a
whitespace/formatting artifact in how the secret value was pasted into GitHub (e.g. a
trailing newline or space) that changes the token's effective bytes.

**No further live Modal dispatch was made or is planned** until the user verifies (or
regenerates) the `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` pair in GitHub's repo secrets
against a fresh Modal dashboard token pair. Nothing about this phase's own code
(image/runtime reuse, L4-only enforcement, bounded timeout) is implicated by this
failure -- none of it was reached before the auth rejection.

**Real bug found and fixed in the same cycle, from this run's own evidence**: the "Run
Modal L4 minimal GPU smoke test" step's script used `set -uo pipefail` intending to
avoid `set -e` so a failing `modal run` would still reach the `exit_code` capture line
-- but GitHub's own `bash` shell wrapper already runs every `run:` script under
`-e -o pipefail` regardless of what `set` flags the script body itself requests; only an
explicit `set +e` actually clears it. Confirmed live in this exact run's own log
(`shell: /usr/bin/bash --noprofile --norc -e -o pipefail {0}`): the failing `modal run`
aborted the script immediately, the `exit_code` output was never set, and the next
step's `if: github.event_name == 'workflow_dispatch'` (no `always()`) meant it was
skipped entirely -- so the raw Modal error only surfaced in the step's own log, not in
the intended formatted summary. Fixed by adding an explicit `set +e` before
`set -uo pipefail` in that step, matching the working pattern the "Print result summary"
step already used. This fix does not change the Modal auth outcome itself -- it only
restores the intended clear PASS/FAIL reporting for the next dispatch, whenever the
token issue is resolved.

### Retest result (2026-09-02): auth PASSED, but a crash-loop FAIL -- fixed, not yet re-tested

**Status: FAILED again, differently -- root cause fixed in code, not yet re-verified live**

After the user regenerated the `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` pair and replaced
the GitHub repo secrets, one authorized retest was dispatched at head `e5c2ccd`
(run [33602989294](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33602989294)).

**1. Authentication: PASSED.** No "Token validation failed" this time -- the new token
pair is valid.

**2. What actually happened: a crash-loop, not a clean pass or a clean stop.** The
Modal container started successfully and repeatedly (confirmed by its own CUDA banner
printing, `CUDA Version 12.4.1`, each time) -- proving image pull, registry access, and
L4 GPU provisioning all worked -- but every single container instance immediately
crashed with:

```
ModuleNotFoundError: No module named 'modal_gpu_config'
```

at `modal_gpu_minimal_test.py`'s own `from modal_gpu_config import (...)` line.
**Root cause**: `modal.Image.from_registry(...).apt_install(...)` only builds the
container's base filesystem -- it does NOT make this repo's own local sibling Python
modules (`modal_gpu_config.py`, `modal_gpu_diagnostics.py`) importable inside the
remote container. `modal run <script>.py` auto-mounts only the one script file being
run, not its local imports -- a real gap in the initial implementation, not something
Modal's docs make obvious by default.

**3. A second real gap this exposed: no retry bound.** Modal's own default
container-crash retry behavior kept relaunching fresh L4 containers on each crash --
visible retries at 07:22:58, 07:23:03, 07:23:07, 07:23:13, 07:23:17, 07:23:22,
07:23:28, 07:23:32, 07:24:06, 07:24:49, with Modal's own crash-loop detector logging
`Function modal_gpu_minimal_test.run_minimal_gpu_check is crash-looping: containers are
repeatedly failing to start.` at 07:27:43, then one more attempt at 07:37:37 -- growing
backoff, but still retrying ~18 minutes in. **The run never reached a clean PASS or a
clean, deliberate STOP**: GitHub's own job-level `timeout-minutes: 20` terminated the
still-in-progress `modal run` process (`##[error]The operation was canceled.`,
conclusion `cancelled`), not this session and not Modal's own function completing or
giving up. This directly violates the explicit "no retry loop" requirement -- the
`@app.function` decorator never set `retries=0`, so Modal's own default retry count
applied instead of the required zero.

**Both fixed in code, same cycle**: `modal_gpu_minimal_test.py`'s image now chains
`.add_local_python_source("modal_gpu_config", "modal_gpu_diagnostics")` after
`.apt_install("ffmpeg")`, making both local modules importable remotely; the
`@app.function(...)` call now passes `retries=0` explicitly. 2 new regression tests
(modal-stubbed, no live call) lock both: `test_image_mounts_local_source_so_the_
container_can_import_it` and `test_function_has_no_retry_loop`. Full D-042+D-043
targeted suite (208 tests) and the complete `tests/test_cutsell_*.py` CI glob both
green; `compileall` clean.

**Cost note**: each crash-looping container failed within ~1 second of starting (the
import error is immediate), so no single container ran for a meaningful GPU-billed
duration; the CI job itself ran for the full ~20 minutes before its own timeout, but
that is GitHub Actions runner time, not billed GPU time. Approximate Modal GPU cost for
this run: low (a handful of sub-second L4 container starts), but non-zero and not
precisely knowable from this evidence alone -- Modal's own billing dashboard would have
the exact figure.

**Not yet re-tested live.** Per the standing "do not blindly retry" discipline, this
fix is reported here rather than immediately re-dispatched -- the next live attempt
follows once explicitly authorized.

### Third retest result (2026-09-02, head f9526c1): both prior fixes confirmed working; a new, different FAIL -- fixed, not yet re-tested

**Status: FAILED a third time, differently -- both defects from the second run are
CONFIRMED FIXED; a new defect found and fixed, not yet re-verified live**

One further authorized retest was dispatched at head `f9526c1` (the crash-loop fix
commit) -- run [33612105029](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33612105029).

**Both prior fixes verified working live:**
- **`retries=0` confirmed effective**: exactly ONE container attempt this time (the
  entire "Run Modal L4 minimal GPU smoke test" step took 16 seconds total), vs. the
  previous run's 10+ retries spread across ~18 minutes. No crash-loop.
- **`add_local_python_source` confirmed effective**: Modal's own setup log shows
  `Created mount PythonPackage:modal_gpu_config, PythonPackage:modal_gpu_diagnostics`
  -- both local modules were successfully included this time, and the container's own
  log shows a torch-internal warning (`Failed to initialize NumPy...`) that only fires
  partway through `collect_gpu_diagnostics()`'s own torch/CUDA block -- proof the
  function body was actually entered and executed remotely, past the point of the
  previous `ModuleNotFoundError`.

**The new failure**: the GitHub Actions runner (running `modal run` locally, with only
the `modal` package pip-installed -- no `torch`) crashed while trying to deserialize
the function's *return value*:

```
Stopping app - uncaught exception raised locally: DeserializationError("Deserialization
failed because the 'torch' module is not available in the local environment.").
...
ModuleNotFoundError: No module named 'torch'
DeserializationError: Deserialization failed because the 'torch' module is not
available in the local environment.
```

**Root cause**: despite `bool()`/f-string casts already applied to most torch-derived
fields, `collect_gpu_diagnostics()` (in `modal_gpu_diagnostics.py`) had at least one
unguarded field (`result["cuda_version"] = torch.version.cuda`, assigned with no
`str()` cast) and no final guarantee that the returned dict was free of any
torch-specific object -- Modal's serialization protocol pickles the return value
including its real type, and unpickling on the caller side (no `torch` installed
there, by design -- only `modal` is) fails if any such object leaked through.

**Fixed, defense-in-depth (two independent layers)**:
1. Every torch-derived field is now explicitly `str()`- or `int()`-cast at the point
   of assignment (`cuda_version`, `torch_version`, `gpu_model`, `compute_capability`),
   not relying on assignment-order luck.
2. The full result is now JSON-round-tripped (`json.loads(json.dumps(result,
   default=str))`) before returning -- a second, independent guarantee that only
   plain JSON-native types (str/int/float/bool/None) ever cross the Modal
   serialization boundary, regardless of which specific field would otherwise have
   leaked. This makes the exact root-cause field moot going forward: any future
   torch-typed leak in this function is now structurally impossible, not just
   patched at the one field that failed this time.

1 new regression test (`test_collect_gpu_diagnostics_never_leaks_a_non_plain_object`)
simulates a "leaky" torch whose version/capability values are custom objects (not
plain `str`/`int`) and asserts the final result is fully plain-JSON-native regardless.
Full D-042+D-043 targeted suite (209 tests) and the complete `tests/test_cutsell_*.py`
CI glob green; `compileall` clean.

**Cost note**: this run's single container ran the full diagnostic body (several
seconds of real L4 GPU time, more than the sub-second crash-loop attempts from the
prior run) before failing at the client-side deserialization step -- the GPU-side
work itself completed and the container exited normally; only the local unpickling
of its result failed. Approximate cost: still low (one L4 container, a few seconds),
non-zero.

**Not yet re-tested live.** Per the explicit "do not patch and immediately rerun
again" instruction, this fix is reported here and the next live attempt awaits
separate authorization.

### Fourth retest result (2026-09-02, head 508f386): PASS -- D-043 first live validation complete

**Status: CANONICAL PASS.** Modal L4 minimal GPU execution is confirmed working
end-to-end for this account. RunPod Serverless and RunPod Pod remain the production
backends, fully untouched.

One further authorized retest was dispatched at head `508f386` (the DeserializationError
fix commit) -- run [33622147447](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33622147447),
app run [ap-1o6K8hS6HndONVkqArGnjq](https://modal.com/apps/automatedretailservices/main/ap-1o6K8hS6HndONVkqArGnjq).
Every step succeeded, including "Print result summary" (the step whose own
`exit_code != "0"` check is the authoritative PASS/FAIL gate).

**Full evidence**:

| Field | Value |
|---|---|
| Authentication | PASS |
| Modal app / function | `cutsell-gpu-minimal-isolation` / `run_minimal_gpu_check` |
| Container attempts | Exactly ONE (`retries=0` confirmed working a second time -- no crash-loop) |
| Local modules mounted | Confirmed: `Created mount PythonPackage:modal_gpu_config, PythonPackage:modal_gpu_diagnostics` |
| GPU model | NVIDIA L4 |
| `torch.cuda.is_available()` | `true` |
| torch version | `2.6.0+cu124` |
| Compiled CUDA version | `12.4` |
| Compute capability | `8.9` |
| ffmpeg | `ffmpeg version 4.4.2-0ubuntu0.22.04.1` (present) |
| ffprobe | `ffprobe version 4.4.2-0ubuntu0.22.04.1` (present) |
| Python version (in container) | `3.10.12` |
| Diagnostic body runtime | `3.465s` (the function's own measured `elapsed_s`) |
| Result deserialization | Succeeded -- the full JSON dict was received and printed by the local `modal run` process with no `DeserializationError` |
| `torch_error` | `null` |
| `ok` / `completion_status` | `true` / `"COMPLETED"` |
| Retry/crash loop | None -- one container, one attempt |
| Clean completion | `Stopping app - local entrypoint completed.` / `✓ App completed.` |
| Scale-to-zero | Confirmed by design (ephemeral `modal run` invocation) and by the app's own completion event -- no persistent container remains |
| Startup latency | ~6.7s from function-object creation to the container's own CUDA banner (image already cached from prior runs) |
| Total step wall time | 16s ("Run Modal L4 minimal GPU smoke test" step, 10:58:05-10:58:21 UTC) |
| Approximate cost | Low, sub-cent range (one L4 container, single-digit seconds of GPU-attached wall time) -- Modal's own billing dashboard has the exact per-run figure |

**This closes out D-043's first-phase live validation.** All three defects found across
this cycle (crash-loop from missing local-source mounting, unbounded Modal-side
retries, and a torch-typed value leaking past Modal's serialization boundary) are
fixed and now verified live, not just in mocked tests. Modal is confirmed as a working
third GPU execution backend for this account on the approved L4 GPU type.

**Per the standing directive, Video00 was NOT run.** The next separately-authorized
step is the full Video00 benchmark Modal was brought in to eventually run -- the exact
same canonical pipeline RunPod Serverless/Pod already run (ASR, Attempt reconstruction,
Idea clustering, Retry families, BestTake, Claim coverage, Composite, Story validation,
CanonicalEditPlan, FinalEditReviewer, Freeze, Boundary, Render, PostRenderWatchListenQC,
Delivery gate, Human Gold regression checks) via the exact same `run_op()` dispatcher --
no new benchmark design, no second editor implementation. That integration (installing
`cutsell_worker` + its full dependency set into the Modal image, wiring the real
Video00 payload through) is not yet built and requires separate authorization before
any code is written for it.

### Full Video00 execution phase (2026-09-02): build checkpoint, authorized

"AUTHORIZED — BUILD MODAL FULL VIDEO00 EXECUTION AND RUN ONE FULL BENCHMARK." Resumes
the exact full Video00 benchmark RunPod Serverless was supposed to execute, on Modal as
an additional execution backend -- not a new benchmark, not a new editor, not a reduced
Modal-specific pipeline. RunPod Serverless and RunPod Pod remain fully available and
untouched.

**Exact test head verified (Section 2).** `git merge-base feature/runpod-pod-on-demand
origin/cutsell/mobile-v1-clean` equals `cutsell/mobile-v1-clean`'s own tip
(`a26c099`) -- zero divergence; the infra branch contains 100% of the editorial
branch's current commits plus D-042/D-043 infra on top. D-037 (`e561b8b`), D-038
(`3f7122b`), D-039 (`796b0dc`), and D-040 (`19f6612`) are all confirmed ancestors of
HEAD. `benchmarks/video00_regression_qa.json` (18 checks) and
`benchmarks/video00_selection_lock.json` are present. The new workflow
(`cutsell-video00-modal-raw.yml`) re-verifies this ancestry live on every dispatch,
rather than trusting a one-time audit.

**Canonical engine preserved (Section 1).** `modal_video00_full_benchmark.py`'s remote
function imports `cutsell_worker.serverless_handler.run_op` inside the function body
(never at module top level -- `cutsell_worker` pulls in torch/mediapipe/faster-whisper,
none of which the plain `modal run` CLI process has) and calls it with the payload
unmodified: `run_op(payload["op"], payload)`. No forked/duplicated editorial logic.

**Modal image extended, not reinvented (Section 3).** `modal_gpu_config.py` gained
`CUTSELL_APT_PACKAGES` (the exact `Dockerfile.cutsell.serverless` apt-get install list,
guarded by a test that re-parses the Dockerfile and asserts byte-for-byte match),
`CUTSELL_REQUIREMENTS_FILE` (a path, not a copied list -- Modal's own
`pip_install_from_requirements` reads the actual `requirements.cutsell.worker.txt` file
at image-build time), and `CUTSELL_RUNPOD_PIP_SPEC` (`runpod>=1.7,<2`, matching the
Dockerfile's separate pip step -- `serverless_handler.py` imports `runpod` unconditionally
at module level even though `run_op()` itself never calls it). The image chain:
`Image.from_registry(CUTSELL_BASE_IMAGE).apt_install(*CUTSELL_APT_PACKAGES)
.pip_install_from_requirements(CUTSELL_REQUIREMENTS_FILE).pip_install(CUTSELL_RUNPOD_PIP_SPEC)
.add_local_python_source("cutsell_worker")` -- the last step mounts the whole package
from the exact checked-out commit, which is this backend's "exact test head" guarantee
(no Docker build/push/digest-pin needed the way RunPod Serverless RAW requires).

**Secrets wired from the single existing source of truth (Section 4).** No hand-typed
env-var list. The workflow fetches the live `EditDNA-Worker-2` RunPod template (same
template the RunPod Serverless RAW workflow itself reads), masks every value with
`::add-mask::`, and writes the full `env` dict to a local JSON file. `modal_video00_full_benchmark.
_resolve_env_secret()` reads that file (path via `CUTSELL_ENV_JSON_PATH`) and builds
`modal.Secret.from_dict(...)` from it at `modal run` invocation time -- values never
printed, never baked into the image, never a second static Modal Secret store to keep
in sync by hand. Returns an empty secret (not an error) when the env var is unset, so
the module stays importable for tests that stub `modal` entirely.

**Return-value discipline reused, not reinvented.** `serverless_handler._focused()`
already returns a small, plain-JSON-native compact summary (the exact shape RunPod
Serverless returns) -- the full diagnostics tree is written to S3 as `result.json` by
`run_op()`'s own existing `_upload_artifact` machinery, never returned in-process. This
sidesteps the D-043 `DeserializationError` class of bug entirely (nothing torch-typed is
ever part of the return value), with the same defensive `json.loads(json.dumps(...,
default=str))` round-trip still applied as a second, independent guarantee.

**Execution safety (Section 11).** Exactly one approved GPU type (L4, via
`require_modal_gpu_type`), `retries=0` (same crash-loop protection this phase's own live
failure required), and a new, separate timeout ceiling
(`DEFAULT_MODAL_VIDEO00_TIMEOUT_S = MAX_MODAL_VIDEO00_TIMEOUT_S = 5400`, mirroring RunPod
Serverless RAW's own 5400s/90-minute poll bound for this exact six-minute source video --
never widening `MAX_MODAL_SMOKE_TEST_TIMEOUT_S`, which stays scoped to the minimal smoke
test). No persistent container: Modal's own scale-to-zero applies the instant the one
ephemeral `modal run` invocation's function call returns.

**New workflow: `cutsell-video00-modal-raw.yml`** (`workflow_dispatch` only, concurrency-
grouped, never a push trigger). Mirrors `cutsell-video00-raw-v5-auto-microtrim.yml`'s
diagnostics-printing and validator steps verbatim (same jq blocks, same three
validators: `validate_video00_selection_lock.py`, `validate_video00_architecture.py`,
and -- newly wired in, previously unused by any workflow --
`validate_video00_regression_qa.py` against the 18-check Human Gold manifest for the
"X/18" report). No RunPod endpoint/template lifecycle management: unlike the Serverless
RAW workflow, this one creates no persistent RunPod resource, so there is nothing to
tear down beyond Modal's own automatic scale-to-zero.

**Tests.** `tests/test_modal_gpu_config.py` gained coverage for the new timeout
ceiling/validator and three Dockerfile-consistency tests (apt packages, requirements
file path, runpod pip spec). `tests/test_modal_video00_full_benchmark.py` (18 tests)
covers the App/Image/Function/Secret wiring and `run_op()` delegation, using the same
modal-package-free stub pattern as `test_modal_gpu_minimal_test.py` -- extended with a
`__cutsell_test_stub__` marker so the two test files' shared `sys.modules["modal"]`
stub merges attributes instead of one file silently winning the stub and leaving the
other's needed attributes (e.g. `modal.Secret`) missing, regardless of pytest's
collection order. Full `tests/test_cutsell_*.py` CI glob (1267 tests) green; the
combined D-042+D-043 targeted suite is now 232 tests (up from 209). `compileall` clean
on every changed/new file.

**Status at this checkpoint: CODE FIXED, TESTS PASS.** Not yet run live -- the one
authorized full Video00 Modal benchmark dispatch and its result are reported separately
once it completes.

### Live full Video00 Modal dispatch result (2026-09-02, head cc25a72)

**MODAL FULL RAW COMPLETE.** First-ever successful end-to-end execution of the canonical
Clean Cut Core V1 engine on Modal (run 33636255124, App `ap-WX9iPZfMQnhPQJsM1rZwLm`,
Function `fu-UtPpmyf89ZpgT6JFPc8mDr`). The infra breakthrough is real: `run_op("focused",
...)` executed the SAME canonical pipeline RunPod runs, unmodified, on an L4 GPU, and
returned a result -- but the two prior live dispatches on this run each surfaced a real
bug diagnosed and fixed live, never guessed at:

- **First dispatch (run 33626736759):** failed before touching Modal -- `jq -n`'s
  default pretty-printed (multi-line) payload JSON was written straight into a plain
  `KEY=value` line appended to `$GITHUB_ENV`, which GitHub's own env-file command
  rejects. Fixed with `jq -nc` (compact, single-line).
- **Second dispatch (run 33626961988):** crash-looped for ~90 minutes (3 container
  attempts, each dying in ~2s with `ModuleNotFoundError: No module named
  'modal_gpu_config'`) before ever reaching `run_op()`. Root cause: the image's
  `.add_local_python_source(...)` mounted `cutsell_worker` but omitted
  `modal_gpu_config`, the sibling module this script's own top-level code imports --
  the exact same class of bug already fixed once for `modal_gpu_minimal_test.py`.
  Diagnosed via a NEW, separate, read-only diagnostic workflow
  (`cutsell-modal-video00-diag-logs.yml`) that queries `modal app list`/`modal app
  logs` by App ID (ephemeral apps are not resolvable by name) WITHOUT touching the
  running benchmark -- confirmed genuinely stuck (deterministic, zero forward
  progress across 3 attempts), then cancelled and fixed.
- **Third dispatch (run 33636255124):** ran clean. `run_op()` returned `ok: true`,
  `elapsed_sec` ~495s of actual pipeline execution (Modal GPU runtime; ~513s total step
  wall time including `modal run` CLI overhead). Rendered a candidate, ran full
  PostRenderWatchListenQC, uploaded the diagnostic-invalidated preview + result.json to
  S3, downloaded them, printed full canonical diagnostics.

**Caveats on this report's completeness:** GitHub's own log masking (`::add-mask::`
applied per-value to the ENTIRE live RunPod template env dict, per D-043 Section 4's
"pass the full template through" design) blanks out many numeric fields in the CI log
wherever those digits happen to coincide with a masked env value (several of the
template's own config values are short integers) -- e.g. `selected_count` shows as
`"***8"` in the raw log. The raw artifact ZIPs (which are NOT masked) could not be
downloaded directly in this session: the organization's egress policy blocks the
Actions artifact blob-storage host (`*.blob.core.windows.net`), confirmed via a direct
`curl` 403/`connect_rejected`. `validate_video00_architecture.py` and
`validate_video00_regression_qa.py` never ran (the workflow mirrors
`cutsell-video00-raw-v5-auto-microtrim.yml`'s own convention of NOT gating these
steps with `if: always()`, so a Selection Lock failure short-circuits the rest). All
of the following is therefore a best-effort MANUAL read of the (unmasked) transcript
TEXT the "Print full canonical diagnostics" step already printed to the CI log --
never an automated validator run -- and should be treated as directionally reliable,
not authoritative to the check.

- **ARCHITECTURE:** no automated PASS/FAIL (validator skipped). Manual review of
  `stage_status` shows the full canonical component chain instantiated and completed
  (ASR 54 segments, attempt_reconstruction, take_grouping, take_judge,
  selection_phase_authority `clean_cut_core_v1_idea_first_keep_discard`,
  final_story_coherence_validation `status: applied`, canonical_edit_plan
  `validation_state: frozen_ready`, final_edit_reviewer `PASS`, repair_loop `PASS`,
  selection_boundary_contract `status: verified`, human_boundary_polish, render,
  live_render_qc) -- structurally intact end to end.
- **SEMANTIC: FAIL.** `selection_locked: false`, `historical_regression_qa_pass: false`
  in the Selection Lock report. Manual comparison of the actual KEEP sequence against
  every item in `benchmarks/video00_regression_qa.json` found at least two clear,
  concrete violations of the manifest's own `forbidden_contains` checks (both present
  when they must be absent):
  - `sonography_bad_take_absent`: the incomplete take ("...pues porque cada año que me
    hacía mínimo dos estados.") is KEPT alongside the complete retake ("...funcionando
    perfectamente.") -- an incomplete+complete duplicate pair survived Selection.
  - `pimples_bad_monolith_absent`: the monolithic pimples take ("También me salían
    espinillas en esta parte de aquí detrás de la oreja y todo el cuello...") is KEPT
    alongside BOTH the micro-fragment version and the later winner -- three
    realizations of the same idea all survived.
  - A likely third: `family_context_preserved`'s idea (the hereditary-cancer claim) has
    TWO separately-phrased retakes both kept ("Esta es mi experiencia. Soy la única en
    mi familia..." and "Soy la primera en mi familia..."), on two clip_ids neither of
    which appears in `canonical_edit_plan.diagnostics.ideas` at all -- meaning
    IdeaClusterer treated them as two unrelated ideas rather than one retry family, so
    BestTakeResolver never got a chance to pick a winner between them.
  All three are exactly the "no duplicate retry realization" / "no incomplete+complete
  duplicate pair" class of regression D-020/D-040 exist to prevent. The diagnostics
  block's own `final_story_coherence_validation.not_implemented` field explicitly lists
  `"general_non_numeric_non_negation_contradiction_detection"` -- StoryValidator's
  contradiction check does not catch this class of issue by design; it depends on
  IdeaClusterer/RetryFamilyResolver correctly grouping retries upstream, which appears
  to be where this regression actually originates. `lost_critical_claims: []` and
  `lost_semantic_atoms: []` -- nothing required was cut; if anything, too much survived.
  Everything else checked by hand (cancer hook, sonography good-take ordering before
  diagnosis, biopsy nodule, papillary cancer sentence, acne-back, all three pimples
  micro-fragments, pimples later winner, hair loss, gastritis, CTA, both required_order
  sequences) appears present and correctly ordered.
- **Human Gold regression QA: not run** (validator skipped, see caveat above). Manual
  best-effort estimate against the 18-check manifest: roughly 15/18, with the 3 failures
  named above -- reported as an estimate, not an authoritative score.
- **CanonicalEditPlan: PASS** (`validation_state: frozen_ready`, `freeze_blocked:
  false`).
- **Freeze: PASS** (not blocked) -- but this reflects only the contradiction classes
  StoryValidator currently implements, not the duplicate-retry-realization issue found
  above.
- **Render attempted: YES** (1 attempt).
- **PostRenderWatchListenQC: FAIL.** `status: PHYSICAL_FAIL_UNREPAIRABLE`, 12 physical
  findings (2 `LINGERING_ACCIDENTAL_SILENCE`, 10 `ABRUPT_AUDIO_DISCONTINUITY`),
  `repair_requested: true`, `repair_applied: null` -- the bounded physical repair loop
  did not resolve it. `output_path: null` -- no deliverable file was produced; only the
  diagnostic-invalidated preview was uploaded to S3.
- **delivery_status: `NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW`. deliverable: false.**
- **Modal GPU runtime:** ~495s (~8.3 min) of actual `run_op()` execution; ~513s total
  step wall time. **Approximate cost:** low cents (single L4, ~8.5 minutes) -- exact
  figure is in Modal's own billing dashboard.
- **Artifacts:** `cutsell-video00-modal-human-review` (289 MB: diagnostic-invalidated
  preview MP4, `video00-modal.json`, Human Gold reference MP4), `cutsell-video00-modal
  -validator-reports` (2.8 KB: Selection Lock report only -- architecture/regression-qa
  reports were never generated), `cutsell-video00-modal-run-log` (2 KB), all on run
  33636255124.

**Per the standing directive: STOPPING here.** No further benchmark launched
automatically. No engine code modified based on this result -- the finding (a likely
IdeaClusterer/RetryFamilyResolver gap letting at least 3 retry pairs both survive as
unrelated "complete" ideas) is reported for the user's review, not acted on
unilaterally. A secondary, lower-priority finding worth fixing regardless of the
semantic result: the workflow's blanket `::add-mask::` over every RunPod template env
value makes ordinary benchmark numbers unreadable in CI logs whenever they coincide
with a masked short numeric config value -- a future fix should mask only genuinely
secret-shaped values (credentials, keys), not every value indiscriminately.

## D-044 — Retry-family / idea-clustering regression audit (2026-09-02)

**Correction to the D-043 report above:** that report's semantic finding, written from
masked CI-log text, speculated the sonography/pimples/hereditary over-retention was an
"IdeaClusterer/RetryFamilyResolver gap." Forensic tracing against the real (unmasked)
`result.json` -- fetched via a new read-only, zero-GPU workflow
(`cutsell-video00-d044-forensic-extract.yml` + `benchmarks/video00_d044_forensic_
extract.py`, since the org's egress policy blocks the Actions artifact blob-storage
host and the D-043 workflow's blanket env masking corrupted numeric CI-log output)
-- found a single, precise, **infrastructure-level** cause, not a code defect.

### Forensic trace (real clip_ids/timestamps/text, run 33636255124)

`diagnostics.semantic_idea_equivalence` = `{"candidate_pair_count": 0,
"merged_pair_count": 0, "status": "not_requested"}` for the ENTIRE run.
`take_grouping_provider.reconcile_semantic_idea_equivalence` returns exactly this
tuple when `arbiter is None` (checked before any candidate pair is even computed --
`if len(groups) < 2 or arbiter is None: return groups, {"status": "not_requested", ...}`,
and `len(groups)==25` here, so it is the `arbiter is None` branch). `brain_runtime.
build_brain_runtime` only constructs that arbiter when `requested_hybrid` (`_env_true
(values.get("CUTSELL_HYBRID_LLM_ENABLED"))`) is true. `take_grouping_reason` for this
run was `"baseline_local; weak_retry_envelope_restored; final_sibling_reconciled"` --
lexical-tier passes only; the bounded semantic-equivalence tier never engaged.

All three named regions show the identical structural signature -- each pair sits in
its own **separate singleton** `take_group_members` entry (never became a candidate
pair at all), each pair is temporally well within the module's own 30-second
cross-group eligibility window, and downstream stages that DID run (BestTakeResolver/
`claim_coverage_best_take`, StoryValidator) behaved correctly on the groups they were
actually given:

- **SONOGRAPHY.** `clip_1fc2eb28a33c9b39f313` (25.60-34.60s, "Nunca se nos ocurrió
  hacer un chequeo de sonografía de la tiroides, pues porque cada año que me hacía
  mínimo dos estados.") and `clip_818bad730eefa36c8620` (35.46-46.42s, "Nunca se nos
  ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis
  exámenes la tiroides salía como que estaba funcionando perfectamente.") -- 0.86s
  apart, reordered near-paraphrase of the same sentence template ("chequeo de
  sonografía de la tiroides" vs "chequeo de la tiroides por sonografía"), each its own
  singleton group. Never reached BestTakeResolver -- no group, no contest.
- **PIMPLES.** `clip_58f755fdf477281c1aad` (191.14-198.12s, micro-fragments "también me
  salían espinillas." / "era como un rush, una alergia."), `clip_a3f17c1603b8cafa3a13`
  (198.88-211.02s, the monolith "también me salían espinillas en esta parte... detrás
  de la oreja y todo el cuello... espinillas de personas con problemas hormonales."),
  `clip_aab224fd03b3a3b81c83` (213.34-222.98s, the later winner "otro síntoma era que
  me salían espinillas... detrás de la oreja y en el cuello. Me salía por
  temporadas.") -- each &lt;3s from its neighbor, each its own singleton. The monolith
  and later-winner share substantial vocabulary yet the lexical tier alone did not
  merge them either (its own reconciliation passes, not the semantic tier, decide
  this; exact lexical non-merge threshold not independently re-verified here).
- **HEREDITARY.** `clip_ea0192bf8dec7ef1e743` (295.52-313.50s, "Esta es mi experiencia.
  Soy la única en mi familia que tiene este tipo de cáncer...") IS correctly grouped
  (lexical tier) with `clip_2e84f4cb59dc3b5632a9` (340.18-346.52s, "cánceres son
  hereditarios. Soy la única en mi familia que tiene este tipo de cáncer.") -- near-
  verbatim phrase overlap, merged despite a 26.68s gap and `clip_0c625f24aae20d68152b`
  sitting temporally between them. `claim_coverage_best_take` then correctly promoted
  `clip_ea0192bf8dec7ef1e743` over `clip_2e84f4cb59dc3b5632a9` within that group
  (`reason: "single_candidate_covers_all_critical_claims_previous_winner_did_not"` --
  the short fragment was missing the hereditary-percentage claim sentences). But
  `clip_0c625f24aae20d68152b` (319.38-334.24s, "Soy la primera en mi familia con este
  tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar en la tiroides ni
  sufre de la tiroides... 5-10%...") -- only 5.88s from the winner and 5.94s from the
  discarded fragment -- was NEVER a member of this or any group. It shares almost no
  vocabulary with "Soy la única..." (different sentence structure throughout), so the
  lexical tier correctly could not have caught it; only the (inert) semantic tier
  could have. **Section 5 determination:** yes, this is a retry of the same
  underlying communicative intent (establish personal/family cancer-history context,
  cite the same 5-10% hereditary statistic, land the same caution/CTA), not two
  deliberately distinct beats -- the near-identical three-part scaffold (family-status
  claim -> statistic -> caution) recurring across both clips, with only the specific
  claim word ("única" vs "primera") and surrounding phrasing differing, is the
  signature of a creator naturally rephrasing the same take rather than composing two
  separate points. This is an interpretive judgment, not something the data proves
  outright -- flagged as such.

### First wrong decision -- classification

**G (another proven cause), and it precedes the A-F taxonomy's own first step**: the
IdeaClusterer's bounded semantic-equivalence tier was never invoked at all for this
run (config-gated `arbiter is None`), so no candidate pair for any of the three
regions was ever generated -- this is not a budget-displacement (C), not a false
negative from the arbiter itself (D), not a grouping-identity bug, and not a
downstream reintroduction (F). Every stage that DID run (lexical grouping,
BestTakeResolver/claim_coverage_best_take, StoryValidator, CompositeResolver hooks --
all confirmed empty/inert for these clips, ruling out F) operated correctly on the
inputs it was given.

### Compare against last-good RAW

The canonical RunPod Serverless RAW workflow (`cutsell-video00-raw-v5-auto-
microtrim.yml`, the workflow that produced `benchmarks/video00_selection_lock.json`'s
own baseline) does not use the base `EditDNA-Worker-2` template's env verbatim -- its
"Create unified Selection template" step explicitly overlays `CUTSELL_BRAIN_BACKEND:
"runpod_local"`, `CUTSELL_EDITORIAL_MODE:"clean_cut"`, `CUTSELL_ASR_MODEL:"medium"`,
`CUTSELL_HYBRID_LLM_ENABLED:"1"`, `CUTSELL_HYBRID_PROVIDER:"google"`, and
`CUTSELL_UNIFIED_SELECTION_REASONER:"1"` on top of the base template before every
live dispatch. `cutsell-video00-modal-raw.yml`'s "Build masked Modal env-secret file"
step passed the base template's raw `.env` straight through with no equivalent
overlay. The observed `semantic_idea_equivalence.status: "not_requested"` is fully
consistent with the base template's own default for `CUTSELL_HYBRID_LLM_ENABLED` (or
`CUTSELL_HYBRID_PROVIDER`) not being truthy/`"google"` -- exactly the condition the
RunPod workflow's own overlay exists to force. This is a **deterministic
infrastructure/config-parity gap** between the two workflows, not ASR jitter, not an
arbiter budget issue, and not a semantic-equivalence false negative -- category
"deterministic code regression" only in the sense that the Modal workflow's env
wiring, not `cutsell_worker`'s engine code, is what changed relative to the
baseline-producing configuration.

### Shared systemic cause

One shared cause explains all three regions: **the semantic-equivalence arbiter tier
of IdeaClusterer was completely inert for this run**, because `cutsell-video00-modal-
raw.yml` never applied the `CUTSELL_HYBRID_LLM_ENABLED=1` / `CUTSELL_HYBRID_PROVIDER=
google` overlay the canonical RunPod workflow applies before every live dispatch. All
three failures are lexically-dissimilar-or-borderline paraphrase pairs that the
lexical-only tier cannot be expected to catch by design -- exactly the class of case
the semantic tier exists for.

### Smallest general fix surface (NOT implemented -- reported for review only)

Add the same env overlay `cutsell-video00-modal-raw.yml`'s "Build masked Modal env-
secret file" step already has the mechanism to apply (it already builds the Modal
Secret from a Python dict merged from the base template) -- overlay
`CUTSELL_HYBRID_LLM_ENABLED=1` / `CUTSELL_HYBRID_PROVIDER=google` (matching the
RunPod workflow's own overlay) before constructing the Modal Secret, so both
execution backends run with parity semantic-arbiter configuration. This is an
infrastructure/workflow-config change, not a `cutsell_worker` engine-code change --
still requires the user's separate authorization per the standing directive before
any code is touched, and before any new Video00 RAW is dispatched to verify it. No
code was modified in this audit; no new benchmark was launched.

### D-044 fix confirmatory run (2026-09-02, head 1e13807, run 33648172326)

**The fix works.** Verified via the real (unmasked) result.json, not the CI log:
`diagnostics.semantic_idea_equivalence` = `{"status": "applied", "provider": "google",
"model": "gemini-3.5-flash-lite", "candidate_pair_count": 61, "checked_pair_count":
14, "merged_pair_count": 2}` -- the bounded semantic-equivalence arbiter is active.

**All 3 originally-audited regions now resolve correctly** (confirmed via
`validate_video00_regression_qa.py` run directly against the real result.json):
`sonography_bad_take_absent`, `pimples_bad_monolith_absent`, and
`family_context_preserved` all PASS. The arbiter's own `merges` list shows the exact
sonography pair merged (`"reason": "Different phrasings of unexpected thyroid test
results.", "confidence": 0.8`) and BestTakeResolver correctly kept the complete
retake over the incomplete one.

**Authoritative Human Gold regression QA: 14/18** (`qa_pass: false`, via the actual
validator -- not estimated). The 4 failures are NOT the 3 originally-audited
regions; they are two new root causes plus their cascading order-check failures:
- `papillary_cancer_preserved` fails because a DIFFERENT idea's claim-coverage
  BestTake override (`claim_coverage_best_take.overrides`, group `tg_839a860f59abd
  938a7`) picked a winner clip that doesn't fully token-cover the "Síntomas que
  tuve..." continuation the manifest expects contiguous with the papillary sentence;
  `final_story_coherence_validation.lost_semantic_atoms` flags this same discarded
  clip with `coverage_against_final_keep: 0.4` and `"blocking": true`.
  `sonography_good_before_diagnosis` (a required_order check whose 4th element is
  this same text) fails as a direct consequence.
- `pimples_micro_2_present` ("Era como un rush,") fails because this run's own
  ASR/attempt-reconstruction pass merged the micro-fragments together WITH the bad
  monolith into one physical attempt (different segmentation than the prior run),
  and the semantic arbiter correctly discarded that whole merged clip in favor of
  the later winner -- correctly per the merge decision, but the word "rush" does
  not recur in any surviving clip, so this specific manifest check (token-coverage
  based) has nothing left to match. `pimples_micro_order` fails as a direct
  consequence (missing its 2nd required element).
Both look like ASR/attempt-reconstruction non-determinism between runs rather than
anything caused by the env-parity fix itself -- the fix's own target regions
resolved exactly as predicted.

**Freeze: BLOCKED this run** (`freeze_blocked: true`, `FinalEditReviewer: FAIL`,
`repair_loop: NEEDS_HUMAN_REVIEW`) -- and this is the architecture working AS
DESIGNED, not a bug: `final_story_coherence_validation.missing_idea_coverage` shows
one entire retry family (`tg_28298998766ee0c8f1`, both member clips discarded, ZERO
winning_clip_ids) vanished with no surviving realization -- exactly the "idea
coverage must not silently vanish" invariant firing correctly, combined with the
same blocking lost_semantic_atom above. `validate_video00_architecture.py` confirms
0 failed checks (`architecture_verified: true`) -- every stage, including
`no_render_attempted_on_a_blocked_semantic_plan`, behaved exactly as the contract
requires. Render was never attempted (`live_render_qc.reason:
"freeze_blocked_no_render"`); `deliverable: false`; `delivery_status:
"NOT_DELIVERABLE_not_attempted"`. `lost_critical_claims: []` -- nothing required was
lost outright, only under-covered.

Selected count: 21 (vs. the frozen lock's 23-count expectation, reported as a
warning per D-032, not a failure). Modal GPU runtime: ~343s wall time for the
benchmark step (faster than the pre-fix run, plausibly because no render was
attempted this time). Approximate cost: low cents (one L4, under 6 minutes).

**Per the standing directive: STOPPING here.** No further benchmark launched, no
engine code modified. The newly-surfaced missing-idea-coverage gap and the two new
regression-qa failures are reported for review, not acted on unilaterally -- a
plausible next audit target if the user wants to pursue it, but out of this
directive's scope (which named only the sonography/pimples/hereditary regions, all
three now confirmed resolved).

## D-045 -- missing idea coverage + physical micro-fragment forensic audit (report only, no code changes)

Forensic-only follow-up to D-044's confirmatory result (benchmark
`video00-modal-33648172326-1`), using the same read-only, zero-GPU
`cutsell-video00-d044-forensic-extract.yml` workflow extended with a new
general `trace_clip_ids()` search (added in this cycle, tooling only --
recursively finds every diagnostics path mentioning a given clip_id across
the ENTIRE raw `result.json`, not just the curated subset the D-044
extractor already pulled). No `cutsell_worker` engine code was read as
speculation -- every claim below is a direct quote from the real,
unmasked diagnostics of that one already-produced result.

### CASE A -- "missing idea coverage" is a false positive, not a content loss

Idea `tg_28298998766ee0c8f1` (members `clip_a6a6f4d1cffd6c94115a`,
`clip_42b0b7919d9f9d025e86`) is the "Al terminar mi contrato... hablé/cambié
de ginecóloga... me mandó a hacer sonografías" retry family. Its
`take_judge_groups[1]` entry shows the D-044 semantic-equivalence arbiter
working correctly: `semantic_candidates` labels `clip_a6a6f4d1cffd6c94115a`
"alternate" (confidence 0.85) and `clip_42b0b7919d9f9d025e86` "winner"
(confidence 0.95); `semantic_override_applied: true` correctly overrides the
narrower baseline watch/listen ranking (0.6855 vs 0.6594, which favored the
other clip) in favor of the semantically-fuller take. `selected_clip_id` /
`semantic_preferred_clip_id` both resolve to `clip_42b0b7919d9f9d025e86` --
the semantic decision itself is correct.

`clip_42b0b7919d9f9d025e86` (95.58-107.48s) is then legitimately split by a
post-selection physical hook (`post_selection_interior_gap_trim`, minting
`__psigl`/`__psigr`-suffixed fragment ids per the D-039 fragment-identity
work) into two fragments -- both of which **are present in the final
`draft.selected` list** (`clip_42b0b7919d9f9d025e86__psiglcad3722cd281` and
`clip_42b0b7919d9f9d025e86__psigr1015a2ec8b00`, `selected[6]`/`selected[7]`
in this run). The winning realization's content was never lost.

`canonical_edit_plan.py`'s own `build_canonical_edit_plan` derives each
idea's `winning_clip_ids`/`discarded_clip_ids` by exact string-equality
between a `take_judge_groups` member's *original* clip_id and
`draft.selected`'s clip_ids. Because the winning clip's id was rewritten
into two fragment-suffixed ids by the post-selection split, the exact-id
check finds neither the original id nor either fragment, and wrongly buckets
`clip_42b0b7919d9f9d025e86` as **discarded** alongside its genuinely-losing
sibling -- producing `winning_clip_ids: []`, `coverage_status: "missing"` for
an idea that in fact has a complete, undamaged winning realization on
screen. `final_story_coherence_validation.missing_idea_coverage` and the one
BLOCKING `lost_semantic_atom` are downstream of this same false reading, and
correctly (given that false input) block Selection Freeze.

**First wrong decision:** `canonical_edit_plan.py`'s winning/discarded
derivation, not any semantic, clustering, or claim-coverage stage --
it is not fragment-provenance-aware (the exact linkage it would need,
`post_selection_interior_gap_trace[*].parent_clip_id`, already exists in
diagnostics; the function just never consults it).

**Classification: deterministic**, not ASR-jitter-sensitive -- this will
reproduce on any run where a post-selection physical split touches a lone,
already-selected winner of a retry family, independent of ASR variance.

**Smallest general fix surface (not implemented):** teach
`build_canonical_edit_plan`'s winning/discarded check to also treat a
`take_judge_groups` member as "present in selected" when any selected
clip's recorded `parent_clip_id` (or an equivalent fragment-provenance
link already produced by the splitting hook) equals that member's id --
a change local to one function's membership test, touching no semantic,
clustering, or scoring logic.

**Correction to the D-044 confirmatory report's own framing:** that report
linked `papillary_cancer_preserved`/`sonography_good_before_diagnosis`
failing to this same idea's "missing coverage." The deeper trace shows
these are in fact independent: that check's exact/ordered text belongs to a
*different* idea, `tg_839a860f59abd938a7` ("sintomas que tuve"), whose
`coverage_status` is `"complete"` -- it already has a winner
(`clip_ece4915a647661808c9b`). But the discarded sibling of that same idea,
`clip_5a99d26352df1219f3d7` ("Sintomas que tuve. Segun yo, era sintomatica,
pero si hubo indicios ahora mirandose atras."), is a near-verbatim match to
the golden fixture text, while the winner's phrasing ("Sintomas que no me
parecian sospechosos...") is a paraphrase that does not satisfy the
`required_exact`/`required_order` checks. This is a genuine take-selection
outcome (the higher-scoring take does not happen to match the fixed golden
wording), not a coverage-loss bug, and not something this directive asked
to be root-caused further -- flagged here only so the two failures are not
conflated going forward.

### CASE B -- "Era como un rush" micro-fragment: physical fusion, not semantic-equivalence

Idea `tg_c80832d2c77c9ff630` (members `clip_deddd53436a5861761bf`,
`clip_557299153657e9c6faf9`) is the pimples retry family.
`take_judge_groups[3]`'s `semantic_candidates` labels **both** members
"winner" at confidence 0.95 (`semantic_override_applied: false`,
`semantic_preferred_clip_id: null`) -- the arbiter correctly saw them as two
independently-valid deliveries of the same idea and deferred to the
baseline watch/listen ranking, which picked `clip_557299153657e9c6faf9`
over `clip_deddd53436a5861761bf` by a narrow margin (0.6843 vs 0.649). This
selection step behaved exactly as designed.

The problem is upstream, at `AttemptReconstructor`. Comparing against the
last run where `pimples_micro_2_present` passed (pre-D-044-fix run
`video00-modal-33636255124-1`, same source audio): that run produced
**three separate physical attempts** covering this region --
`clip_58f755fdf477281c1aad` (191.14-198.12s, text "...tambien me salian
espinillas. Era como un rush, una alergia."), `clip_a3f17c1603b8cafa3a13`
(198.88-211.02s, the bad monolith wording), and
`clip_aab224fd03b3a3b81c83` (213.34-222.98s). The desired micro-fragment
"Era como un rush, una alergia." lived in its own independently-selectable
attempt, physically separate from the bad monolith, with a 0.76s gap
between them (198.12 to 198.88s).

In the confirmatory run, `AttemptReconstructor` fused essentially the same
audio span (192.36-210.62s -- matching the earlier run's two attempts'
combined range within ~1.2s at each edge) into **one single attempt**,
`clip_deddd53436a5861761bf`, whose text concatenates the good micro-fragment
and the bad monolith wording together. No code in `AttemptReconstructor`
or its merge threshold changed between these two runs (D-044's fix touched
only two Modal env vars feeding the semantic-equivalence arbiter, unrelated
to attempt reconstruction). An internal ~0.76s silence gap sitting right at
the boundary of whatever gap-duration threshold governs attempt merging is
the most consistent explanation for two runs of the same source audio
producing different attempt boundaries here.

**First wrong decision:** `AttemptReconstructor`'s attempt-merge boundary
call across the ~198s gap, which fused two semantically-independent
utterances (a clean micro-claim and a separately-retried monolith) into one
physical unit before any take-grouping or scoring ever saw them
separately. Once fused, `BestTakeResolver`/`DeliveryScorer` had no
mechanism to recover the good subspan from a monolith that lost the
best-take contest as a whole -- the entire physical unit is atomic by the
time scoring runs.

**Classification: A (ASR word-timing jitter) at a B-adjacent
(AttemptReconstructor merge-threshold) boundary condition** -- a ~0.76s gap
is a plausible run-to-run jitter range for GPU-decoded ASR timestamps, and
sits close enough to a merge threshold that a small shift could flip the
decision; this reads as run-to-run sensitivity at a borderline value, not a
logic regression from a code change (none touched this path).

**Smallest general fix surface (not implemented):** this is squarely
General Invariant 2 (good subspan preservation) as the user framed it --
not a semantic-equivalence fix. The narrowest fix surface sits at or after
`AttemptReconstructor`: either a more conservative/hysteresis-aware merge
gap threshold, or preserving sub-span candidate boundaries inside a merged
attempt so a losing physical monolith's independently-meaningful subspans
remain separately selectable by `BestTakeResolver` even after ASR-level
merging treats them as one transcript unit.

### Shared systemic cause: no -- but a shared category

Case A (a post-selection *split* not recognized by downstream
id-equality bookkeeping) and Case B (a pre-selection *merge* fusing two
independent utterances before scoring) are different code paths --
`canonical_edit_plan.py`'s membership derivation vs.
`AttemptReconstructor`'s boundary/merge logic -- and neither's fix implies
the other's. They do share one general theme, matching both invariants
the user asked to be evaluated (not implemented): whenever a clip's
physical boundary is *mutated* -- split (Case A) or merged (Case B) --
relative to the identity a later semantic/coverage-bookkeeping stage
expects, that stage can silently misjudge what actually survived. Neither
invariant is implemented by this directive; both are confirmed here as
real, distinct gaps worth closing independently.

No code changed. No new Modal RAW launched. Reported for review per the
standing directive.

## D-046 -- implement the two D-045 root-cause fixes separately

Both D-045 root causes implemented, tested, and validated offline. No Modal/
RunPod infrastructure touched, D-044's hybrid-semantic config untouched,
IdeaClusterer untouched, render/QC untouched. No new Modal RAW launched --
per the standing directive, this stops for review before one.

### FIX A -- fragment-provenance-aware CanonicalEditPlan

Root cause (D-045 Case A): `post_selection_interior_gap_trim.py` can split
an already-SELECTED winning realization into two physical pieces, but
never stamped D-036's existing `parent_semantic_clip_id` provenance field
on them (the same field `human_boundary_polish_v5.py` already uses for
exactly this purpose). `canonical_edit_plan.py`'s winning/discarded
derivation and `final_story_coherence_validation.py`'s
`_missing_idea_coverage` both determine "did this take_judge_groups member
survive?" via exact `clip_id` equality against `draft.selected` -- with no
provenance link, neither could find the split winner anywhere and both
wrongly reported the idea as vanished (`coverage_status: "missing"`,
`missing_idea_coverage`), falsely blocking Selection Freeze.

Fix (three files, all additive):
- `post_selection_interior_gap_trim.py`: split pieces now carry
  `parent_semantic_clip_id` (root-tracked through chained splits, mirroring
  `human_boundary_polish_v5.py`'s own `root_parent` pattern), plus
  `render_fragment_id`, `boundary_reason`, and `fragment_index`/
  `fragment_count` for full D-036 contract parity.
- `canonical_edit_plan.py`: a `take_judge_groups` member now counts as
  surviving when its `clip_id` is in `draft.selected` OR any selected
  clip's `effective_parent_semantic_clip_id` names it.
- `final_story_coherence_validation.py`: `_missing_idea_coverage` uses the
  same provenance-aware check.

No Video00 special-casing; the general D-036 provenance contract already
in `contracts.py` is the only mechanism used. Exact clip identity semantics
are unchanged everywhere else (a fragment of a genuinely-discarded clip
that never made it into `draft.selected` cannot revive it -- only
`draft.selected` is consulted).

Tests: 10 required categories in `tests/test_cutsell_d046_fix_a_fragment_
provenance.py` (unsplit winner, one-fragment split, multi-fragment split
counted once, discarded sibling stays discarded, a discarded clip's own
fragment cannot revive it, provenance survives a second chained split,
mixed original+fragment doesn't double-count, coverage_status stays
complete, `missing_idea_coverage` empties out, an unrelated genuinely-
missing idea still blocks Freeze) plus a literal regression-lock test
replaying the exact real clip_ids/group_id from the D-045 Case A incident
(`tg_28298998766ee0c8f1`, `clip_42b0b7919d9f9d025e86` and its real
`__psigl`/`__psigr` fragment ids) -- 11 tests total, all passing. 3 more
tests added directly to `post_selection_interior_gap_trim.py`'s own suite
proving the provenance stamping itself (11 tests total there, up from 8).
1 new CleanCutBench fixture (52) reaches the same false positive through
the REAL take-grouping/idea-equivalence/take-judge/coherence chain with
generic (non-Video00) vocabulary.

### FIX B -- good subspan preservation under a borderline attempt merge

Root cause (D-045 Case B): `AttemptReconstructor` merges two ASR-level
candidates into one physical attempt whenever the gap between them is
below `max_continuation_gap_sec` (1.20s) and no stronger boundary signal
fires. The observed regression gap was ~0.76s -- comfortably inside the
merge zone -- fusing an independently-usable good micro-fragment ("Era
como un rush, una alergia.") into a larger monolith that later correctly
lost its Best Take contest, destroying the good subspan along with it.

Design choice (both options from the directive were evaluated): tightening
the boundary rule itself (Option A -- e.g. treating any gap where both
sides are already `complete_idea` as a new hard boundary) was REJECTED --
`complete_idea` is deliberately permissive for ordinary sentence-length
speech, so that rule would re-split large amounts of currently-correct
continuous multi-sentence delivery across every Video project, a global
fragment explosion rather than a targeted fix. Implemented Option B
instead: additive-only subspan preservation.

`attempt_reconstruction.py` now also computes, per merged bucket, the
single widest internal gap that is a real pause (>= 0.55s, comfortably
below the 1.20s merge threshold) between two members BOTH already
independently `complete_idea`, with each side carrying enough of its own
content (>= 3 content tokens) to be worth preserving. When found, both
sides are reconstructed as their own standalone candidates -- capped at
one split per bucket, so a bucket with several ordinary internal pauses
cannot fragment-explode. The merge decision itself, and the fused attempt
it always still produces, are completely unchanged.

These extra candidates are deliberately NOT mixed into
`reconstruct_delivery_attempts`'s own `attempts` return value --
`attempt_boundary_integrity.py`'s pre-existing terminal-connector-guard
monkeypatch indexes `attempts[-1]`/`attempts[-2]` assuming exactly one
entry per merged bucket, and mixing extras in would silently corrupt that.
Instead, a new `preserved_subspan_candidates(takes, diagnostics)` function
recomputes the same buckets from the diagnostics `reconstruct_delivery_
attempts` already produced and is called explicitly by `flow_b.py` right
after reconstruction, appending its result to the live candidate pool.
Nothing downstream (IdeaClusterer, DeliveryScorer, ClaimCoverageBestTake)
was modified -- the extra candidates are just more input to the existing
"multiple realizations of one idea, pick the winner(s)" machinery those
stages already implement.

Tests: `tests/test_cutsell_d046_fix_b_subspan_preservation.py`, 13 tests --
the independent micro-fragment surviving a borderline merge; just-below/
at/just-above the 0.55s preservation floor; +/-50ms, +/-100ms, and +/-250ms
jitter around the observed 0.76s gap (the last straddles the floor by
design, proving the boundary is where intended); determinism across 5
small shifts within the preserve zone; a genuine incomplete continuation
correctly gets nothing extra; a gap already at/above the merge threshold
needs no preservation (both sides already independent); no fragment
explosion across a 4-member bucket with 3 internal gaps (still only 2
extra candidates); preserved candidates are never duplicates of the fused
attempt or each other; and an unsplit pool preserves nothing. 3 existing
attempt_reconstruction/attempt_reconstruction_suffix tests re-run
unaffected (byte-identical `attempts` return value confirmed).

The real texts from the D-045 incident were deliberately NOT used verbatim
in a regression-lock test here (unlike FIX A): checked directly, the exact
real wording trips the PRE-EXISTING, unrelated `_restart_evidence`
("lexical_restart") rule when modeled as two adjacent CandidateTakes,
because the real fusion happened across many smaller ASR-level fragments
this two-take simplification can't reproduce byte-for-byte. The synthetic
fixtures instead use the same 0.76s gap value and equivalent (not
identical) phrasing, isolating the new mechanism cleanly.

### Validation

- `python3 -m compileall -q cutsell_worker tests`: clean.
- `pytest -q tests/test_cutsell_*.py` (the canonical CI glob): **1294
  passed**, 0 failed.
- Offline replay proves both cases: the D-045 Case A regression-lock test
  (exact real clip_ids) shows `missing_idea_coverage == []` and
  `coverage_status: "complete"` where the live run had shown `"missing"`;
  the D-046 FIX B tests show the equivalent-shaped micro-fragment surviving
  as an extra candidate at the exact observed 0.76s gap and through +/-250ms
  of jitter around it.
- No behavior change detected outside the target conditions: every
  pre-existing test in the modified files (`post_selection_interior_gap_
  trim.py`'s own 8, `attempt_reconstruction.py`'s 9,
  `canonical_edit_plan.py`/`final_story_coherence_validation.py`'s 34, the
  full 51-fixture CleanCutBench suite, and the full 1294-test canonical
  glob) still passes unchanged.

No new Modal RAW launched. Stopping here for review, per the standing
directive.

## D-046 confirmatory live result -- one Modal Video00 RAW at head 5e23f86

Live confirmatory run (`video00-modal-33669148915-1`, HEAD `5e23f86`, D-044
hybrid-semantic overlay confirmed active, `retries=0`, same source video,
same 18-check manifest). Verified via the real, unmasked `result.json`
(read-only zero-GPU forensic-extract workflow), not the masked CI log.

**D-046 CASE A: PASS (mechanism not exercised, no adverse effect).**
`clip_fb136663d61593a59463` ("El año pasado comencé a notar hinchazón...")
was split by `post_selection_interior_gap_trim` into
`clip_fb136663d61593a59463__psigl11279a14d51e` /
`__psigr2208741ba22d`, both present in `selected`. This idea had no
competing retry this run (a take_group_members singleton), so
`canonical_edit_plan.py`'s provenance-aware rescue was never actually
NEEDED here -- but its stamping ran cleanly with no side effects, and
`missing_idea_coverage: []` confirms nothing broke. Neither of the two
D-045 Case A/B failure geometries reproduced live this run (consistent
with both being ASR/attempt-boundary run-to-run non-determinism, not a
deterministic condition) -- FIX A's rescue branch and FIX B's
`preserved_borderline_subspans` (empty in diagnostics) both went unused
by coincidence, not by malfunction.

**D-046 CASE B: PASS.** `pimples_micro_2_present` now passes (was FAILing
in the D-044 confirmatory run). `attempt_reconstruction.attempts` shows
the pimples region reproduced the SAME (good, non-fused) physical
boundaries as the original pre-D-044-fix passing run: the micro-fragment
(`clip_4ad06de981a449a08cc2`, 191.14-198.12s) and the bad monolith
(`clip_acd1df1b0e7e24a393e4`, 198.88-211.02s) are two separate attempts
again this run -- byte-identical timing to the historical passing run.
Direct further confirmation that this is real run-to-run ASR/attempt-
boundary variance, not a deterministic code regression.

**NEW finding (not D-046, not touched, reported only):** Human Gold
16/18, two NEW failures unrelated to D-045 Case A/B --
`pimples_bad_monolith_absent` (the bad monolith text IS present in
`selected`) and `gastritis_preserved`. Root cause for the first:
`semantic_idea_equivalence.distinct_addition_blocked` shows the arbiter
correctly identified `clip_acd1df1b0e7e24a393e4` (bad monolith) and
`clip_b46d4134c9be18017964` ("Otro síntoma era que me salían
espinillas...") as the same idea (confidence 0.95) but the take-grouping
distinct-addition guard (`take_grouping_provider.py`, IdeaClusterer
territory) blocked the merge anyway -- the same "otro síntoma..." pattern
this guard was built to protect (CleanCutBench fixture 50, RAW
33432104336) is here producing a false positive: "otro síntoma" is a
narrative bridge into a RETRY of the same point, not a genuinely new
fact. Because the two were never merged into one retry family, the bad
monolith was never put in competition with the clean winner and survived
unchallenged. Freeze correctly BLOCKED anyway (a real, separate
contradiction finding plus two blocking lost_semantic_atoms, both
pre-existing/unrelated territory). Architecture PASS (0 failed checks).
Per the standing directive this is IdeaClusterer territory -- not
diagnosed further and not patched.

**Full result:** Human Gold 16/18, architecture PASS, CanonicalEditPlan
built (`plan_83d59af0e7f4ea82`), FinalEditReviewer FAIL (expected --
mirrors the correct freeze block), Freeze BLOCKED, render never attempted
(`no_render_attempted_on_a_blocked_semantic_plan: ok`). Modal GPU runtime
~6m14s (374s), one L4, low-cost. Selection lock: 20 vs the frozen 23-count
expectation (warning only, per D-032).

No code changed based on this result. No further RAW launched. Reported
for review per the standing directive.

## D-048 -- implement the two D-047 root-cause fixes

Both D-047 root causes implemented, tested, and validated offline. No
Modal/RunPod infrastructure touched, D-046 untouched, render/QC untouched.
No new Modal RAW launched -- per the standing directive, this stops for
review before one.

### FIX 1 -- content-divergence-gated distinct-addition guard

Root cause (D-047 Case 1): `take_grouping_provider.py`'s D-039 guard
blocked a semantic merge whenever exactly one side carried a "new/
additional item" discourse marker, with no check of whether the marked
side's content actually diverged from the other side's. The arbiter
confirmed the pimples monolith and its "Otro sintoma..." retry as the same
idea at 0.95 confidence, sharing the same specific symptom AND location
("detras de la oreja"/"cuello") -- the marker was a narrative restart, not
evidence of a genuinely new point.

Fix: a new `_marked_side_diverges_in_content` check gates the block. Both
candidates' content vocabulary (marker/connector/stopword-stripped, via a
small local bag-of-words helper mirroring `final_sibling_grouping._content`
-- kept local to avoid a circular import, since that module imports FROM
this one) is compared; the block only fires when shared content is thin
(< 6 tokens, or < 55% coverage of the shorter side's own vocabulary).
Thresholds derived directly from the two calibration shapes named in the
directive: the founding D-039 incident (arm vs leg mentions) shares only 3
content tokens; the D-047 Case 1 false positive shares 9, including the
sentence's own distinguishing nouns. Arbiter confidence is deliberately
NOT part of the decision either way (supporting evidence only, per the
directive). Marker vocabulary itself is unchanged -- "tambien"/"also" were
evaluated and explicitly left out: even content-gated, they are common
enough ordinary connectors to flag far more pairs than intended (confirmed
by an existing test regression during implementation); "on top of that"/
"an additional"/"one more thing" already cover the same "additive
framing" category without that collision risk.

Tests: 12 required categories in `tests/test_cutsell_d048_fix1_distinct_
addition_guard.py` (founding case blocked, D-047 Case 1 shape unblocked,
English "another symptom" both shapes, additive-framing both shapes, both/
neither side marked unaffected, high-topical-but-different-entity blocked,
strong-overlap+high-confidence unblocked, weak-overlap+high-confidence
still blocked, lexical-tier-only regression check) -- 12/12 passing. All
65 pre-existing take-grouping/semantic-equivalence tests re-run unaffected
after one necessary wording fix to an existing test whose synthetic
placeholder text happened to accidentally collide with the founding
incident's own generic framing.

### FIX 2 -- claim-criticality gate for the ClaimCoverageBestTake override

Root cause (D-047 Case 2): a retry family's only extracted CRITICAL claim
was a bare negation riding on an incidental year inside an ordinary
temporal aside ("... en una temporada, en 2023, no hay que preguntar."),
source-exclusive to the one thin candidate containing that exact wording.
Because that candidate trivially "covered every critical claim" (the set
had exactly one member: its own), `claim_coverage_best_take.py` swapped it
in as winner over an already-correct, substantively richer realization
carrying the actual diagnosis/treatment content -- neither sibling
registered a CRITICAL claim of its own (a separate, pre-existing marker-
matching gap in `classify_claim`, not touched here), making the group's
critical-claim set collapse to exactly the one incidental claim.

Fix: does NOT change `classify_claim`'s own importance labels or
StoryValidator's freeze-blocking posture (a negation still always blocks
Freeze there, unchanged -- a deliberate, separate "WHEN UNCERTAIN, KEEP"
backstop). Adds a second, narrower gate scoped to this module's own
override decision -- `_is_low_information_incidental` (no independently
substantive marker: diagnosis/identification, correction, a genuinely
unit/percent/currency/dose-qualified number, cause-effect connector,
unique-conclusion statistic, or state-result language; its only CRITICAL
signal is a bare negation/number riding on a recognizable temporal-aside
shape -- mirrors `classify_number_atom`'s own D-031 CONTEXTUAL rule for a
bare year, applied to the whole claim) -- combined with a source-
exclusivity check (no OTHER sibling covers the claim either) and a
richer-content check (the candidate is not otherwise richer than the
current winner). All three must hold to suppress; a single substantive,
non-source-exclusive, or genuinely-richer-candidate claim keeps the
override eligible exactly as before. The same gate is also applied to the
2-piece composite path (`_unique_contribution_is_incidental`), which would
otherwise pull the same incidental claim in via a different mechanism.
Suppressed overrides are recorded in a new
`diagnostics["claim_coverage_best_take"]["suppressed_incidental_overrides"]`
list for observability.

Tests: 15 in `tests/test_cutsell_d048_fix2_claim_criticality_gate.py` (14
required categories -- contextual year/date, incidental temporal aside,
filler/self-referential aside all correctly non-override-forcing; unique
diagnosis/negation/causal/treatment/family-hereditary claims all correctly
override-forcing; contextual-claim-with-richer-winner preserved; multiple-
critical-claims-better-covered-by-alternate still overrides; source-
exclusive-but-critical remains eligible; source-exclusive-and-contextual
is not winner-forcing; no override when it would lose more content than it
restores; untouched ranking when no critical gap exists at all -- plus a
literal regression-lock replay of the D-047 Case 2 incident's real clip
texts and group id) -- 15/15 passing. All 69 pre-existing semantic-claims/
claim-coverage tests re-run unaffected.

### Real D-047 regression fixtures (CleanCutBench, real chain)

Two fixtures added to `test_cutsell_clean_cut_core_evaluation_suite.py`
(53, 54), reaching each false positive through the REAL take-grouping/
idea-equivalence/take-judge/claim-coverage chain with generic (non-Video00)
vocabulary:
- 53: a high-specific-content-overlap "Otro sintoma..." retry (facial/hand
  swelling) merges into one winning realization instead of blocking.
- 54: a richer diagnosis/treatment realization survives a claim-coverage
  self-source trap (a vague, source-exclusive, incidental-temporal-aside
  candidate) through the full chain, with the suppression recorded in
  diagnostics.

### Validation

- `python3 -m compileall -q cutsell_worker tests`: clean.
- `pytest -q tests/test_cutsell_*.py` (the canonical CI glob): **1324
  passed**, 0 failed (up from 1295 pre-D-048; 29 new tests: 12 + 15 + 2
  CleanCutBench fixtures).
- Offline replay proves both required results: the PIMPLES-equivalent
  fixture (53 / the FIX 1 test suite's own D-047-shaped case) shows the
  semantic merge no longer blocked by the distinct-addition guard when
  specific content strongly overlaps; the GASTRITIS-equivalent fixture (54
  / the FIX 2 regression-lock test with the real clip texts) shows the
  correct richer BestTake winner remains winner and the contextual
  source-exclusive aside cannot force an override.
- No behavior change detected outside the target conditions: every
  pre-existing test in every modified file still passes unchanged, after
  one necessary wording fix to a pre-existing FIX-1-adjacent test whose
  synthetic text incidentally collided with the founding D-039 shape (not
  a real Video00 fixture, no production behavior implication).

No new Modal RAW launched. Stopping here for review, per the standing
directive.

## D-048 confirmatory Modal Video00 RAW

Dispatched at fixed HEAD `1f8f57c` (benchmark_id
`video00-modal-33676615917-1`). Pre-run verification confirmed: L4 GPU,
`retries=0`, D-044 hybrid-semantic overlay
(`CUTSELL_HYBRID_LLM_ENABLED=1`/`CUTSELL_HYBRID_PROVIDER=google`) intact,
same `SOURCE_KEY`, Human Gold manifest unmodified since D-032.

- **D-048 PIMPLES: PASS** -- `pimples_bad_monolith_absent` in
  `passed_checks`.
- **D-048 GASTRITIS: PASS** -- `gastritis_preserved` in `passed_checks`;
  `suppressed_incidental_overrides: []` (FIX 2's rescue mechanism was not
  exercised this run -- the vulnerable geometry didn't recur -- but no
  regression either).
- **Human Gold: 15/18** (`benchmarks/validate_video00_selection_lock.py`'s
  embedded `historical_regression_qa`, run inside the CI job itself --
  authoritative, not estimated). Failed: `papillary_cancer_preserved`,
  `family_context_preserved`, `sonography_good_before_diagnosis` -- a
  DIFFERENT set of 3 failures than the 2 D-048 targeted (which both now
  pass), consistent with known run-to-run ASR/pipeline non-determinism, not
  a D-048 regression.
- **Architecture: PASS** (`architecture_verified: true`, 0 failed checks).
  `CanonicalEditPlan: PASS` (`plan_c4b2dcea26bd07cb`, v1).
  `FinalEditReviewer: FAIL` (blocking findings; see D-049 below).
  `Freeze: BLOCKED` (`coverage_ledger_story_validator.freeze_blocked=true`).
  `Render: NO` (`live_render_qc.status: "not_attempted"` -- correctly
  skipped since Freeze was blocked). `PostRender QC: NOT_REACHED`.

Investigated via two zero-GPU `cutsell-video00-d044-forensic-extract.yml`
dispatches against this run's own `result.json` (no RAW). Findings written
up as D-049 below.

## D-049 -- final 15/18 forensic audit (papillary/symptom-transition +
family/hereditary)

Report-only, no code changes, no RAW. Full forensic trace via two
zero-GPU forensic-extract dispatches against `video00-modal-33676615917-1`
(`clip_trace`) and one comparison dispatch against the last-passing run
`video00-modal-33669148915-1` (D-046 confirmatory, 16/18).

**Case A (papillary_cancer_preserved + sonography_good_before_diagnosis)**
-- ONE shared root cause. `diagnostics.hybrid_editorial_chunks[2].decisions[4]`
authoritatively deleted `clip_9b4436e683ddafba3255` ("Síntomas que tuve.
Según yo, era sintomática, pero sí hubo indicios ahora mirándose atrás.")
as a "failed" take (`delete_basis: "semantic_failed_plus_local_performance"`,
`local_failure_reasons: ["dense_physical_reset:6"]`) with
`later_retry_replacement_id: null` and `later_retry_semantic_overlap: 0.0`
-- i.e. deleted with no verified replacement, unlike a sibling in the same
chunk (`clip_184d5f8d669481af48d6`) which carried its own local-failure
flag but was correctly downgraded to `"kept_fail_open"`. This one deleted
clip is the sole realization of a Gold text required BOTH directly by
`papillary_cancer_preserved` AND as the last element of
`sonography_good_before_diagnosis`'s 4-part ordered sequence (the two
checks share the identical required string in
`benchmarks/video00_regression_qa.json`) -- one missing segment, two failed
checks. Confirmed against the last-passing run: both checks passed there
with different ASR clip ids; the specific candidate never tripped
`dense_physical_reset` that run. Classification: ASR/jitter-triggered
exposure of a deterministic code gap (missing replacement-verification
before an authoritative delete). Smallest general fix (not yet
implemented): require `later_retry_replacement_id` non-null and
`later_retry_semantic_overlap` above a minimum (or a bag-of-words
content-overlap check, mirroring D-048 FIX 1's
`_marked_side_diverges_in_content`) before `applied_delete=true`;
otherwise downgrade to `"kept_fail_open"`.

Side finding (does not trip any of the 3 failing checks, out of D-049's
scope): `semantic_idea_equivalence.merges` shows `clip_bb8b91e25dc2e56b9b03`
("Me hice un test a bordo... mi metabolismo...") merged at confidence 0.85
into the gynecologist-referral idea with reason "Both describe requesting
comprehensive tests from a gynecologist" -- a reason that doesn't match
bb8b91's actual content. This let an unrelated-but-cleanly-delivered clip
register as that idea's "winning" realization while the real referral
content (`clip_184d5f8d669481af48d6` et al.) was discarded, even though
`canonical_edit_plan_ideas` bookkeeping shows `coverage_status: "complete"`.
No current Gold check covers that exact sentence, so it doesn't explain
any of the 3 failures -- logged for a future directive, not fixed here.

**Case B (family_context_preserved)** -- independent root cause in a
different pipeline stage. `take_judge_groups[3]` (group
`tg_467dddcb681caf780c`, after a correct semantic merge of
`clip_00badced576948b11b07` + `clip_35b7c1847d8f7c271172` at confidence
0.9) ranked the short single-sentence `clip_00badced576948b11b07` ahead of
the full realization `clip_3788d39998c51605dcde` (score 0.6363) on
`watch_listen_baseline` delivery scoring alone (no content/claim signal).
`ClaimCoverageBestTake` correctly detected the 0-of-3-critical-claims gap
but reported `unresolved_gaps` with reason
`"no_single_or_paired_candidate_safely_covers_every_critical_claim"`:
`clip_3788d399` alone covers 2 of 3 critical claims (the `NEGATION` +
one "5-10%" `MEASUREMENT_QUANTITY` claim); the 3rd claim is
`clip_35b7c1847`'s own independent, mid-sentence-cutoff restatement of
the SAME "5-10%" fact in different words. `semantic_claims.py` never
dedupes near-identical restatements of one fact across two
already-semantically-merged retry-family members, inflating the group's
true 2-fact requirement to 3 -- no single candidate can satisfy it, and
the 2-piece composite correctly refuses to pair `3788d399`+`35b7c1847`
(D-048's same-claim-type collision guard working as designed, since both
remaining claims are `MEASUREMENT_QUANTITY`). Classification against the
6 listed options: primarily (B) ClaimCoverage failed to override, rooted
in a claim-deduplication gap; (A) is a contributing factor (BestTake
scoring has no claim-richness signal); (C)/(D)/(E) not confirmed --
composite, negation, and grouping logic all behaved correctly. Confirmed
against the last-passing run: `family_context_preserved` passed there
(16/18) with different ASR clip ids -- the 2-clip split of the "5-10%"
statistic didn't occur that run. Classification: ASR/jitter-triggered
exposure of a deterministic code gap (missing cross-member claim dedup).
Smallest general fix (not yet implemented): dedupe claims across a retry
family's own members in `claim_coverage_best_take.py`'s critical-claims
aggregation, using the same bag-of-words/content-overlap pattern (D-048
FIX 1's `_content_tokens`), before requiring separate coverage for each.

**Why 3 failed checks = 2 root causes:** confirmed --
`sonography_good_before_diagnosis`'s required sequence literally repeats
`papillary_cancer_preserved`'s own required text as its last element, so
both trace to the identical missing segment (Case A). `family_context_preserved`
is structurally independent (Case B, a different pipeline stage and idea).
Neither cause was introduced by D-048 (which never touched
`hybrid_editorial_chunks` or claim deduplication) -- both are pre-existing
gaps this run's particular ASR/vision segmentation happened to trigger.

No code changes made. No new RAW launched. Holding for review before any
FIX-3/FIX-4 directive.

## D-050 -- engine architecture consolidation audit (report-only)

Full audit delivered in chat (execution graph traced from real source for
every stage of `run_op("focused", ...)`; duplicated-authority findings for
"who decides the winner"/"what is one idea"/"who may delete content"/"who
proves a composite is safe"/"is an idea covered"; a proposed canonical
identity model, semantic ledger, 6-layer authority hierarchy, ASR-jitter
strategy, D-037-through-D-049 rule classification, benchmark expansion
plan, Semantic Selection Stability metric, and a staged D-050A-F refactor
plan). No code changes, no RAW. Authorized to proceed with D-050A only.

## D-050A -- canonical identity/provenance foundation (additive-only)

Implements the identity/provenance layer D-050's audit proposed, as pure
shadow metadata: nothing in the active pipeline reads any id below to make
an editorial decision yet -- confirmed by running the full pre-existing
`tests/test_cutsell_*.py` glob (1324 tests, CleanCutBench's 54 real-chain
fixtures included) unmodified and green both before and after this change.

**Identity model** (all additive, defaulted to `None`/`""`, existing
`clip_id`/`parent_semantic_clip_id`/`render_fragment_id` untouched):
`CandidateTake` gains `source_span_id`, `attempt_id`, `realization_id`.
`DraftClip` gains `realization_id`, `semantic_idea_id`, `retry_family_id`,
`parent_realization_id`. `semantic_claims.Claim` gains
`canonical_claim_id`.

**Physical vs semantic identity** (the directive's central anti-jitter
requirement): `source_span_id` is a physical-observation identity and is
deliberately timestamp-sensitive, mirroring `source_identity.
stable_clip_id`'s own existing shape. `attempt_id`/`realization_id`/
`semantic_idea_id`/`canonical_claim_id` are canonical SEMANTIC identities
and are minted purely from content (normalized text) and structural
membership -- never from start/end timestamps -- so small ASR timing
jitter alone can never change a semantic id. See
`cutsell_worker/canonical_identity.py`'s module docstring for the full
design note and the one-owner-per-id table.

**Minting owners** (one call site each): `take_segmentation.py`
(`source_span_id`, including its own internal boundary-fragment-repair
join); `attempt_reconstruction.py`'s `_merge_attempt` (`attempt_id`, both
fused and singleton-passthrough, reused unmodified by
`_preserve_borderline_subspans`); `pipeline.py`'s `build_flow_b_draft`
(`realization_id`, immediately before take-grouping; `semantic_idea_id`/
`retry_family_id`, minted from the final post-semantic-equivalence
`take_group_id` -- D-050A intentionally conflates the two, per the D-050
audit's own Phase 3 finding); `post_selection_interior_gap_trim.py` and
`human_boundary_polish_v5.py` (`parent_realization_id`, mirroring the
existing D-036 `parent_semantic_clip_id` pattern exactly -- `realization_id`
itself is never touched by either split site, only carried through
`dataclasses.replace()`); `semantic_claims.py`'s `extract_claims`
(`canonical_claim_id`, from `claim_type` + `content_tokens` only, the seed
for a future D-050C cross-realization claim-dedup fix, not wired into any
coverage decision here).

**Observability**: `pipeline.py` adds a new, additive
`diagnostics["canonical_identity_chain"]` key (via
`canonical_identity.build_identity_chain_diagnostics`) listing every
selected clip's full identity chain in one place.

**Tests**: `tests/test_cutsell_d050a_canonical_identity.py` (26 tests) --
minting-function determinism/content-anchoring/anti-jitter, one-owner-
never-remints, physical trim preserves `realization_id`, physical split
(both sites, including a re-split of an already-split fragment) preserves
`realization_id` and stamps `parent_realization_id`, group members keep
distinct `realization_id`s while sharing one `semantic_idea_id`, no
duplicate/orphan realization ids through the real `build_flow_b_draft`
chain, no cycle in chained-split parent pointers, legacy `clip_id`
computation and `Claim.claim_id` untouched.

**Validation**: compileall clean; full `tests/test_cutsell_*.py` glob
1350 passed (1324 pre-existing + 26 new), 0 failed -- behavioral parity
confirmed (CleanCutBench's real-chain fixtures assert exact
winner/order/discarded/claim-coverage/Freeze outcomes and all 54 still
pass unchanged).

No editorial behavior changed. No Modal RAW launched. No RunPod touched.
No winner/grouping/ClaimCoverage/Freeze logic modified. Holding for review
before D-050B (Semantic Ledger).

## D-050B -- Semantic Ledger, shadow mode

Implements `cutsell_worker/semantic_ledger.py`: a typed, provider-neutral
Ledger (`RealizationRecord`, `SemanticIdeaRecord`, `CanonicalClaimRecord`,
`DecisionRecord`, `DiscardRecord`, `CompositeRecord`, `CoverageRecord`,
`ProvenanceEdge`) with a narrow write API
(`register_realization`/`register_semantic_idea`/`assign_retry_family`/
`register_claim`/`record_winner_decision`/`record_discard`/
`record_composite`/`record_coverage`/`record_physical_fragment`) --
internal state is name-mangled and only ever exposed through immutable
read views. `register_*` calls are idempotent when identical, raise
`LedgerIntegrityError` the instant two writes disagree about the same id
-- "no stage may silently create duplicate entries" is structural, not
conventional. A physical split can only ever attach a `render_fragment_id`
to an EXISTING realization (`record_physical_fragment`) -- there is no
method that mints a new realization from a fragment.

**Shadow reconstruction, not live hooks**: `build_semantic_ledger_shadow(draft)`
is a pure, read-only function called exactly once in
`universal_clean_cut.py`, right after CanonicalEditPlan/FinalEditReviewer/
StoryValidator have all run and written their own diagnostics, and
strictly before Freeze. It reconstructs the full decision history
(DELIVERY_SCORE_WINNER, SEMANTIC_WINNER_OVERRIDE, CLAIM_COVERAGE_OVERRIDE,
COMPOSITE_CREATED, CLIP_DISCARDED, REPLACEMENT_DECLARED,
DRAFT_REVIEW_REMOVED) from diagnostics keys every one of those stages
already writes today (`take_judge_groups`, `hybrid_editorial_chunks`,
`claim_coverage_best_take`, `canonical_edit_plan`,
`final_story_coherence_validation`, `draft_review_removed_ids`) rather
than injecting eight separate live hooks into eight stages -- a
deliberately lower-risk implementation of "shadow integration" since
every one of those facts is already present verbatim by the time this
runs. Documented, honest best-effort reconstructions: a discarded clip's
`semantic_idea_id` (not stamped by pipeline.py by design) is borrowed
from a stamped group-mate when one exists; per-realization claims are
extracted fresh via the same `extract_claims` ClaimCoverage itself calls.
Writes one new additive diagnostics key,
`diagnostics["semantic_ledger"]` -- nothing reads it back to decide
anything.

**Structural validators** (report, never raise, never change behavior):
`find_orphan_realizations` (a missing `semantic_idea_id` is only a real
orphan if no `DiscardRecord` explains its absence -- e.g. a
hybrid_editorial_chunks delete before the clip ever reached grouping,
D-049 Case A's exact shape, is NOT an orphan), `find_unknown_parent_ids`,
`find_fragments_without_parent_realization`, `find_provenance_cycles`
(DFS over the child->parent provenance graph), `find_duplicate_semantic_ids`
(always empty by construction -- `register_semantic_idea` already raises
on conflict).

**Parity checker**: `build_ledger_parity_report(ledger, draft)` -- the
Ledger's own idea `coverage_status` is deliberately derived independently
(from winner presence, not copied verbatim from CanonicalEditPlan) so
this comparison can actually disagree; checks selected/discarded state
parity, CanonicalEditPlan coverage parity, StoryValidator missing-coverage
parity, and fragment-provenance attachment. Reports `LedgerMismatch`
rows; never fails production behavior.

**D-049 shadow observability, proven without fixing either gap**: Case A
(hybrid_editorial_chunks deletes a unique realization with no verified
replacement) reconstructs as a `DiscardRecord` with
`replacement_realization_id=None`/`replacement_verified=False`. Case B
(near-identical canonical claims from retry-family siblings) reconstructs
as multiple `MEASUREMENT_QUANTITY` claims from two different
`source_realization_ids` visible under one `semantic_idea_id` --
deliberately not deduped, proving D-050C will have the information it
needs.

**Tests**: `tests/test_cutsell_d050b_semantic_ledger.py` (26 tests) --
one idea/multiple realizations, multiple independent ideas, provisional-
winner-then-override decision history and traversal
(`decision_history_for`), discard with/without verified replacement,
composite membership, cross-realization claim visibility (D-049 Case B
shape), physical fragments never minting a new realization, orphan/
unknown-parent/cycle/duplicate-id detection, shadow reconstruction from
real-shaped drafts (including D-049 Case A/B shapes), parity-report clean
and mismatch cases, JSON-safety of the diagnostics view, and a direct
proof that building the Ledger mutates nothing but the one new
diagnostics key.

**Validation**: compileall clean; full `tests/test_cutsell_*.py` glob
1376 passed (1350 pre-existing + 26 new), 0 failed -- behavioral parity
confirmed with the Ledger wired live into `universal_clean_cut.py`.

No editorial behavior changed. No winner/grouping/ClaimCoverage/Freeze/
Render/QC logic modified -- nothing reads Ledger state to decide
anything. No Modal RAW launched. No RunPod touched. Holding for review
before D-050C (winner/coverage authority consolidation).

## D-050C1 -- Unified Realization Resolver, shadow authority

Implements `cutsell_worker/realization_resolver.py`: a single provider-
neutral resolver that consumes a `SemanticLedger` and computes, in ONE
pass per `semantic_idea_id` (not as a sequential override stack), what a
unified Realization Resolution authority WOULD decide -- semantic safety,
critical-claim completeness, factual/negation consistency, delivery
quality, and contextual richness, evaluated together with a fixed
precedence order rather than weighted scores. Produces one
`RealizationResolution` per idea (`candidate_realization_ids`,
`winner_realization_id`, `composite_realization_ids`,
`covered_canonical_claim_ids`, `missing_critical_claim_ids`,
`discarded_realization_ids`, `retained_for_contextual_value`,
`decision_status` one of `RESOLVED_WINNER`/`RESOLVED_COMPOSITE`/
`REVIEW_REQUIRED`, `decision_reason`, `confidence`, `evidence`) plus one
`OrphanRealizationReview` per pre-grouping discard (D-049 Case A's exact
shape -- a realization with no `semantic_idea_id` at all).

**Hard invariants A-E**, all implemented as structural properties of the
decision procedure rather than bolted-on checks: (A) idea survival --
`resolve_realizations_shadow` iterates every `semantic_idea_id` the Ledger
knows and never omits one; (B) critical claim preservation -- a
`RESOLVED_WINNER`/`RESOLVED_COMPOSITE` status is only ever returned with
`missing_critical_claim_ids == ()`, otherwise the idea is
`REVIEW_REQUIRED`; (C) no duplicate retry realization -- the composite
search's own criterion 5 rejects any member contributing zero coverage
beyond the other chosen members; (D) discard requires safety --
`discarded_realization_ids` only ever contains a realization that is
either fully redundant with the chosen winner/composite's own coverage or
carries a Ledger-verified replacement, everything else lands in
`retained_for_contextual_value` instead of vanishing; (E) physical quality
cannot silently delete unique semantics -- `resolve_orphan_realizations_
shadow` walks every pre-grouping discard directly and returns
`REPLACEMENT_VERIFIED_SAFE` only with a Ledger-verified replacement,
`REVIEW_REQUIRED` otherwise, never a silent agreement.

**Canonical claim dedup (shadow-only)**: `build_requirement_groups`
greedily clusters `CanonicalClaimRecord`s within one idea into
`RequirementGroup`s via `_claims_dedup_equivalent` -- same `claim_type`
(structurally blocks "has X" vs "does not have X", since `classify_claim`
already gives a negated proposition its own type), compatible negation-
marker polarity, compatible quantitative meaning, and high content
equivalence on the remaining tokens (overlap coefficient, not plain
Jaccard -- a real sibling restatement adds filler words around an
identical core claim, and containment-style similarity survives that
where Jaccard-over-the-union would not). The quantitative-meaning check
reads each claim's own raw clause text (`CanonicalClaimRecord.text`, a new
additive field on the D-050B record, populated from `semantic_claims.
Claim.text`) rather than `content_tokens` alone: `_content`'s own
>=3-character token floor silently drops a bare short number ("5%" is 2
characters) while keeping a longer one ("10%", "5-10%") intact, so
`content_tokens`-only comparison would let "5% vs 10%" falsely dedup the
moment one side's digit fell below that floor in real ASR text --
reading raw text catches the conflict every time while still correctly
folding "5-10%"/"5 -10 %" restatements (the literal D-049 Case B shape)
into one requirement.

**Composite model (shadow-only, 6 criteria)**: same semantic idea only;
combined coverage is a superset of every CRITICAL requirement group;
no two members are on opposite sides of a detected contradiction signal;
no member is itself an unsafe-without-replacement realization; every
member contributes at least one group no other chosen member already
covers (no redundant member); smallest valid member count wins, ties
broken by sorted realization_id. Does not touch `composite_resolver.py`.

**Contradiction detection**: `_detect_contradiction_signals` flags two
claims from different realizations of the same idea as a genuine conflict
(negation-polarity or quantitative-value) only when they are the same
`claim_type`, topically related (Jaccard over non-digit content
>= 0.5), and NOT dedup-equivalent. A contradiction blocks BOTH single-
winner selection and composite formation for that idea -- the resolver
returns `REVIEW_REQUIRED` rather than silently picking a side or merging
both, mirroring D-020's existing "contradictory retries block Selection
Freeze for human review" posture.

**D-049 Case A required shadow result**: proven directly --
`resolve_orphan_realizations_shadow` returns `REVIEW_REQUIRED` for a
hybrid_editorial_chunks delete with no verified replacement, never a
silent discard confirmation.

**D-049 Case B required shadow result**: proven directly -- the two
near-identical "5-10%"/"5 -10 %" `MEASUREMENT_QUANTITY` claims from the
rich/vague sibling realizations fold into ONE `RequirementGroup`, and the
richer realization (covering both that group and its own NEGATION claim)
resolves as the single `RESOLVED_WINNER` rather than the short delivery-
only take.

**Shadow wiring**: `universal_clean_cut.py` calls
`resolve_realizations_shadow(ledger)` immediately after the existing
`semantic_ledger` shadow block, writing `build_realization_resolver_
diagnostics(report)` into a NEW, separate diagnostics key --
`diagnostics["realization_resolver_shadow"]` -- strictly before Freeze.
Nothing in `pipeline.py`, `deterministic_best_take_authority.py`,
`claim_coverage_best_take.py`, `composite_resolver.py`,
`canonical_edit_plan.py`, `final_story_coherence_validation.py`,
`final_edit_reviewer.py`, `selection_boundary_contract.py`, or any
Boundary/Render/QC module imports `realization_resolver` or reads either
diagnostics key -- confirmed by direct grep audit (only import site is
`universal_clean_cut.py`'s own shadow-wiring block) and by
`test_shadow_resolver_never_mutates_draft_timeline`.

**Parity report**: `build_resolver_parity_report(report, ledger)`
classifies each idea's engine-vs-shadow difference into one of the 7
named categories (`SAME`, `CONTENT_SAFETY_IMPROVEMENT`,
`CLAIM_DEDUP_DIFFERENCE`, `COMPOSITE_DIFFERENCE`,
`DELIVERY_RANK_DIFFERENCE`, `POTENTIAL_REGRESSION`,
`REVIEW_REQUIRED_DIFFERENCE`). Exercised over a representative
CleanCutBench-shaped cross-section (single candidate, exact retry,
paraphrased good takes, false-start-then-clean-retry, composite-required,
contradictory numeric retries, unique-fact preservation) in
`tests/test_cutsell_d050c1_parity_report.py`, re-running the exact real
production chain (`safe_group_takes` -> `reconcile_semantic_idea_
equivalence` -> `rank_takes` -> `apply_deterministic_best_take_authority`
-> `apply_claim_coverage_best_take` -> `apply_final_story_coherence_
validation`) plus the D-050A identity stamping CleanCutBench's own cheaper
harness intentionally omits. Result on that cross-section: 9 ideas, 6
SAME, 1 DELIVERY_RANK_DIFFERENCE, 1 COMPOSITE_DIFFERENCE, 1
REVIEW_REQUIRED_DIFFERENCE, 0 POTENTIAL_REGRESSION, 0
CONTENT_SAFETY_IMPROVEMENT, 0 CLAIM_DEDUP_DIFFERENCE (the dedicated
Case B fixture proves dedup capability directly; it didn't happen to
recur in this cross-section). Every difference found is explainable and
non-alarming: the DELIVERY_RANK_DIFFERENCE and COMPOSITE_DIFFERENCE are
both between two candidates the resolver judges equally safe, and the
REVIEW_REQUIRED_DIFFERENCE is the resolver correctly refusing to silently
pick a side on a genuinely contradictory 5%-vs-10% retry pair the cheap
synthetic engine path currently resolves by delivery score alone.

**Tests**: `tests/test_cutsell_d050c1_realization_resolver.py` (26 tests)
covers all 19 directive-required scenarios plus contradiction-detector
unit coverage -- one candidate, multiple retries, delivery-best-also-
complete, delivery-best-loses-critical-claim, richer-candidate-wins,
duplicate-claim dedup, number/negation/causal-direction non-dedup,
contextual-only retention, critical-uncovered-blocks-verdict, safe/unsafe
discard, valid/contradictory/redundant composite, no-coverage ->
REVIEW_REQUIRED, idea survival, D-049 Case A/B generic fixtures (both
verdict branches), and a direct mutation-freedom proof.
`tests/test_cutsell_d050c1_parity_report.py` (2 tests) adds the parity
report and shadow-quality-metrics printouts above.

**Validation**: compileall clean; D-050A (26), D-050B (26), D-050C1 (28)
all green; all 54 CleanCutBench fixtures green unchanged; full
`tests/test_cutsell_*.py` glob green with zero regressions.

No editorial behavior changed. No winner/grouping/ClaimCoverage/Composite/
CanonicalEditPlan/StoryValidator/Freeze/Render/QC logic modified --
nothing reads the new resolver's output to decide anything. No Modal RAW
launched. No RunPod touched. Holding for review before D-050C2 (the
actual authority cutover).

## D-050C1.5 -- full shadow parity qualification: NOT YET, 7 named blockers

Ran the D-050C1 shadow resolver against ALL 54 CleanCutBench fixtures
individually (`tests/test_cutsell_d050c1_5_full_cleancutbench_parity.py`),
not the D-050C1 representative 7-shape sample. Method: monkeypatches
`test_cutsell_clean_cut_core_evaluation_suite._run_core` for one pass over
every real `test_*` function (each one's own assertions still execute),
capturing every real `(takes, draft, arbiter)` call under its fixture's
own name -- zero duplication of fixture construction, zero modification of
the canonical suite file. Also fixed a real fidelity gap found while
building this: `_run_core`'s `take_judge_groups` entries never carry
`local_selected_clip_id` (only `pipeline.py`'s real chain stamps that, at
`pipeline.py:252`), so the Ledger never recorded delivery-score evidence
for ANY CleanCutBench-driven fixture until this harness reproduced that
stamp too -- applied identically in this file and in
`test_cutsell_d050c1_parity_report.py`.

**Result: 72 semantic ideas across 54 fixtures -- 59 SAME (81.9%), 6
CONTENT_SAFETY_IMPROVEMENT, 3 COMPOSITE_DIFFERENCE, 2
DELIVERY_RANK_DIFFERENCE, 2 POTENTIAL_REGRESSION, 0
REVIEW_REQUIRED_DIFFERENCE, 0 CLAIM_DEDUP_DIFFERENCE.** Per the directive's
own instruction ("do not automatically call a difference an improvement"),
every non-SAME row was individually re-examined against the fixture's own
real diagnostics rather than trusted at face value -- this reclassified 5
of the 8 auto-labeled "improvement" rows and surfaced two bugs in the
parity machinery itself. Net: 0 of the 8 non-SAME/non-benign findings
survive re-examination as an unqualified "good", producing 7 named,
precisely root-caused blockers:

- **F1 incidental/low-information claim importance not downgraded**
  (3-4 rows: `test_claim_coverage_self_source_trap_...`,
  `test_incidental_year_safely_omitted_...`,
  `test_redundant_date_repeated_...`, contributing to the multilingual
  case below). `semantic_claims.classify_claim` marks EVERY bare
  negation/number CRITICAL; production's `claim_coverage_best_take.
  _is_low_information_incidental`/`_override_blocked_by_incidental_self_
  source_claims` (D-047/D-048 FIX 2) downgrades a source-exclusive,
  temporal-aside-shaped bare fact so it never forces an override or
  composite -- the resolver has no equivalent, so it demands a composite
  (or prefers a candidate) production correctly decided was unnecessary.
- **F2 negation-marker detection loses short markers** (multilingual
  case). `_negation_tokens` reads `content_tokens`, which already dropped
  bare "no" (2 characters, below `_content`'s >=3-char floor) -- the same
  root cause already fixed once for digits via `_claim_digit_values`
  (reading raw `text`) but never applied to polarity detection. Two
  genuinely-equivalent negated paraphrases ("nunca...", "no...") fail to
  dedup, inflating a coverage requirement and forcing an unneeded
  composite.
- **F3 CORRECTION-marker over-trust + narrow contradiction-relatedness
  threshold** (`test_numeric_correction_across_two_takes_conservatively_
  blocks_freeze`). The resolver treats a bare correction marker
  ("actually, I checked") as enough to promote one of two competing
  numbers to sole-CRITICAL and auto-resolve -- exactly the "guessing"
  behavior that fixture's own docstring says production deliberately
  avoids (block Freeze, let a human resolve it in seconds); separately,
  `_CONTRADICTION_RELATEDNESS_THRESHOLD=0.5` over non-digit Jaccard is too
  narrow to even flag "2019" vs "2020" in structurally different
  sentences as related.
- **F4 no `claim_equivalence_arbiter` wiring** (`test_arbiter_is_
  consulted_only_for_ambiguous_coverage_...`). By design -- same
  provider-neutral, honest-gap pattern as every other bounded arbiter in
  this codebase -- so the resolver can't yet match a production decision
  an arbiter-confirmed paraphrase legitimately produced. Not urgent on
  its own, but blocks cutover for any idea an arbiter resolves.
- **F5 composite model has no completeness/narrative-order-validity
  check** (`test_claim_missing_from_its_own_idea_still_fails_...`, the one
  POTENTIAL_REGRESSION Section 4/5 asks not to wave through). Three
  critical claims split three ways across 3 INCOMPLETE (`complete_idea=
  False`) fragments plus one complete-but-content-free filler: production
  correctly blocks Freeze rather than guess (its own bounded ClaimCoverage
  composite resolution is capped at 2 pieces, deliberately); the resolver
  instead confidently assembles all 3 incomplete fragments into a 3-piece
  composite with no check on delivery completeness or narrative/temporal
  coherence between members.
- **F6 `SemanticIdeaRecord.current_winner_realization_id` goes stale once
  a composite forms** (`semantic_ledger.py` bug, found via
  `test_complementary_critical_claims_require_a_composite`, which the
  raw auto-classifier mislabeled POTENTIAL_REGRESSION). `record_composite`
  never updates or clears `current_winner_realization_id`, so a stage's
  earlier `record_winner_decision` (here: DeliveryScorer's local top
  score, before ClaimCoverage's composite override) is what a parity
  comparison sees, even though the real engine's actual final selection
  (`draft.selected`) matches the shadow composite exactly.
- **F7 parity comparison trusts a provisional engine pick when the real
  engine never converged** (the numeric-correction case above, and any
  `freeze_blocked`/multi-selected idea generally). When the final draft
  legitimately keeps MULTIPLE realizations for one idea (a deliberate
  "ask a human" outcome), `current_winner_realization_id` still reports
  DeliveryScorer's tentative pick as if it were the engine's confident,
  final answer.

None of these are editorial-behavior bugs -- every one lives entirely in
shadow code (`realization_resolver.py`'s claim-criticality/dedup/
composite model, or `semantic_ledger.py`'s winner-tracking field, or the
parity-comparison function itself) that nothing downstream reads. F1-F5
are real gaps in the resolver's own decision quality; F6-F7 are bugs in
how confidently the parity report can be trusted to represent "what the
engine actually decided" for a composite-formed or unresolved idea --
worth fixing before F1-F5's own counts can be trusted as complete, since
a stale/wrong "engine winner" can hide or manufacture a finding either
way.

**Retained forensic case replay** (Section 2): D-049 Case A (dedicated
fixture, `resolve_orphan_realizations_shadow` verdict) PASS;
D-049 Case B (dedicated fixture, dedup + richer-realization-wins) PASS;
D-048 FIX 1 distinct-addition false positive (CleanCutBench fixture,
appears as SAME in the full sweep) PASS; D-048 FIX 2 claim-coverage
self-source-exclusive trap (CleanCutBench fixture) FAIL -- this is F1
above, the shadow resolver does not yet replicate this specific
production guard; contradictory-quantity-retries (dedicated fixture +
representative-sample fixture, both REVIEW_REQUIRED) PASS;
number-sensitive non-dedup (dedicated fixture, clean ASCII text) PASS;
negation-sensitive non-dedup (dedicated fixture, clean ASCII text) PASS
but FAILS on realistic Spanish "no"-marker text (F2) -- the synthetic
unit fixture's markers ("not"/"never") happen to clear `_content`'s
length floor where real "no" does not.

**Shadow guarantee re-proven** (Section 7): grep-audited again -- only
`universal_clean_cut.py` imports `realization_resolver`; `freeze_blocked`
still derives solely from `coherence_diag`/`repair_result` (unchanged,
line 254); every one of the 54 fixtures' identity-stamped drafts were
asserted, per-fixture, to keep an identical `selected`/`discarded`
clip_id set before and after Ledger/resolver construction.

**Validation**: compileall clean; D-050A (26)/D-050B (26)/D-050C1 (28)
green; full 54-fixture CleanCutBench green unchanged; full
`tests/test_cutsell_*.py` glob green, 1405 passed, 0 failed (1404
pre-existing + 1 new full-sweep test). No GPU used anywhere in this
directive.

**RECOMMEND D-050C2 CUTOVER? NOT YET.** 0 POTENTIAL_REGRESSION is not met
(2, one substantive [F5] one methodological [F6]); F1/F2 mean "all
CleanCutBench expected outcomes compatible with shadow decisions" is not
met; F3 is a direct instance of "any number/negation semantics incorrectly
[resolved]" the readiness criteria names explicitly. No zero-realization
silent failures were found (Invariant A held on all 72 rows); no invalid
composite was silently accepted without being surfaced in this report; the
core mechanisms this directive chain set out to prove (D-049 Case A/B,
physical-split survival, contradiction detection on related conflicting
quantities) all work correctly in their own tested shapes. Next round
should fix F1/F2/F6/F7 together (F6/F7 first, so the parity signal itself
is trustworthy), then re-run this full sweep before F3/F5/F4 are
separately judged. No Modal RAW. No RunPod touched. No Render/QC
modified. Holding for review before D-050C2.

## D-050C1.6 -- shadow resolver qualification fixes: all 7 blockers resolved

Fixed F6/F7/F1/F2/F3/F5/F4, in the directive's mandated priority order,
re-running the full 54-fixture sweep after each phase. All fixes are
shadow-only (`realization_resolver.py`, `semantic_ledger.py`) plus two
small, additive `contracts.py`/`pipeline.py` fields threaded through for
observability -- none change authoritative editorial behavior.

**Phase 1 (F6/F7)**: added `SemanticIdeaRecord.engine_resolution_status`
(`ENGINE_RESOLVED_WINNER`/`ENGINE_RESOLVED_COMPOSITE`/`ENGINE_REVIEW_
REQUIRED`/`ENGINE_BLOCKED_UNRESOLVED`) and the narrow write method
`finalize_idea_engine_resolution`, called once at the END of
`build_semantic_ledger_shadow` from ground-truth realization `state` +
recorded composites -- never from decision-event order. Also fixed a
real bug this surfaced: `claim_coverage_best_take.py` writes a
composite's members under `"clip_ids"`, but the Ledger's composite
reconstruction read a `"member_clip_ids"` key that never existed,
silently reconstructing ZERO composites from every real ClaimCoverage
composite ever formed. `build_resolver_parity_report` now branches on
`engine_resolution_status` first: a composite-formed idea compares
composite-vs-composite (not composite-vs-a-stale-provisional-winner,
F6's fix); an unresolved/blocked idea is never scored a
POTENTIAL_REGRESSION for the shadow reaching a confident answer the
engine deliberately deferred (F7's fix) -- SAME if the shadow also
declines, CONTENT_SAFETY_IMPROVEMENT if it resolves.

**Phase 2 (F1)**: `_effective_importance` downgrades a CRITICAL
requirement group to SUPPORTING when ALL its member claims are low-
information/incidental (reuses `claim_coverage_best_take._is_low_
information_incidental` directly -- ONE shared utility, per the
directive, not a reimplementation of D-047/D-048 FIX 2's own logic) AND
the fact is source-exclusive (no second, independent realization also
raised it). Corroboration by >1 realization keeps it CRITICAL. Also
fixed a companion bug this exposed: the dedup quantitative-meaning gate
required exact digit agreement whenever EITHER side showed any digit
evidence, which wrongly blocked dedup between a claim and its own
paraphrase that merely adds an incidental year (an absent number is not
a conflicting one) -- now only compared when BOTH sides show digit
evidence.

**Phase 3 (F2)**: `_claim_has_negation` reads a claim's raw `text` via
`final_sibling_grouping._negations` (the SAME general negation check
`classify_claim` itself already uses) instead of `content_tokens`, which
silently drops bare "no" (2 characters) below `_content`'s length floor
while keeping "not"/"never"/"nunca"/"sin"/"without" intact. Used by both
the dedup polarity gate and `_detect_contradiction_signals`, which also
now compares a NEGATION-typed claim against a differently-typed positive
counterpart (structurally impossible under the old same-claim_type-only
gate, since `classify_claim` gives a negated proposition its own type by
construction) -- proven directly against the real "no soy la unica" vs
"soy la unica" shape.

**Phase 4 (F3)**: `_correction_explicitly_supersedes` only lets a
CORRECTION-typed claim override a conflicting prior claim when its own
raw text names AND rejects (a negation marker within a short window of)
the prior claim's specific value, AND the two claims share some non-digit
topical content (guards "correction of a different entity" -- two claims
can share a bare digit substring, e.g. "5" a dose vs "5%" a rate, with no
shared topic at all). `_detect_contradiction_signals` no longer requires
topical relatedness for a CORRECTION-involving pair before examining it
(a correction is often anaphoric, with little lexical overlap with what
it corrects) -- this is what actually catches "Actually, I checked, and
it was 2020" (no named/rejected prior value) as still-blocking, the
literal CleanCutBench fixture this directive chain has referenced since
D-050C1.5.

**Phase 5 (F5)**: `_MAX_COMPOSITE_SIZE` lowered from 4 to 2 (matching
production's own ClaimCoverageBestTake bound) -- the D-050C1.5 full
sweep's one real POTENTIAL_REGRESSION was exactly a 3-fragment assembly
production's own bounded resolution would never attempt; bounding alone
makes it structurally unreachable, falling to REVIEW_REQUIRED instead.
Added `_composite_members_temporally_compatible` (no two members may
occupy overlapping time windows) as an honest, narrower proxy for full
narrative/causal-order validity -- true causal-order validation would
need the bounded `CausalOrderArbiter` this codebase defines elsewhere,
deliberately not wired here. An EARLIER version of this phase also
excluded `complete_idea is False` realizations from composite
membership; empirically wrong (caught by re-running the full sweep):
`test_complementary_critical_claims_require_a_composite` composites two
realizations BOTH marked incomplete BY DESIGN, exactly the point of a
composite. Reverted -- completeness is not a composite-eligibility gate.

**Phase 6 (F4)**: `_claims_dedup_equivalent` now accepts the existing,
provider-neutral `semantic_claims.ClaimEquivalenceArbiter` contract (no
new model/provider) and consults it ONLY when a pair clears every hard
gate (type/polarity/quantity) and its deterministic overlap falls in the
genuinely ambiguous band (`_DEDUP_AMBIGUOUS_FLOOR=0.4` to
`_CLAIM_DEDUP_THRESHOLD=0.7`) -- below the floor, confidently distinct,
arbiter never consulted. Fails open to FALSE (distinct claims, never
silently collapsed) on no arbiter, an exception, or a non-`True` verdict.
Every consultation is recorded on `ResolverReport.arbiter_consultations`
and surfaced in `diagnostics["realization_resolver_shadow"]
["arbiter_consultations"]`.

**Corrected baseline after Phase 1 (F6/F7) alone**: re-running the full
sweep immediately after Phase 1 (before touching F1-F5) already
recategorized several rows the raw auto-classifier had gotten wrong --
the composite-key-name bug fix alone flipped `test_complementary_
critical_claims_require_a_composite` from a false POTENTIAL_REGRESSION
to SAME, and F7 flipped `test_contradictory_factual_retries_block_
freeze_not_silently_resolved` from a coincidental false SAME to an
honestly-labeled CONTENT_SAFETY_IMPROVEMENT (the resolver was in fact
picking a side on a genuine unresolved contradiction the raw comparison
logic had been masking).

**Final full sweep** (all 7 fixes applied): 72 semantic ideas, 62 SAME
(86.1%), 4 CONTENT_SAFETY_IMPROVEMENT, 2 COMPOSITE_DIFFERENCE, 3
DELIVERY_RANK_DIFFERENCE, 1 REVIEW_REQUIRED_DIFFERENCE, 0
CLAIM_DEDUP_DIFFERENCE, **0 POTENTIAL_REGRESSION**. Every remaining
non-SAME row was individually re-examined: the 4 CONTENT_SAFETY_
IMPROVEMENT and 3 DELIVERY_RANK_DIFFERENCE rows are all genuine/benign
(both sides independently confirmed critically complete, or the engine
never converged and the shadow safely does); the 2 remaining
COMPOSITE_DIFFERENCE rows are safe-conservative (no content lost --
one is arbiter-dependent, since no arbiter is wired into this offline
sweep; wiring a real one could close it); the 1 REVIEW_REQUIRED_
DIFFERENCE is the corrected F5 case, now conservatively correct (the
engine's own filler-clip pick actually loses 3 critical facts; the
resolver refuses to guess a composite and correctly declines instead of
confidently picking the worse answer).

**Forensic case re-check**: D-049 Case A PASS; D-049 Case B PASS
(unaffected by the digit-gate fix -- both sides already show real digit
evidence in raw text); D-048 FIX 1 (distinct-addition) PASS; D-048 FIX 2
(self-source-trap) now PASS (was FAIL in D-050C1.5, fixed by F1);
contradictory-quantity-retries PASS; number-sensitive non-dedup PASS;
negation-sensitive non-dedup now PASS on both clean-ASCII AND realistic
Spanish "no" text (was a partial FAIL in D-050C1.5, fixed by F2/F3).

**Shadow guarantee re-proven**: grep-audited again -- only
`universal_clean_cut.py` imports `realization_resolver`; `freeze_blocked`
unchanged (line 254); the full-sweep test's per-fixture mutation-freedom
assertion still passes for all 54 fixtures.

**Tests**: `tests/test_cutsell_d050c1_6_qualification_fixes.py` (28 new)
covering all 6 phases -- F6 finalization non-staleness, F7 never-a-
regression-when-unresolved, F1 incidental-downgrade with/without
corroboration, F2 negation survival (Spanish "no", English "not", "sin",
same-fact-never-dedupes, double-negative-fails-safe, cross-type
contradiction), F3 explicit-vs-ambiguous-vs-different-entity correction,
F5 valid/invalid/overlapping/incomplete-ok/redundant composite shapes,
F4 deterministic-first/ambiguous-consults/fail-open-on-false-or-
exception-or-no-arbiter/diagnostics-recorded/bounded-calls.

**Validation**: compileall clean; D-050A (26)/D-050B (26)/D-050C1 (28)
green; new D-050C1.6 tests (28) green; all 54 CleanCutBench fixtures
green unchanged; full `tests/test_cutsell_*.py` glob green, 1433 passed,
0 failed (1405 pre-existing + 28 new). No GPU used anywhere in this
directive.

No editorial behavior changed. No winner/grouping/ClaimCoverage/
Composite/CanonicalEditPlan/StoryValidator/Freeze/Render/QC logic
modified. No Modal RAW launched. No RunPod touched.

**RECOMMEND D-050C2 CUTOVER?** All of D-050C1.5's stated readiness
criteria are now met: 0 POTENTIAL_REGRESSION, no CleanCutBench expected
outcome contradicted, no critical claim lost, no numeric/negation
contradiction collapsed, no unsafe composite accepted, no silent
zero-realization idea, all D-045->D-049 forensic fixtures correct. That
said, cutover itself -- making this resolver AUTHORITATIVE over
DeliveryScorer/`_semantic_best_take`/ClaimCoverageBestTake/
CompositeResolver in live production -- is a materially larger,
harder-to-reverse step than anything qualified here: this evidence base
is CleanCutBench's 54 synthetic fixtures plus ~140 targeted unit tests,
not a live RAW/Human-Gold run, and F4's arbiter path has only been
proven against a test stub, never a real bounded arbiter end-to-end.
Recommend D-050C2 proceed only with explicit user authorization, behind
an explicit rollback flag (mirroring `CUTSELL_CLEAN_CUT_CORE_V1`'s own
precedent), and only after this qualification evidence has been
reviewed by a human -- not as an automatic next step from this report.

## D-050C2 -- Unified Realization Resolver, controlled authority cutover

User authorized D-050C2 explicitly as a CONTROLLED cutover, not
permission for an irreversible replacement: legacy logic stays in the
codebase as evidence/rollback, Modal RAW stays gated on offline
qualification, RunPod untouched.

**Feature flag** (`cutsell_worker/resolver_mode.py`, new): typed 3-state
`CUTSELL_UNIFIED_REALIZATION_RESOLVER` env var --
`LEGACY`/`SHADOW`/`AUTHORITATIVE` -- never an ambiguous boolean pair, so
"half on" has no representable state. `resolve_resolver_mode(env=None)`
reads case-insensitively and fails safe to `LEGACY` on unset, empty, or
any unrecognized value; never silently escalates toward
`AUTHORITATIVE`. Rollback is resetting one env var, no code revert.

**Authority cutover point** (`realization_resolver.py`, new section):
one function, `apply_authoritative_realization_resolution(draft,
ledger, report) -> AuthoritativeApplicationResult`, is the entire
surface where the resolver's decision can change `DraftTimeline.
selected`/`alternates`/`discarded`. It is called from exactly one
place, `universal_clean_cut.py`, gated strictly on `resolver_mode ==
RESOLVER_MODE_AUTHORITATIVE`; `LEGACY` and `SHADOW` both skip the call
entirely and leave the draft the legacy stages already produced
untouched. Per semantic idea: `RESOLVED_WINNER` keeps exactly that
realization's clips selected and discards every other candidate;
`RESOLVED_COMPOSITE` keeps exactly the validated composite members;
`REVIEW_REQUIRED` leaves that idea's clips completely untouched (no
guess) and escalates the overall application `status` to
`REVIEW_REQUIRED`, which `universal_clean_cut.py` OR's into
`freeze_blocked` alongside the existing StoryValidator/repair-loop
conditions -- Freeze cannot proceed silently. A realization the
resolver names but that cannot be mapped back onto any clip in the
draft (an unmapped id) is treated the same as an unresolved orphan:
fail-closed to `REVIEW_REQUIRED`, never a silent drop. Orphan
realizations (no `semantic_idea_id` at all -- the D-049 Case A shape)
are re-checked via the existing `resolve_orphan_realizations_shadow`
and any that still resolve to `REVIEW_REQUIRED` also force the overall
status, so a realization with unique content can never be
authoritatively discarded without either a verified replacement, proof
of redundancy, or an explicit review escalation -- D-049 Case A is
structurally unreachable at this boundary.

**Legacy modules now evidence-only in AUTHORITATIVE mode**:
`DeliveryScorer`/`take_judge.rank_takes`, `_semantic_best_take`,
`deterministic_best_take_authority`, `ClaimCoverageBestTake` (including
its own `CompositeResolver` composite formation) all still run and
still write their diagnostics -- nothing upstream of the cutover point
was removed or skipped -- but in `AUTHORITATIVE` mode none of their
conclusions may subsequently overwrite the resolver's applied
selection, because the cutover point is the single place selection is
finalized and nothing runs after it that re-reads their diagnostics to
re-decide. `CanonicalEditPlan`/`FinalEditReviewer`/`StoryValidator`
diagnostics are an explicitly documented C2-phase exemption (per the
directive's own Section 8): they still compute from the same
comparison inputs they always have and are not yet rewired to read the
post-cutover resolved state as their source of truth -- proven, not
assumed, by `test_authoritative_diagnostics_expose_legacy_vs_
authoritative` and `test_canonical_edit_plan_and_reviewer_diagnostics_
survive_boundary_unchanged`-style non-regression on the LEGACY path.
This is a known, deliberate scope boundary for C2, not an oversight; a
future phase should give StoryValidator/CanonicalEditPlan one shared
coverage/provenance read API rather than two independent
recalculations.

**Observability**: `build_authoritative_resolution_diagnostics` writes
one `diagnostics["realization_resolver_authority"]` block per run
(schema `cutsell.realization_resolver_authority.v1`) carrying mode,
overall status, unresolved-orphan ids, and per-idea: winner/composite
ids, covered/missing canonical claim ids, discarded ids,
contextually-retained ids, decision reason, the legacy winner/composite
for comparison, and whether legacy and authoritative agreed -- one
place to read, not twenty diagnostics dicts reconstructed by hand. The
D-050C1 shadow diagnostics (`realization_resolver_shadow`) keep
computing unconditionally in every mode, unchanged.

**Offline cutover qualification** (`tests/test_cutsell_d050c2_
authority_cutover.py`, new, 19 tests): `_run_full_cutover_sweep()`
reuses the D-050C1.5 harness (`_collect_every_fixture_call`/
`_stamp_identity`) to run all 54 CleanCutBench fixtures through
identity stamping, `resolve_realizations_shadow`, and
`apply_authoritative_realization_resolution`, then checks per-idea
safety invariants directly against the applied draft: a
`RESOLVED_WINNER`/`RESOLVED_COMPOSITE` idea outcome must have zero
missing critical claims and at least one surviving selected clip; any
`REVIEW_REQUIRED` idea outcome must correspond to an escalated overall
`status`. Result: **54/54 fixtures, 0 unsafe findings** (23 fixtures
identical clip-level selection to LEGACY, 31 differ -- expected, since
these are the same idea-level differences D-050C1.6 already qualified
as 0 POTENTIAL_REGRESSION/mostly CONTENT_SAFETY_IMPROVEMENT, now
visible at the whole-draft bucket level; 3 fixtures resolve to
`REVIEW_REQUIRED` under both LEGACY's own coherence validation and the
authoritative path, in agreement). The remaining ~16 migration tests
cover: mode resolution (default/unrecognized-fails-safe/case-
insensitive per state), SHADOW mode computing the identical resolver
report AUTHORITATIVE would but never applying it even where the
resolver actively disagrees with legacy (mirrors `universal_clean_cut.
py`'s own `if resolver_mode == RESOLVER_MODE_AUTHORITATIVE` gate
directly), authoritative winner/composite replacement, REVIEW_REQUIRED
leaving selection untouched with no silent fallback, critical-claim/
number/negation safety preserved through application, causal
(temporal-overlap) ambiguity forcing REVIEW_REQUIRED rather than a
guess, the D-049 Case A delete-without-replacement shape forcing
REVIEW_REQUIRED via orphan detection, physical fragment provenance
(two fragments sharing one realization_id) moving as one unit never
split across buckets, legacy evidence unable to overwrite the
authoritative resolution, and Freeze blocking on REVIEW_REQUIRED /
permitting SEMANTICALLY_RESOLVED.

**Validation**: compileall clean; D-050A (26)/D-050B (26)/D-050C1
(28)/D-050C1.5/D-050C1.6 (28)/D-050C2 (19) all green (128 across the
whole D-050 series); all 54 CleanCutBench fixtures green under LEGACY
(default), explicit `LEGACY`, explicit `SHADOW`, and explicit
`AUTHORITATIVE` env values alike -- CleanCutBench's own harness
(`_run_core`) does not call into `universal_clean_cut.py`'s cutover
wiring at all, so this confirms the flag is inert to CleanCutBench's
own expected-outcome assertions, not a second independent AUTHORITATIVE
qualification (that qualification is the 54-fixture offline sweep
above, which does call the real cutover point). Full `tests/
test_cutsell_*.py` glob: **1452 passed, 0 failed** (1433 pre-existing +
19 new) with the env var unset (LEGACY, the default) -- byte-for-byte
LEGACY parity confirmed at full-suite scale, not just CleanCutBench.

**Known characteristic, not a regression**: forcing
`CUTSELL_UNIFIED_REALIZATION_RESOLVER=AUTHORITATIVE` across the *entire*
`tests/test_cutsell_*.py` glob (beyond what this directive required --
an extra diligence check) surfaces 3 failures in
`test_cutsell_universal_clean_cut.py`, all in fixtures that hand-build
bare `DraftClip`s with no `semantic_idea_id`/`realization_id` at all
(these predate D-050A and were never updated to carry real
`pipeline.py`-only identity stamping). Under `AUTHORITATIVE` mode this
is correctly detected as unmapped/orphan realizations and fail-closed
to `REVIEW_REQUIRED`, blocking Freeze -- exactly the D-049 Case A
safety behavior this directive requires, not a bug. Every clip a real
Clean Cut Core V1 run produces is stamped with this identity by
`pipeline.py`'s `_draft_clip` before it ever reaches the cutover point,
so this shape cannot occur on a genuine production draft; it is purely
an artifact of these three tests' pre-D-050A synthetic fixtures never
being exercised in `AUTHORITATIVE` mode by design (the default is
`LEGACY`, so none of these tests break in any normal run or CI). Left
unmodified -- fixing them is out of this directive's scope, and
weakening the fail-closed behavior to make them pass under a forced
`AUTHORITATIVE` override would be the actual regression.

No editorial behavior changed in `LEGACY` mode (the default and the
only mode active anywhere in CI/production today). No legacy logic
removed. No Modal RAW launched. No RunPod touched.

**READY FOR MODAL CANARY?** Offline cutover qualification passes every
stated gate: LEGACY exactly unchanged, AUTHORITATIVE satisfies every
CleanCutBench safety invariant with 0 unsafe findings, 0 critical-claim/
contradiction/number/negation regressions, 0 unsafe composites, 0
silent zero-realization ideas, 0 resolver-authority overwrite by legacy
modules. Per this directive's own closing instruction, this is a report
of readiness, not a launch: no Modal canary has been run, and one
should not be until the user explicitly authorizes it.

## D-050C2 MODAL CANARY -- run 33701641177

User authorized ONE Modal Video00 canary in `AUTHORITATIVE` mode on
head `ade4578` (later `324233b` after the one-line env overlay this
directive itself required -- see below). The Modal RAW workflow
(`cutsell-video00-modal-raw.yml`) had no mechanism to pass
`CUTSELL_UNIFIED_REALIZATION_RESOLVER` into the run at all; the user
explicitly authorized adding the same one-line overlay pattern the
D-044 fix already uses for `CUTSELL_HYBRID_LLM_ENABLED`/
`CUTSELL_HYBRID_PROVIDER` (a runtime-config-only change, no
cutsell_worker editorial module touched) rather than dispatch a run
that would have silently stayed on `LEGACY` and tested nothing new.

**Result** (run 33701641177, ~7m14s on one L4, low-cost): the resolver
genuinely ran in `AUTHORITATIVE` mode -- confirmed directly from
`diagnostics.realization_resolver_authority.mode == "AUTHORITATIVE"`
with real per-idea output. 17 semantic ideas evaluated: 16
`RESOLVED_WINNER` (0 composites), 1 `REVIEW_REQUIRED`; 13/16 winners
agreed with legacy's own pick, 3/16 differed with zero critical-claim
loss in any; 0 of the 16 resolved winners has a nonempty
`missing_critical_claim_ids` (the only nonempty case is the one
declined idea, exactly the fail-closed contract). 6 orphan/unmapped
realizations forced the overall status to `REVIEW_REQUIRED`, which
correctly escalated `freeze_blocked` -- Freeze did not proceed, no
render was attempted, `deliverable: false`.

Freeze was independently also blocked by legacy `FinalEditReviewer`
(computed pre-cutover, per the then-current C2 evidence-only exemption)
on a `CRITICAL_CLAIM_LOST` finding for the same real-video "5% of
cancers are hereditary" claim this repo's D-047/D-048 work already
root-caused -- strong circumstantial evidence (same claim text/theme,
singular FAIL and singular `REVIEW_REQUIRED`) that the resolver's own
`REVIEW_REQUIRED` idea is the SAME idea, correctly declining rather
than accepting the same lossy composite legacy's evidence-only
computation had settled for. Not provable by exact id (see below).

**Human Gold could not be read**: `Verify frozen Selection lock` failed
(Freeze never froze anything to compare against the pinned LEGACY
baseline), and GitHub Actions' own step-dependency chain skipped the
architecture and Human Gold (18-check) validators as a result -- a
pre-existing CI-wiring gap, not a resolver defect, and not patched at
the time (see D-050C3 Section 8 fix below).

**Honest limitation**: GitHub's own cross-log secret masking (the
workflow's existing "mask the live RunPod template's env" step) blanked
many bare digits throughout the run's log wherever they collided with a
masked secret substring, including unrelated diagnostic numbers and
clip/realization ids. The 6 orphan realizations and the one
`REVIEW_REQUIRED` idea could not be traced to real ids from this log;
Section 5/6 below are a structural code audit, not a literal per-id
replay of this run.

## D-050C3 -- authoritative downstream integration + identity closure

**Section 1 (execution order proof)**: confirmed by direct code trace of
`universal_clean_cut.py`: the D-050C2 order was exactly
`legacy Selection -> StoryValidator -> CanonicalEditPlan ->
FinalEditReviewer -> Semantic Ledger -> Unified Resolver ->
apply_authoritative_realization_resolution -> Freeze OR-in`.
CanonicalEditPlan/StoryValidator/FinalEditReviewer ran once, on the
pre-cutover draft, and their diagnostics were what Freeze read even in
`AUTHORITATIVE` mode (the C2 evidence-only exemption) -- the primary
downstream integration gap this directive closes.

**Section 2-4 (reordered pipeline, evidence-only exemption removed)**:
`universal_clean_cut.py` now runs, in `AUTHORITATIVE` mode only
(`clean_cut_core_v1_enabled` and `resolver_mode == AUTHORITATIVE`):
grouping/DeliveryScorer/ClaimCoverageBestTake (unchanged) -> first-pass
StoryValidator/CanonicalEditPlan/FinalEditReviewer/repair loop (still
needed here because the Semantic Ledger's own reconstruction reads
`final_story_coherence_validation`/`canonical_edit_plan` diagnostics
for one enrichment) -> Semantic Ledger -> Unified Resolver ->
`apply_authoritative_realization_resolution` -> a SECOND pass of
StoryValidator, then CanonicalEditPlan + FinalEditReviewer + bounded
repair, now on the resolver's own resolved draft -> Freeze. The first
pass's `canonical_edit_plan`/`final_edit_reviewer`/`repair_loop`/
`final_story_coherence_validation` diagnostics are relabeled
`*_legacy_evidence` before the second pass overwrites the real keys;
nothing below this point ever reads a `_legacy_evidence` key for a
decision, only for comparison. `LEGACY`/`SHADOW` modes, and the two
non-Clean-Cut-Core-V1 selection branches, run the first pass exactly
once and are otherwise untouched -- proven by the full suite staying at
1452 passed under the default (unset) env var, byte-for-byte.

**Section 5 (orphan root-cause audit)**: traced every production
construction site that mints or copies a `DraftClip`'s identity fields.
`human_boundary_polish_v5.py` and `post_selection_interior_gap_trim.py`
both mint fragments via `dataclasses.replace(original, ...)`, which
copies `realization_id`/`semantic_idea_id`/`retry_family_id` forward
unless explicitly overridden -- neither overrides them; both are clean.
`claim_coverage_best_take.py`'s composite formation never mints a new
clip at all (`replace(clip, selected=...)` on existing clips only) --
clean. The one and only other production `DraftClip(...)` construction
site is `pipeline.py`'s `_draft_clip`, called from three places:
`selected`/`alternates` correctly pass `group_id=clip_to_group.get(take.
clip_id)`; **`discarded_clips` hardcoded `group_id=None`
unconditionally** for both its inputs -- `discarded` (pre-grouping
clean_cut/hybrid-cleanup rejects, which is a no-op fix, they were never
grouped) and, critically, `review_removed` (post-grouping draft_review
rejects, which per `removed_group_ids`'s own construction DID go
through grouping and had a real `group_id` the hardcoded `None` was
silently discarding). Classified closest to the directive's category G
("another proven cause"): identity minting for the discarded bucket
ignored the exact same `clip_to_group` lookup already used and
available two lines above for `selected`/`alternates`. Fixed to the
identical lookup -- no clip-id hardcoding, general to any discard path
reaching this constructor. A discarded realization with this bug never
showed up as a flagged "orphan" (`find_orphan_realizations()` excludes
any realization a `DiscardRecord` already explains, and pipeline.py's
own discard bookkeeping always records one) -- its actual effect was
silent invisibility: it never joined its true idea's `realization_ids`
in the Ledger at all, so a discarded retry's unique claims (if any)
could never be considered by the resolver's requirement-group analysis
for that idea. Fixed regardless, as a general identity-propagation
correctness issue independent of whether it explains canary
33701641177's specific 6 orphans (which, per the honest limitation
above, could not be traced by id). Full suite unaffected: still 1452
passed after the fix.

**Section 6 (hereditary REVIEW_REQUIRED audit)**: from the canary's own
(masked) diagnostics, the resolver's one declined idea has
`decision_reason: "no_single_or_composite_realization_covers_all_
critical_claims"` -- it tried both a single-realization and a bounded
2-piece composite path and neither achieved full critical-claim
coverage. Legacy's own independent evidence-only `FinalEditReviewer`
FAIL, on the real video's own already-root-caused "5% hereditary"
claim, corroborates this is a genuine content gap in the *available*
candidates, not a resolver defect. Classified **F: genuinely
ambiguous/impossible -- REVIEW_REQUIRED is correct** on the balance of
available evidence. Cannot fully rule out C (a candidate carrying the
missing claim was among the 6 orphans) without exact id tracing, which
this session could not obtain (see honest limitation above); the
Section 5 fix is the appropriate general mitigation either way, and
whether it actually changes this specific idea's outcome can only be
confirmed by a second live canary (not run in this directive).
Resolver logic itself was not changed for this section, per the
directive's own instruction not to without a proven general defect.

**Section 7 (typed resolved-state contract)**: `AuthoritativeSemanticState`/
`AuthoritativeSemanticIdeaState` (`realization_resolver.py`, new) --
one frozen, typed object per run, built once from `AuthoritativeApplicationResult`
plus the Ledger it was computed from: `semantic_idea_id`, winner-or-
composite realization ids, canonical critical claim ids, covered/missing
claims, discarded realizations, contextually-retained ids, `replacement_
verified` (true iff resolved with zero missing critical claims),
`story_order_position` (rank by earliest winning clip start among
resolved ideas, `None` for `REVIEW_REQUIRED`), and `provenance`
(`source_span_ids` per surviving realization, from the Ledger, never
guessed). Attached to `diagnostics["authoritative_semantic_state"]` in
`AUTHORITATIVE` mode only. Downstream modules are not rewritten to take
this object as a parameter (that would mean rewriting CanonicalEditPlan/
StoryValidator/FinalEditReviewer's internals, well beyond this
directive's scope) -- instead, per Section 2-4's reorder, they operate
directly on the resolver's own resolved `DraftTimeline` as their sole
selection input, so there is structurally one resolved state for them
to agree with, not two to reconcile.

**Section 8 (CI observability)**: `cutsell-video00-modal-raw.yml`'s
`Verify Video00 architecture` and `Verify Human Gold regression QA`
steps now carry `if: always()` -- a blocked Freeze (a failed `Verify
frozen Selection lock` step, itself unchanged) no longer skips either
validator; both still run against the same `result.json` and upload
their own reports regardless. CI observability only, no engine change.

**Section 9 (tests)**: `tests/test_cutsell_d050c3_downstream_integration.py`
(new, 10 tests) -- pipeline.py's discard-path identity fix (grouped vs
genuinely-ungrouped), fragment/composite realization-id sharing
preserves the Ledger's mapping, `human_boundary_polish_v5`'s real split
function keeps identity on both physical pieces, the authoritative
winner/composite reflected in the real (non-legacy-evidence)
`canonical_edit_plan`, `REVIEW_REQUIRED` blocks Freeze while a resolved
idea permits it, the legacy-evidence `canonical_edit_plan` key is
provably identical to a genuine LEGACY-mode run of the same draft (so
it is truly untouched, not resolver-influenced), LEGACY mode carries
none of the new `_legacy_evidence`/`authoritative_semantic_state` keys,
SHADOW computes the resolver report without ever splitting evidence,
and the Modal workflow's two validator steps carry `if: always()`.

**Section 10 (full qualification)**: compileall clean. Full D-050
series (A/B/C1/C1.5/C1.6/C2/C3): 138 passed. All 54 CleanCutBench
fixtures green under LEGACY (default) and explicit `AUTHORITATIVE`
(CleanCutBench's own harness still never calls into
`universal_clean_cut.py`'s wiring, so this remains a LEGACY-parity
check on that harness, not a second reorder qualification -- the
reorder qualification is the Section 9 end-to-end tests above). Full
`tests/test_cutsell_*.py` glob: 1452 passed (before the new file) / see
final report for the post-file count -- 0 failed with the env var unset
(LEGACY, the default). 0 identity-orphan violations introduced by the
Section 5 fix (full suite unaffected). 0 critical-claim regressions. 0
unsafe composites. 0 semantic dual-authority findings: the reorder's
own design (legacy diagnostics relabeled before the second pass writes
the real keys) makes a legacy CanonicalEditPlan overwriting an
authoritative one structurally impossible, not just empirically absent.

Known characteristic carried forward from D-050C2, now with one more
instance: forcing `AUTHORITATIVE` across the pre-D-050A synthetic
fixtures in `test_cutsell_universal_clean_cut.py` (fixtures with no
identity stamping at all, never exercised in `AUTHORITATIVE` mode by
default) now shows 4 (was 3) fail-closed differences from LEGACY --
the new one is a `repair_loop.attempt_count` difference, explained
identically: the Ledger finds zero semantic ideas (no clip carries a
`semantic_idea_id`), so `apply_authoritative_realization_resolution` is
a true no-op on an already-first-pass-repaired draft, and the SECOND
downstream-validation pass correctly finds nothing left to repair. Not
a regression; still purely a synthetic-fixture artifact, still
unreachable on a genuine production draft.

No Modal RAW launched in this directive. No RunPod touched. No
Video00 phrases patched. No resolver semantic logic changed except the
general, non-Video00-specific pipeline.py identity fix in Section 5.

## D-050D -- front-half stability + orphan closure audit (report only)

Report-only directive against canary 33727122331 (5 orphans) plus a
direct code trace, no code changes. Full findings live in the
session transcript; the load-bearing conclusion, confirmed and acted
on in D-050D1 below: `mint_realization_id` (the sole minting owner,
`canonical_identity.py`) ran only over `pipeline.py`'s `kept` list,
strictly AFTER `apply_clean_cut`/`apply_provider_judgements` (clean_cut
removal) and `apply_composite_resolution` (hybrid_editorial removal)
had already partitioned candidates -- so anything either stage removed
never received a canonical identity at all, falling back to its own
`clip_id` in the Ledger. All 5 orphans in that canary traced to exactly
this mechanism: 3 were `apply_composite_resolution`'s own semantic
delete decisions (`discard_reason` values matching hybrid_editorial
vocabulary -- `semantic_failed_plus_local_performance`, `high_
confidence_semantic` x2), 2 were the generic pre-grouping catch-all.
Separately (Section 4 of that report): `resolve_orphan_realizations_
shadow` conflated "never had semantic understanding applied" (ordinary
clean_cut rejects) with "should have had it but lost it" (genuine D-049
Case A) under one blanket REVIEW_REQUIRED verdict -- also identified as
worth closing, acted on below.

## D-050D1 -- early realization identity minting

**Section 1 (relocation)**: `pipeline.py`'s minting loop moved from
after `apply_composite_resolution` to immediately over `take_tuple`,
before `apply_clean_cut` is ever called -- the complete candidate pool
(AttemptReconstructor output plus any `preserved_subspan_candidates`,
already merged upstream in flow_b.py) is now identified before any
editorial stage can keep, discard, or transform it. Still exactly ONE
minting owner (`mint_realization_id`); the old post-composite-
resolution block was deleted, not duplicated. Verified safe by
inspection: `apply_clean_cut`/`apply_provider_judgements` are pure
partitions of the same `CandidateTake` objects (never rebuild them),
and `apply_composite_resolution`'s own kept/deleted split is the same
shape, so minting once, at the top, survives every branch by
construction.

**Section 3 (survives partitioning)**: auditing every production
`CandidateTake(...)` construction site (the same grep discipline as
D-050C3's `DraftClip` audit) found one genuine second break:
`clean_cut_provider.py`'s `_candidate_from_words` (the mixed-trim's
kept child and discarded edge fragments) built a bare `CandidateTake`
that silently dropped `attempt_id`/`realization_id` entirely. Fixed to
carry both forward explicitly -- a mixed trim is a physical word-
boundary adjustment of one already-minted recorded delivery, not a
genuinely new semantic realization, the same reasoning `human_boundary_
polish_v5`'s own fragment splitting already uses via `dataclasses.
replace()`. `source_span_id` is deliberately NOT carried forward (a
mixed trim IS a new physical span; re-minting that id is out of this
directive's scope). `attempt_reconstruction.py`'s own `_merge_attempt`/
`_preserve_borderline_subspans`, and `hybrid_gold_reconciliation.py`'s
`combined` CandidateTake (confirmed ephemeral -- used only for a local
coverage comparison, never persisted into any real candidate pool),
were both audited and confirmed clean; `lexical_self_correction.py` has
the same class of bare-construction bug but is confirmed dead code (no
caller anywhere in the live call graph) and was left untouched.

**Sections 4-6 (PRE_GROUP_REJECTED vs true orphan)**: `realization_
resolver.py`'s `resolve_orphan_realizations_shadow` now draws a
three-way verdict instead of two, using ONLY `DiscardRecord.
discarding_stage` (set once, unconditionally, by the Ledger's own
read-only reconstruction) as the discriminator:
- a verified replacement is always `REPLACEMENT_VERIFIED_SAFE`,
  regardless of origin (unchanged);
- absent that, a discard whose `discarding_stage != "hybrid_editorial_
  chunks"` is `PRE_GROUP_REJECTED` (new) -- it never reached hybrid
  editorial's own semantic judgment at all, so D-049 Case A's concern
  does not apply; does not force Freeze to block;
- absent that, a discard whose `discarding_stage == "hybrid_editorial_
  chunks"` remains `REVIEW_REQUIRED` (unchanged) -- this candidate WAS
  judged meaningful/unique enough to record a `delete_basis`, with no
  verified replacement; D-049 Case A's exact shape; still blocks Freeze
  unconditionally. No fake semantic_idea_id is ever minted to eliminate
  an orphan count; the Ledger still records every `PRE_GROUP_REJECTED`
  realization's full provenance (realization_id, clip_ids, discard
  stage/reason, replacement status) via the same `orphan_reviews` list
  already surfaced in `diagnostics["realization_resolver_shadow"]`.

**Section 8 (behavioral parity)**: full D-050 series (A/B/C1/C1.5/C1.6/
C2/C3/D1) all green, unchanged counts. All 54 CleanCutBench fixtures
green under LEGACY and explicit AUTHORITATIVE, unchanged. Full
`tests/test_cutsell_*.py` glob green under the default (unset) env var,
byte-for-byte LEGACY parity. One measured, expected, and welcome side
effect: forcing `AUTHORITATIVE` across `test_cutsell_universal_clean_
cut.py`'s pre-D-050A synthetic fixtures (bare `DraftClip`s with no
identity stamping) previously showed 4 fail-closed differences from
LEGACY (D-050C3); with the new `PRE_GROUP_REJECTED` classification, 3
of those 4 now resolve correctly -- their "loser" realization is
exactly an ordinary discarded clip with no `hybrid_editorial_chunks`
provenance, which no longer force-blocks Freeze. The 4th (a `repair_
loop.attempt_count` 0-vs-1 difference, unrelated to orphan
classification -- the Ledger finds zero semantic ideas at all in that
specific fixture, so the authoritative pass is a true no-op on an
already-repaired draft) persists, unchanged, still a benign synthetic-
fixture artifact.

**Tests**: `tests/test_cutsell_d050d1_early_identity_minting.py` (new,
12 tests) -- every candidate receives `realization_id` before
`apply_clean_cut` runs, clean_cut-kept/discarded candidates both retain
it, the mixed-trim fix (kept child + edge fragments retain parent
identity), no-double-minting, timestamp-independence, the `PRE_GROUP_
REJECTED`/`REVIEW_REQUIRED` distinction (both directions -- an ordinary
reject vs a hybrid_editorial semantic delete), a verified replacement
staying safe regardless of origin, a `PRE_GROUP_REJECTED` discard never
blocking Freeze, a hybrid_editorial delete without replacement still
escalating, the Ledger recording full pre-group-rejected provenance,
and one full-pipeline run confirming the relocation changed nothing
about bucket assignment.

No Modal RAW launched. No RunPod touched. No Unified Realization
Resolver idea-level decision logic changed (only the orphan/pre-group
classification, which the directive explicitly asked to be
introduced). No Human Gold, grouping, or CleanCut decision logic
touched.

## D-051 -- front-half determinism + hybrid budget audit (report only)

Three completed AUTHORITATIVE-mode Modal Video00 canaries (33701641177 /
`324233b`, 33727122331 / `5e83abe`, 33733618943 / `fce4588`) were audited
against each other. Findings, all report-only, no code changed:

- ASR `segment_count` on the identical `SOURCE_KEY` video already diverges
  (48 / 54 / ~5x) before any candidate/attempt/clean_cut logic runs --
  this is the FIRST material divergence stage, not clean_cut or hybrid
  editorial. `asr.py`'s `FasterWhisperASR` left every Faster-Whisper decode
  parameter beyond `word_timestamps`/`vad_filter` as a silent library
  default (`compute_type="auto"`, unpinned `beam_size`/`best_of`/
  temperature fallback ladder/`condition_on_previous_text`), and the
  library exposes no seed control for its beam-search/temperature-fallback
  decoding at all.
- The hybrid-editorial-chunks stage's `$0.0075` `DollarBudgetLedger`
  (`hybrid_google_transport.py`) is a single per-run ledger shared across
  every chunk call in plain `chunk_index` order; all 3 runs show exactly 4
  chunks receiving real provider evaluation and everything requested
  beyond that (1 chunk in runs 1-2, 2 chunks in run 3, since requested
  count itself drifted 5->5->6 as a downstream consequence of the ASR
  variance above) failing deterministically with
  `RuntimeError("hybrid edit/test dollar budget exhausted")` -- a pure
  function of call order, with zero regard for what a chunk contains.
- The Unified Selection Reasoner and the semantic-equivalence arbiter each
  carry their own, separate `DollarBudgetLedger` (never shared with hybrid
  editorial chunks); neither StoryValidator's claim/clause arbiters nor
  the semantic-equivalence arbiter were ever actually invoked in any of
  the 3 runs (0 arbiter consultations) -- their budgets are not a live
  factor in the observed variance.
- The latest canary's 4 true hybrid `REVIEW_REQUIRED` orphans all
  genuinely received real provider evaluation (verified: a
  budget-exhausted chunk always shows `decisions: []`, and
  `delete_basis="high_confidence_semantic"` is only reachable from a
  non-empty decision) -- they are real editorial decisions, not
  incomplete-evaluation artifacts.
- Verdict: current Video00 canaries are only PARTIALLY comparable
  experiments. See the full report delivered to the user for the complete
  run table, budget trace, and commercial-runtime recommendation (D-052
  Section 8/9 below implements the accepted part of that recommendation).

## D-052 -- deterministic ASR evidence + semantic compute planner

Closes the two D-051-proven stability gaps architecturally, both
additive-only and flag-gated OFF by default (no live behavior change
without an explicit opt-in env var; no Modal RAW run as part of this
change).

**Part A -- `canonical_asr_evidence.py` (new module).**
`CanonicalASREvidence` is a provider-neutral contract for "what was
said": a flat, source-ordered `normalized_words` word stream + language +
ASR model/config fingerprint + `evidence_hash`. `evidence_hash` is
computed from normalized word TEXT only -- never from timestamps or
segment boundaries -- so two transcriptions of the same words always
match regardless of how Whisper happened to group or time them.
`normalize_transcript_segments` deterministically re-derives segment
boundaries from the flattened word timeline using only a word-timestamp
gap (`0.75s`, the same constant `take_segmentation.py`'s own
`_speech_units` already used -- reused, not widened, per the directive's
explicit "do not globally widen thresholds") and sentence-ending
punctuation, discarding Whisper's own per-segment shape entirely.
`asr.py`'s `FasterWhisperASR` gained explicit `beam_size`/`best_of`/
`temperature_ladder`/`condition_on_previous_text`/`word_timestamps`/
`vad_filter`/`initial_prompt` fields (every default value identical to
faster-whisper's own documented library default -- audit/fingerprint
only, not yet threaded into the live `model.transcribe()` call, since
this sandbox has no faster-whisper install to verify the actually-pinned
GPU-image version's defaults against; wiring them in is the recommended,
separately-tested follow-up) plus a `config_fingerprint()` method and a
`load_asr_provider_from_env` builder with an explicit
`CUTSELL_ASR_COMPUTE_TYPE` override (default unchanged: `"auto"`).
`take_segmentation.segment_takes` gained the
`CUTSELL_ASR_CANONICAL_NORMALIZATION` flag (default OFF): when enabled,
every source's segments are canonically re-normalized before any of
`segment_takes`'s own existing logic runs.

**Part B -- `semantic_compute_planner.py` (new module).**
`SemanticWorkItem`/`SemanticWorkPriority` (P0 safety-critical / P1 retry-
equivalence / P2 editorial-quality) / `SemanticComputePlan` /
`build_semantic_compute_plan` -- a provider-neutral (no Gemini-specific
field anywhere in this module) planner that computes, BEFORE any paid
call, which work items fit a cost ceiling via a stable priority sort (P0
always reserved first, ties keep original relative order, so the
caller's own iteration order never changes which priority tier executes
first). `hybrid_session_cleanup.py` gained the
`CUTSELL_SEMANTIC_COMPUTE_PLANNER` flag (default OFF, preserving today's
exact `chunk_index`-order dispatch): when enabled,
`apply_hybrid_session_cleanup` builds every window across every partition
first, classifies each into P0 (via a pre-call, deterministic reuse of
`final_sibling_grouping`'s own `_negations`/`_numbers` extractors -- a
window containing a same-idea pair that disagrees on negation/number is
P0) / P1 (contains a `_failed_local_evidence`-corroborated candidate,
already pre-call-available) / P2 (everything else), plans via
`build_semantic_compute_plan`, and dispatches every window in planned
order instead of enumeration order -- the real `DollarBudgetLedger`
remains the sole authoritative spend enforcement; only WHICH window is
attempted first changes, which is what guarantees a P0 window is never
starved purely for having been requested fifth or sixth. Diagnostics gain
`planner_execution_rank`/`planner_priority`/`planner_predicted_planned`
per window (all `None` when the flag is off) and
`HybridSessionCleanupResult.semantic_compute_plan` (`None` when off).
`build_cost_contract_report` renders the D-052 Section 9 predictable
per-video cost contract (estimated pre-execution, actual post-execution)
from an already-built plan.

Tests: `tests/test_cutsell_d052_canonical_asr_evidence.py` (14) --
evidence-hash content/timestamp independence, 2-vs-3-segment and
50/100/250ms-jitter/VAD-boundary/punctuation equivalence classes, and the
`segment_takes` flag's off-by-default parity + on-mode segment-boundary
independence. `tests/test_cutsell_d052_semantic_compute_planner.py` (13)
-- the 9 named planner cases (4/6 items sufficient budget, 6 items
constrained budget, P0-before-P2 regardless of arrival order, order-
independence, cost-ceiling respected, P0/P2 exhaustion actions, the exact
D-051 "fifth call" positional-starvation shape) plus 4
`hybrid_session_cleanup.py` integration tests proving the flag's off-mode
parity and on-mode fix.

Validation: compileall clean. Full D-050+D-052 series: 177 passed (150
pre-existing D-050 + 27 new). All 54 CleanCutBench fixtures green under
LEGACY and explicit AUTHORITATIVE, including with both new flags forced
on simultaneously. Full `tests/test_cutsell_*.py` glob: 1501 passed (1474
pre-existing + 27 new), env unset (both flags off, LEGACY default) --
byte-for-byte parity confirmed by the exact count match. No GPU/Modal RAW
run as part of this change.

## D-053 -- ASR determinism qualification (isolated ASR-only phase; COMPLETE -- determinism NOT achieved)

Follows directly from the D-052 stability battery's live finding: 3
identical-config Modal Video00 RAWs produced 3 different
`CanonicalASREvidence.evidence_hash` values on the byte-identical source
video/config -- ASR word-sequence output itself is not deterministic, even
with D-052's canonical normalization layer on. This phase isolates and
qualifies a fix for ASR determinism specifically, without touching any
downstream stage (Unified Realization Resolver, Semantic Ledger,
AttemptReconstructor, Human Gold, hybrid editorial, Semantic Compute
Planner's ordering/decisions, CanonicalEditPlan, Freeze, Render/QC) and
without running another full Video00 RAW.

**Ground-truthing method.** Rather than assume faster-whisper's documented
defaults (which can silently drift version-to-version), the exact pinned
wheel this repo installs (`faster-whisper==1.0.0`,
`requirements.cutsell.worker.txt`) was downloaded and its
`transcribe.py`/`vad.py` source read directly. Confirmed:
`temperature` defaults to the LADDER `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, not
a deterministic scalar -- the primary attempt is temperature 0.0
(deterministic beam search), but a segment escalates to a later, sampling
(non-deterministic) rung whenever it fails the library's own quality gates
(`compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
`no_speech_threshold=0.6`), and `condition_on_previous_text=True`
compounds any such escalation forward through the rest of the transcript
via its context window. This is the confirmed mechanism (not merely a
leading suspect, as D-051/D-052 had it) for the observed run-to-run
word-sequence variance. `VadOptions` defaults were ground-truthed the same
way (`threshold=0.5, min_speech_duration_ms=250,
max_speech_duration_s=inf, min_silence_duration_ms=2000,
window_size_samples=1024, speech_pad_ms=400`).

**`asr.py` rewrite.** `FasterWhisperASR` now threads every one of these
ground-truthed parameters explicitly into the live `model.transcribe()`
call (previously audit/fingerprint-only fields per D-052) -- for the
LEGACY (default-constructed) provider every value is identical to the
library's own default, so this is a zero-behavior-change,
make-explicit-and-fingerprintable change. `DETERMINISTIC_TEMPERATURE =
(0.0,)` is the ONE deliberate difference: a scalar temperature with no
fallback rung for the decode-with-fallback loop to ever escalate to.
`build_deterministic_asr_provider()` builds this candidate;
`load_asr_provider_from_env()` gained `CUTSELL_ASR_DETERMINISTIC_CONFIG`
(default OFF, LEGACY unchanged) to select it.
`canonical_asr_evidence.ASRConfigFingerprint` was extended with every new
field (`task`, `patience`, `length_penalty`, `repetition_penalty`,
`no_repeat_ngram_size`, the quality-gate thresholds,
`prompt_reset_on_temperature`, `vad_parameters`) plus a derived
`sampling_fallback_enabled` property (`len(temperature_ladder) > 1`).
`compute_type` stays `"auto"` for both providers -- pinning precision
(e.g. `float16`) is deliberately deferred as a separate, separately-tested
change, not bundled into this determinism fix.

**Isolated ASR-only harness.** `cutsell_worker/asr_only_benchmark.py`
(new): `run_asr_only_benchmark()` downloads the same Video00 source,
transcribes it via `load_asr_provider_from_env()` (so it automatically
respects `CUTSELL_ASR_DETERMINISTIC_CONFIG` exactly like the real
pipeline), builds `CanonicalASREvidence`, and stops -- never constructing
AttemptReconstructor, hybrid editorial, the resolver, or render/QC.
`collect_asr_runtime_audit()` ground-truths the live installed
faster-whisper/ctranslate2 versions and the actual installed
`WhisperModel.transcribe` signature via `inspect.signature` on the live
import, GPU model and CUDA runtime via the `nvidia-smi` CLI (deliberately
never `torch` -- `cutsell_worker` is a torch-free package by architectural
contract, enforced by
`tests/test_cutsell_clean_worker_dependency_boundary.py`). Modal transport:
`modal_asr_only_benchmark.py` (new; mirrors
`modal_video00_full_benchmark.py`'s image/app/secret pattern exactly) runs
on a separate Modal App (`MODAL_ASR_ONLY_APP_NAME`, never the full-engine
app) with its own, materially shorter timeout ceiling
(`DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S=900s` vs. the full RAW's 5400s). CI
entry point: `.github/workflows/cutsell-asr-only-modal.yml`
(`workflow_dispatch`, one boolean input toggling
`CUTSELL_ASR_DETERMINISTIC_CONFIG` per dispatch so both baseline and
candidate runs use the same workflow).

**Two D-052 observability-only bugs found live by the stability battery,
fixed this phase (never changing semantic execution ordering or provider
decisions):**
1. `diagnostics.semantic_compute_plan` showed `absent_flag_off` in all 3
   battery runs despite the flag being on and the planner demonstrably
   running (per-window `planner_priority` diagnostics were correctly
   populated). Root cause: several of `composite_resolver.py`'s 19 chain
   hooks reconstruct `HybridSessionCleanupResult` via a keyword
   constructor call written before `semantic_compute_plan` existed,
   silently reverting it to `None`. Fixed via the same `ContextVar`
   side-channel pattern this codebase already established for
   `_SPLIT_IDS`/`_COMPOSITE_SPLIT_IDS`
   (`hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN`, set
   unconditionally at the base call, read-and-cleared once by
   `composite_resolver._restore_semantic_compute_plan`) -- no hook file
   needed editing.
2. `planner_predicted_planned` was `false` for every window in all 3
   battery runs even though the real ledger accepted 4 of 6, because
   `_estimate_window_cost_usd`'s old flat `0.0015 USD/member` heuristic
   overshot real cost badly enough that a typical 10-member window's
   estimate alone exceeded the entire default `$0.0075` ceiling. Fixed by
   reusing the SAME real token-based formula the live transport bills
   against (`HybridProviderSettings.estimate_cost_usd`, fed by
   `estimate_tokens_from_chars` and an output-token tiering duplicated
   from `hybrid_google_transport._compact_output_token_ceiling`) --
   estimate/bookkeeping only; dispatch order never depended on estimate
   accuracy.

**Tests (all new, all green):** `tests/test_cutsell_d053_asr_determinism.py`
(14) -- explicit transcribe-parameter wiring, deterministic-vs-legacy
temperature shape, VAD/compute-type explicitness, fingerprint
stability/sensitivity, `CUTSELL_ASR_DETERMINISTIC_CONFIG` on/off parity.
`tests/test_cutsell_asr_only_benchmark.py` (7) -- the isolated harness
never invokes editorial stages, respects the env-driven config loader,
JSON-serializable result. `tests/test_modal_asr_only_benchmark.py` (14) +
`tests/test_modal_gpu_config.py` additions (8) -- Modal
App/Image/Function/Secret wiring, distinct app name/timeout/payload-env
from the full-engine benchmark. `tests/test_cutsell_composite_resolver.py`
additions (4) and `tests/test_cutsell_d052_semantic_compute_planner.py`
additions (4) -- the two observability fixes above, including a direct
unit test of `_restore_semantic_compute_plan`'s never-overwrite and
clear-after-read behavior.

**Validation so far:** compileall clean across `cutsell_worker/`,
`modal_asr_only_benchmark.py`, `modal_gpu_config.py`, and `tests/`. Full
`tests/test_cutsell_*.py` glob: 1531 passed (1501 pre-existing + 30 new),
zero regressions. `tests/test_cutsell_clean_worker_dependency_boundary.py`
(the torch-free architectural contract) still passes -- the runtime-audit
GPU/CUDA introspection deliberately uses `nvidia-smi`, never `torch`.

**Live Modal audit (Section 1, ground-truthed on the actual L4 container):**
`faster_whisper==1.0.0`, `ctranslate2==4.8.2`, Python `3.10.12`, GPU
`NVIDIA L4`, `nvidia-smi`-reported driver CUDA version `13.0`,
`cuda_available=true`. `compute_type="auto"` confirmed unchanged (Section
4 tradeoff data deferred, per the directive, to a separate future task).

**Live 3+3 ASR-only battery (Sections 6/8/9, one source, one commit,
`CUTSELL_ASR_DETERMINISTIC_CONFIG` toggled per run):**

| run | config | `evidence_hash` | words | elapsed_sec |
|---|---|---|---|---|
| A | LEGACY (flag OFF) | `asrev_4635ffbc25094ca1157d5e53` | 623 | 42.1 |
| B | LEGACY (flag OFF) | `asrev_afb71a01dd24859f4635ec0a` | 623 | 37.5 |
| C | LEGACY (flag OFF) | `asrev_fac7fb4c1e601c8e6228c80b` | 623 | 30.7 |
| D | deterministic (flag ON) | `asrev_37ac9540809c3d2762175a97` | 623 | 35.3 |
| E | deterministic (flag ON) | `asrev_62b25b19d9d4d7a78e1594fa` | **650** | 29.2 |
| F | deterministic (flag ON) | `asrev_d54c748fd7098230456aecb6` | 623 | 36.2 |

`asr_config_fingerprint` was identical across all 3 LEGACY runs
(`asrcfg_bccad85973b58720`) and identical across all 3 deterministic runs
(`asrcfg_664a49ba49f6baf6`, `sampling_fallback_enabled=false` confirmed in
every deterministic run) -- configuration itself is stable; every hash
difference below is genuine audio-decode variance, not a config drift
artifact.

**Result: 0/3 identical in BOTH configurations.** LEGACY produced 3
different hashes with a constant word count (623/623/623) -- word-level
substitutions only. The deterministic candidate ALSO produced 3 different
hashes, and one run (E) diverged in word COUNT too (650 vs. 623) -- a
larger, more concerning divergence than anything LEGACY showed. Removing
the temperature fallback ladder (the mechanism D-051/D-052/D-053's own
analysis had named as the confirmed, capable-of-causing-it culprit)
measurably changed the config fingerprint but did **not** reduce
observed non-determinism at all. Per Section 9's own acceptance
criteria ("if words differ, report exactly, do NOT compensate
downstream"): the root cause is NOT (solely, or even primarily) the
decode-with-fallback loop -- it sits deeper, most plausibly in
non-deterministic GPU floating-point reduction order inside
ctranslate2's CUDA beam-search/decode kernels themselves (a well-known
class of run-to-run GPU non-determinism, independent of temperature or
any faster-whisper-level setting, and out of this phase's scope to fix --
forcing deterministic CUDA/cuDNN algorithms is a separate, much larger,
not-yet-authorized change).

**Recommended ASR config: LEGACY (`CUTSELL_ASR_DETERMINISTIC_CONFIG`
stays OFF by default).** The deterministic candidate is not disqualified
architecturally (it is still a defensible, more explicit configuration:
no risk of ever escalating to a random sampling temperature), but it
delivered zero measured reproducibility benefit in this live battery, so
there is no evidence-based case to flip the flag on yet. Both providers
remain fully available behind the flag for future investigation once a
GPU-kernel-level determinism fix is separately scoped.

**Semantic compute observability fixes:** validated by the 8 new targeted
tests (`test_cutsell_composite_resolver.py`/
`test_cutsell_d052_semantic_compute_planner.py` additions) and the full
regression suite, not by this ASR-only harness -- it deliberately never
constructs the hybrid editorial / CompositeResolver stage those fixes
live in.

## D-054 -- ASR variance severity + commercial stabilization (report only; no code)

Re-analyzes D-053's six existing ASR-only artifacts (no rerun) to answer
what D-053 could not: not "do hashes match" but "does the variance touch
anything commercially critical." Full word sequences + per-word
timestamps were recovered from the six Modal job logs (the console
summary only prints the first 40 words; the full arrays are earlier in
the same job's own "Run Modal ASR-only benchmark" step output, which
`get_job_logs` with a larger `tail_lines` does return).

**Confounder found first:** `evidence_hash` embeds `source_asset_id`,
which this harness derives from `benchmark_id` (unique per GitHub Actions
run/attempt) -- so `evidence_hash` is GUARANTEED to differ run-to-run
regardless of ASR content. Proof: runs D and F have word-for-word
IDENTICAL transcripts (0 diff ops, 100% similarity) yet different
`evidence_hash` values. D-053's "0/3 identical hashes" framing is
therefore partly a harness-identity artifact, not solely ASR
non-determinism -- real ASR variance is still present (proven directly
below by actual word-content diffs), but hash inequality alone was never
a clean proxy for it, exactly as D-053 Section 8 had already warned.

**Word-level alignment (classic Levenshtein WER, full backtrace):**
LEGACY A-B 99.84% similar (1 substitution only); A-C 90.69%; B-C 90.85%.
Deterministic D-F 100% identical (0 diffs); D-E 87.54%; E-F 87.54%. In
both groups, two of three runs are near-identical and ONE run (C in
LEGACY, E in deterministic) is the actual source of essentially all
observed variance -- not three independently-diverging runs.

**T0-T4 severity classification (302 diff tokens across all 6 pairwise
comparisons):** T0 (formatting/punctuation) 162, T1 (filler/homophone
orthography, e.g. `si`/`sí`, `aliméntate`/`alimentate`) 14, T2
(inflection/paraphrase, e.g. `pedía`/`pedí`, `salió`/`salía`) 10, T3
(other lexical variance) 104, T4 (negation/number/entity trigger) 12.
The T3/T4 counts look large in isolation but cluster into just ONE
localized real content event per outlier run, not dozens of independent
facts changing:
1. A dropped/reworded descriptive clause about a resolved back-acne
   symptom (present in C/E, absent or garbled as `resor...` elsewhere).
2. The colloquial idiom "no hay que preguntar" (rhetorical "don't even
   ask") becomes "hay que voltar" in C/E -- a real negation deletion
   plus a nonsensical word substitution. This is genuine ASR
   hallucination on a hard, idiomatic, filler phrase -- but it is
   rhetorical/emotional filler, not a clinical statement (contrast
   "no tengo cáncer"), and changes no diagnosis, treatment, number, or
   hereditary fact anywhere in the transcript.
3. A `sonografía`/`sonografías` singular/plural variance on one mention
   of an ultrasound procedure.
The apparent "5"/"-10" insertion/deletion the raw aligner flags in D-E is
a REPHRASING artifact (the same 5-10% hereditary-cancer statistic stated
in two different sentence structures across runs), not a number
appearing or disappearing -- confirmed by the whole-transcript check
below.

**Critical claim stability (`semantic_claims.extract_claims`, run
standalone, no clause-role arbiter, one mega-clip per full transcript):**
raw `canonical_claim_id` intersection across all 6 runs is 11/18 CRITICAL
claims (61.1%); 12/17 (70.6%) within the deterministic group alone. This
number is understated by a real second-order effect, not by lost facts:
`canonical_claim_id` is minted from an exact clause's content-token set,
and ASR's comma/period placement variance shifts where
`_split_into_clauses` draws a clause boundary, producing a "different"
claim id for what is the same underlying sentence reworded only in
punctuation. Proof this isn't real fact loss: `final_sibling_grouping.
_negations()` and `_numbers()`, run over the FULL transcript text (the
same token-set primitives the live engine's own retry-family/coverage
logic already uses), are IDENTICAL across all 6 runs --
`negations={no, nunca}`, `numbers={3, 5, 10, 2023}` -- and a direct
substring check confirms every named entity/diagnosis/hereditary-stat
fact (`cáncer`, `tiroides`, `hereditario`, `sonografía`, `ginecóloga`,
`metabolismo`, the "5-10%" figure, the "In Body Mass" test name) is
present in literally all 6 transcripts. The one real negation event (the
idiom above) is a single clause-level exception inside an otherwise
100%-stable negation/number token set.

**Timestamp drift (aligned identical words only):** median 0.0s in
every pairwise comparison. p95 ranges 0.0-2.06s; max ranges 0.0-7.12s --
but the two byte-identical-content pairs (D-F: 0 diffs) and near-
identical pairs (A-B: 1 diff) show max drift of exactly 0.0s and 0.04s
respectively. The larger p95/max values only appear in pairs that also
have the real content-divergence event above (which shifts every
downstream word's absolute time by the length of the differing clause) --
they are a cascading effect of content variance, not independent timing
jitter. When content genuinely matches, timing is sub-frame-accurate,
comfortably inside `take_segmentation`'s existing 0.75s gap-splitting
threshold.

**Decision: CASE A** (D-054 Section 6) -- hashes differ, but the
transcript is semantically equivalent for every commercially-relevant
purpose and critical claims (numbers, negations, diagnoses, named
entities) are stable in substance. Do NOT chase bit-identical CUDA/GPU
determinism. The one genuine anomaly found (the idiom-corruption
negation drop) is a real, worth-monitoring ASR hallucination on hard
colloquial filler, but is not a clinical/factual claim and does not
change this phase's conclusion.

**Recommended commercial ASR strategy:** keep the current Faster-Whisper
GPU provider (LEGACY config, per D-053) and invest instead in a
Canonical Transcript Equivalence layer -- normalizing harmless
T0-T2 lexical/filler/punctuation/clause-boundary variance to one
semantic evidence representation, so `evidence_hash`-style identity
reflects meaning-equivalence rather than raw token-sequence equality.
Fixed `compute_type`, CPU-only deterministic transcription, two-pass
consensus transcription, and an alternate ASR backend are all CASE-B
remedies; none are needed given this phase's CASE-A finding, and none
are implemented here (informational comparison only, per the directive).

No full Video00 RAW run. No engine code touched (Resolver, Ledger,
CleanCut, Human Gold, Gemini, Render all untouched) -- this phase is
pure offline analysis of already-existing D-053 artifacts.

## D-055 -- Canonical Transcript Equivalence (small stability layer; no editorial changes)

Builds the small, additive stability layer D-054 recommended instead of
chasing bit-identical CUDA. Touches only `canonical_asr_evidence.py`
(plus wiring the two new fields into `asr_only_benchmark.py`'s
diagnostics output) -- Resolver, Ledger, CleanCut, Human Gold, Selection,
Gemini, and Render are all untouched; nothing new is fed into Selection
or claim extraction.

**Section 1 -- separating source-scoped id from content equivalence.**
`CanonicalASREvidence.evidence_hash` (D-052/D-053) is unchanged and kept
as the SOURCE-SCOPED evidence id (still includes `source_asset_id`, for
existing provenance/lineage callers). Two new, purely additive fields:
`content_hash` (`compute_content_hash`) -- identical word-text hashing
logic minus `source_asset_id`, so two independently-dispatched runs over
the byte-identical audio (D-054's exact finding) now correctly match --
and `canonical_equivalence_hash` (`compute_canonical_equivalence_hash`),
a further narrowing on top of `content_hash`.

**Section 2 -- Canonical Transcript Equivalence.** New
`canonicalize_transcript_words()` normalizes ONLY the two variance
classes D-054 actually proved harmless: T0 (punctuation/casing/spacing,
via `final_sibling_grouping._tokens` -- the SAME tested tokenizer the
engine's own retry-family grouping already uses, per this directive's
own "reuse existing normalization machinery" instruction, not a second
hand-rolled regex) and a single bounded T1 accent-orthography pair
(`"sí" -> "si"`, the ONLY accent-only pair D-054's live battery actually
observed and proved harmless). T2 (inflection/paraphrase) is deliberately
NOT normalized -- no deterministic stemmer exists anywhere in this
codebase, and inventing one here would be the "aggressive paraphrase"
this directive forbids.

**Section 3 -- protected semantics, structurally.** Digits, negation
vocabulary, and every named entity/diagnosis word pass through
`canonicalize_transcript_words` completely unchanged by construction: the
accent table above is a 1-entry allowlist that never collides with a
digit or a negation word, so no separate guard was needed to keep numbers/
negation/diagnosis/entity/causal-order distinguishable -- verified
directly by 6 of the Section 7 tests (number/negation/diagnosis/causal-
direction/word-order changes all still produce different
`canonical_equivalence_hash` values).

**Section 4 -- stability-only.** Both new hashes are read only by
`asr_only_benchmark.py`'s own diagnostics dict (so a live dispatch already
shows whether two runs are content-equivalent without an offline replay)
and by this section's own replay script. Nothing in Selection or claim
extraction reads either field; the original transcript stays the sole
authoritative evidence.

**Section 5 -- replay of all six D-054 transcripts** (offline, no GPU --
the saved D-054 artifacts were available, so no rerun was needed).
`content_hash`/`canonical_equivalence_hash` were computed for all six
using synthetic per-run `source_asset_id` stand-ins (the one input the
saved logs do not preserve verbatim; every word/timestamp value is the
real recovered transcript). Result: the six runs collapse into **4**
genuine `canonical_equivalence_hash` classes, not 6 -- and the split is
NOT along the LEGACY-vs-deterministic config line D-054's own framing
implied. Runs A (LEGACY), D (deterministic), and F (deterministic) share
ONE identical canonical-equivalence class (their raw transcripts are, in
fact, byte-for-byte identical to each other); B, C, and E are each their
own singleton class. So within LEGACY: 3 distinct classes (A alone here,
B alone, C alone -- no LEGACY-internal collapse); within the
deterministic group: 2 distinct classes (D+F together, E alone). This is
a materially stronger finding than D-054's own within-group framing: the
"outlier" behavior is not tied to either config -- across all six runs,
4 of 6 (A, D, F) actually agree on one dominant transcription, and only
B, C, E each diverge in their own distinct way. Pairwise canonical-token
Jaccard similarity (unordered bag-of-tokens, naturally more forgiving
than D-054's sequence-order WER numbers): A-B 99.16%, A-C 96.25%, B-C
97.07%, D-E 95.02%, D-F 100.00% (byte-identical, as the shared class
above already showed), E-F 95.02%.

**Section 6 -- critical claim equivalence, re-verified on the canonical
representation.** Re-running `final_sibling_grouping._negations()`/
`_numbers()` on the CANONICAL (post-`canonicalize_transcript_words`)
token representation instead of raw text reproduces D-054's whole-
transcript findings exactly: negation set `{no, nunca}` and number set
`{3, 5, 10, 2023}` are IDENTICAL across all 6 runs. Direct fact-presence
checks on the canonical representation confirm every checked fact
(cáncer, tiroides, hereditario, the 5-10% hereditary-cancer stat,
sonografía, ginecóloga, metabolismo, the resorcina treatment mention) is
present in literally all 6 canonicalized transcripts. No CTA phrase
exists anywhere in this particular source video in any of the 6 runs
(vacuously stable -- nothing to lose, not a failure).

**Section 7 -- tests** (19 new,
`tests/test_cutsell_d055_canonical_transcript_equivalence.py`): identical
words + different `source_asset_id` -> same `content_hash` (and
`evidence_hash` still differs, proving the separation is real);
punctuation/casing/spacing differences -> same
`canonical_equivalence_hash`; the proven-safe `si`/`sí` pair -> same; an
UNPROVEN accent pair (`mas`/`más`) -> deliberately stays distinguishable
(fail conservative); number change, negation change (the directive's own
"no tenía" vs "tenía" example), diagnosis-noun change (the directive's
own "gastritis" vs "alergia" example), causal-direction change, and
word-order change -> all different; timestamp-only changes -> same
`content_hash` and `canonical_equivalence_hash`; plus direct unit
coverage of `canonicalize_transcript_words` (punctuation stripping, the
accent table applying ONLY to its one entry, digits/negation passing
through untouched, a pure-punctuation "word" vanishing) and two
D-054-story-shaped replay tests (a punctuation-only pair collapsing to
one class; the real "no hay que preguntar"/"hay que voltar" pair staying
distinct).

**Validation:** compileall clean.
`tests/test_cutsell_d050*.py tests/test_cutsell_d052*.py
tests/test_cutsell_d053_asr_determinism.py
tests/test_cutsell_asr_only_benchmark.py
tests/test_cutsell_d055_canonical_transcript_equivalence.py
tests/test_cutsell_composite_resolver.py
tests/test_cutsell_clean_cut_core_evaluation_suite.py`: 286 passed (this
single evaluation-suite file IS all 54 CleanCutBench fixtures run
through the real production chain under LEGACY;
`test_cutsell_d050c1_5_full_cleancutbench_parity.py`, part of the same
run, captures every one of those same 54 fixtures' real takes and
independently cross-checks AUTHORITATIVE shadow-resolver parity against
them -- both modes are exercised in one pass, nothing skipped). Full
`tests/test_cutsell_*.py` glob: 1550 passed (1531 pre-existing + 19 new),
zero regressions. No GPU/Modal dispatch was needed or used.

## D-056 -- Final full-engine stability battery / commercial validation gate (report only; no code)

Directive: run exactly three full Modal Video00 RAWs at the fixed D-052-era
config (`CUTSELL_UNIFIED_REALIZATION_RESOLVER=AUTHORITATIVE`,
`CUTSELL_ASR_CANONICAL_NORMALIZATION=1`, `CUTSELL_SEMANTIC_COMPUTE_PLANNER=1`,
`CUTSELL_HYBRID_LLM_ENABLED=1`, `CUTSELL_HYBRID_PROVIDER=google`), same
source, same Human Gold manifest, same commit, zero code changes between
runs, and deliver a 10-section stability/commercial-readiness report.

**Two runs (A, B) completed cleanly and were fully analyzed. A valid third
run could not be obtained in-session** despite five workflow_dispatch
attempts, documented below as its own finding (Section 8 below).

### Section 1 -- Transcript equivalence
Both runs' `stage_status.canonical_asr_evidence` reported `status: complete`
with `canonical_normalization_applied: true`. Exact evidence_hash /
content_hash / canonical_equivalence_hash digit comparison could not be
certified: `cutsell-video00-modal-raw.yml`'s console-log secret masking
(unlike the fixed `cutsell-asr-only-modal.yml` from D-053) blanket-masks
every numeric-looking RunPod-template env value, and GitHub then redacts
that literal substring everywhere else it recurs in the log -- corrupting
every hash string and every printed count in both runs' `Print full
canonical diagnostics` output. Structural (non-numeric) evidence that
survived masking: Run A `raw_segment_count`=55 -> `normalized_segment_count`=49;
Run B raw≈4X (masked) -> normalized=45. Retry-family count (4 vs 6) and
resolved-idea count (17 vs 18) differ between runs -- evidence of genuine
ASR/semantic-clustering variance, not just formatting/punctuation.
**TRANSCRIPT EQUIVALENCE: not certifiable as HARMLESS this battery** --
recommend fixing the masking-threshold bug in this workflow (same class of
bug fixed in D-053 for the ASR-only workflow) before the next stability run.

### Section 2/3 -- Front-half + semantic compute plan
Run A: 4 retry families (`take_judge_groups`), `semantic_idea_equivalence`
candidate_pair_count=58/merged_pair_count=5, IdeaClusterer resolved 17
semantic ideas (16 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE), 5
PRE_GROUP_REJECTED, 2 pre-group REVIEW_REQUIRED verdicts.
Run B: 6 retry families, candidate_pair_count=59/merged_pair_count=5 (plus
one `distinct_addition_blocked` entry not seen in A), resolved 18 semantic
ideas (17 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE), 4 PRE_GROUP_REJECTED, 2
REVIEW_REQUIRED, 1 REPLACEMENT_VERIFIED_SAFE verdict not seen in A.
`diagnostics.semantic_compute_plan` was present in both (planner active)
but every numeric field (`cost_ceiling_usd`, `deferred_call_count`,
`eligible_work_item_count`, `estimated_semantic_cost_usd`, tier cost
buckets) was masked; the schema does not expose per-tier (P0/P1/P2) call
counts at all, only cost buckets, so P0/positional-starvation cannot be
independently verified from these diagnostics either run.

### Section 4 -- Unified resolver
Run A: 16 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE, top-level
`status: REVIEW_REQUIRED` (2 `unresolved_orphan_realization_ids`), 3/17
ideas with `legacy_vs_authoritative_same: false`, 0 ideas with a non-empty
`missing_critical_claim_ids`, no invalid composites, no silent
zero-realization ideas. Run B: 17 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE,
same top-level `REVIEW_REQUIRED` shape, 3/18 ideas with
`legacy_vs_authoritative_same: false`, 0 critical-claim-missing ideas, no
invalid composites, no silent zero-realization ideas.

### Section 5 -- Semantic selection stability (the key finding)
Both runs independently hit a genuine negation contradiction that blocked
Selection Freeze -- but **on different retry families**: Run A's
contradiction is in the "problemas de estómago / no hay que preguntar"
family; Run B's is in a "hereditary cancer statistics" family the
SemanticArbiter merged differently this run. `FinalEditReviewer` finding
kinds differ too: Run A = CONTRADICTION + 5x UNIQUE_FACT_LOST +
CRITICAL_CLAIM_LOST (blocking) + REQUIRED_CONTINUATION_LOST (warning); Run
B = DUPLICATE_IDEA + UNRESOLVED_RETRY + CONTRADICTION + 2x
UNIQUE_FACT_LOST, **no** CRITICAL_CLAIM_LOST. Human Gold check-level
stability (the most concrete measurable proxy available): 10/18 checks
identical pass/fail across A and B (56%), 3 consistently fail in both,
8 flip. **SEMANTIC_SELECTION_STABILITY_PERCENT ~= 56%** by this measure;
retry-family count, idea count, and contradiction location all differ on
top of that. Per-idea instability list: the stomach/gastritis family (A)
and the hereditary-cancer-statistics family (B) are each unstable in the
sense that which family becomes the blocking contradiction is itself
run-dependent.

### Section 6 -- Human Gold (18-check authoritative validator)
Run A: 12/18 (fails: pimples_micro_1/2/3_present, pimples_later_winner_present,
gastritis_preserved, pimples_micro_order). Run B: 10/18 (fails:
sonography_good_take_part1_present, sonography_good_take_completion_present,
sonography_bad_take_absent, papillary_cancer_preserved,
pimples_micro_3_present, gastritis_preserved, pimples_micro_order,
sonography_good_before_diagnosis). PASS-in-both (7): cancer_hook_preserved,
biopsy_nodule_preserved, acne_back_preserved, pimples_bad_monolith_absent,
hair_loss_preserved, family_context_preserved, cta_preserved. FAIL-in-both
(3): pimples_micro_3_present, gastritis_preserved, pimples_micro_order.
FLIPPING (8): sonography_good_take_part1_present,
sonography_good_take_completion_present, sonography_bad_take_absent,
papillary_cancer_preserved, pimples_micro_1_present, pimples_micro_2_present,
pimples_later_winner_present, sonography_good_before_diagnosis. Not
patched (per directive).

### Section 7 -- Freeze/delivery
Identical shape both runs: `selection_boundary_contract.status =
not_frozen_freeze_blocked_by_coherence_review` (a premature freeze was
computed then correctly superseded once StoryValidator found the
contradiction), architecture check confirms Clean Cut Core V1 active and
the whole-video Unified Selection reasoner correctly inert, `live_render_qc
= not_attempted` (Boundary/Render never runs on an unfrozen Selection, by
design), `delivery_status = NOT_DELIVERABLE_not_attempted`,
`deliverable = false`. **PRODUCT DECISION (block-for-human-review, do not
ship) is stable across both completed runs.**

### Section 8 -- Cost / Run C dispatch saga
Real per-run GPU wall time (from job step timestamps, not masked): Run A
benchmark step ~6m10s, Run B ~4m44s. Exact $ COGS is not computable: no
Modal-dollar-cost field exists anywhere in the diagnostics, and
`semantic_compute_plan.estimated_semantic_cost_usd` is masked in console
output for both runs. **Operational finding, not a code defect:** across
five `workflow_dispatch` attempts at a third run, `mcp__github__actions_list`
(`list_workflow_jobs`)/`actions_get` (`get_workflow_run`) reported the
`Run Modal full Video00 benchmark` step as `in_progress` continuously for
2-9+ hours per attempt with the job's own `updated_at`/step timestamps
never advancing, on a step whose own real completion (confirmed every time
by cancelling and reading the resulting -- now terminal -- step timestamps)
took 2.5-6 minutes. The only way found to force a fresh read was
`cancel_workflow_run`, but cancelling **always** produces
`conclusion: cancelled` on that step and a downstream chain with no usable
`artifact/video00-modal.json` (the wrapper only writes its local
`modal-video00-result.json` marker on the `modal run` CLI's own successful
exit; a SIGTERM at any point -- even after the real Modal computation
already finished -- prevents that marker from ever being written, so
`Verify frozen Selection lock` gets skipped and architecture/regression-QA
report `No result JSON was downloaded`). Confirmed on 4 of 4 cancelled
attempts. This makes cancellation **structurally unable** to recover data
from this specific workflow, and the `in_progress`-status staleness
appears to be a real, reproducible defect (in this session's view into the
Actions API, the workflow's own runner reporting, or both) independent of
Modal itself. Recommend: (a) fix the masking-threshold bug (Section 1);
(b) make the wrapper write its local result marker incrementally/earlier so
a cancelled run's already-completed data isn't discarded; (c) investigate
the `in_progress` staleness before relying on this workflow for future
timed stability batteries.

### Section 9 -- Commercial pass criteria
Critical claims: NOT stable (Run A found one blocking CRITICAL_CLAIM_LOST,
Run B found none -- presence/absence of this finding itself flips).
No unsafe silent discard in either run (every discard traces to the
StoryValidator contradiction-block, never a silent drop). No invalid
composite in either run. Product decision (ship/no-ship) stable (both
block). Semantic-selection stability low (~56% by Human Gold proxy,
retry-family/idea counts differ). Transcript differences not certified
harmless (Section 1). No P0 budget-starvation evidence found, but the
schema doesn't expose the data needed to fully rule it out. Given these,
Video00 does **not** clear the READY bar this battery.

### Final decision (verbatim, as delivered)
D-056 COMPLETE
TRANSCRIPT EQUIVALENCE: FAIL (not certifiable -- console masking bug; retry-family/idea counts differ between runs)
CRITICAL CLAIM STABILITY: FAIL (blocking CRITICAL_CLAIM_LOST present in A, absent in B)
SEMANTIC COMPUTE PLAN: PARTIAL (planner active both runs; P0/P1/P2 call-count fields not exposed by the schema, cost fields masked)
SEMANTIC_SELECTION_STABILITY: ~56% (10/18 Human Gold checks stable; contradiction location, retry-family count, and idea count all differ)
Human Gold: A: 12/18, B: 10/18, C: N/A (five dispatch attempts did not yield valid data -- see Section 8)
CONSISTENT FAILURES: pimples_micro_3_present, gastritis_preserved, pimples_micro_order
FLIPPING CHECKS: sonography_good_take_part1_present, sonography_good_take_completion_present, sonography_bad_take_absent, papillary_cancer_preserved, pimples_micro_1_present, pimples_micro_2_present, pimples_later_winner_present, sonography_good_before_diagnosis
PRODUCT DECISION STABILITY: PASS (both runs independently blocked for human review; delivery gate never fired on unreviewed content)
AVERAGE COGS: NOT COMPUTABLE (no dollar-cost field exists in diagnostics; GPU wall time ~5-6 min/run)
VIDEO00 ENGINE STATUS: MOSTLY_STABLE (the safety mechanism -- contradiction detection blocking Freeze -- fires correctly and consistently; which content triggers it, and how many Human Gold checks pass, is not yet stable)
VIDEO00 DEVELOPMENT STATUS: MORE WORK REQUIRED
READY FOR VIDEO01/VIDEO02/VIDEO03? NO
Then STOP. Did not patch. Did not launch a further run beyond the five
dispatch attempts already documented in Section 8.

## D-056.1 -- Benchmark execution reliability only (infra, no engine-semantic change)

Issued directly off D-056 Section 8's root-cause findings. Explicitly scoped to
workflow/wrapper/masking/observability fixes: no Resolver, Ledger, ASR, Human
Gold, Selection, or Freeze code touched. Zero live GPU dispatch during this
directive -- offline validation only, same commit head
(`feature/runpod-pod-on-demand`).

### 1. Benchmark result persistence (item 1)
D-056 Run C root cause: `modal_video00_full_benchmark.py`'s local entrypoint
only wrote `modal-video00-result.json` AFTER its blocking
`run_video00_benchmark.remote(payload)` call returned cleanly -- a SIGTERM to
the local `modal run` process at any point before that write permanently
discarded the result, even when the remote computation had already fully
succeeded (confirmed on 4 of 4 cancelled Run C attempts).

Fix: `run_video00_benchmark` (the REMOTE function) now also persists its
compact summary directly to S3, from inside the remote call itself, before it
ever returns to the local caller -- `_persist_benchmark_result()`, keyed by
`modal_gpu_config.benchmark_result_s3_key(benchmark_id)` (new namespace:
`cutsell/benchmark-results/{benchmark_id}/compact-result.json`, deliberately
separate from `serverless_handler._focused()`'s own full-diagnostics
`cutsell/serverless/{benchmark_id}/result.json` upload). Uses the same AWS
credentials already injected into the remote container via
`cutsell_env_secret` -- no second credential path. Never raises: a
persistence failure never turns an otherwise-successful benchmark into a
reported failure. The persisted/local result additionally carries a new,
purely-observational `benchmark_result_uri` field.

### 2. Stale-status recovery / poll-not-trust (item 2)
`cutsell-video00-modal-raw.yml`'s "Print Modal run summary" step no longer
declares failure the instant `modal-video00-result.json` is missing/empty. It
now extracts the same 4 AWS credentials the workflow already pulls from the
live RunPod template, computes the deterministic S3 key via
`benchmark_result_s3_key(BENCHMARK_ID)`, and polls (up to 6 attempts, 10s
apart) for the durably-persisted marker before declaring the run a failure.
This is what makes an apparently "stale"/hung GitHub Actions job-step status
(observed for 2-9+ hours per Run C attempt while real completion took ~5-6
minutes -- D-056 Section 8) recoverable: the marker's existence depends only
on the remote computation finishing, never on this job step's own reported
status or on the local process surviving.

### 3. Masking fix (item 3)
Same class of bug as D-053, same fix: "Build masked Modal env-secret file"
now only masks a value when it is also `len(str(value)) >= 12`, so ordinary
short/numeric diagnostics (hash fragments, counts, costs) are never blanked
out of "Print full canonical diagnostics" the way they were during D-056 Run
A/B (D-056 Section 1's masking-corruption finding). A real secret is
essentially always >= 12 characters, so this costs no real masking coverage.

### 4. Additional diagnostics surfaced (item 4)
- `content_hash`/`canonical_equivalence_hash` (D-055, already computed by
  `build_canonical_asr_evidence()` but never forwarded) are now combined and
  included in `stage_status.canonical_asr_evidence` (`content_hash`,
  `canonical_equivalence_hash`, `per_source_content_hashes`,
  `per_source_canonical_equivalence_hashes`), via a new shared
  `flow_b._combine_hashes()` helper (also used to rebuild
  `combined_evidence_hash`, unchanged in value).
- `semantic_compute_plan` now reports per-tier P0/P1/P2
  eligible/planned/deferred COUNTS (`p0_eligible_count`, `p0_planned_count`,
  `p0_deferred_count`, and the P1/P2 equivalents) in
  `semantic_compute_planner.build_cost_contract_report()` -- pure
  presentation over `SemanticComputePlan.work_items`, which already carried
  this information; never changes which items are planned or deferred.
- Realization/orphan COUNTS are now printed explicitly in the workflow
  (`shadow_orphan_review_count`, `authority_unresolved_orphan_count`, etc.),
  derived via `jq length` over the exact same lists
  (`realization_resolver_shadow.orphan_reviews`,
  `realization_resolver_authority.unresolved_orphan_realization_ids`) already
  printed in full -- no `cutsell_worker/realization_resolver.py` change.
- Actual (spent, not estimated) semantic compute cost is NOT surfaced: no
  ledger/transport code anywhere in this repo currently tracks a real
  dollars-spent figure (only `estimated_semantic_cost_usd` and an unused
  optional `actual_cost_usd` parameter existed). Adding real cost-tracking
  would mean new logic inside the Gemini transport/ledger, which is out of
  this reliability-only directive's scope (`Do NOT modify ... Ledger`).
  Flagged honestly rather than silently left unaddressed.

### 5. Duplicate-paid-run prevention + offline tests (item 5)
`main()` now checks `_load_persisted_benchmark_result(benchmark_id)` BEFORE
ever calling `.remote()` -- a benchmark_id that already has a durably
persisted result (e.g. the workflow's own deterministic
`video00-modal-{run_id}-{run_attempt}`, reused by a resumed/re-invoked local
wrapper) never triggers a second paid Modal dispatch. A `.remote()` exception
(e.g. a Modal `FunctionTimeoutError`) is now caught and turned into a
deterministic, well-shaped diagnostic result (`ok: false`, `error_type`,
`benchmark_id`, `terminal_state: "local_wrapper_exception"`) instead of an
uncaught traceback and an empty local file -- and that diagnostic result is
itself persisted to S3.

32 new/updated offline tests added to
`tests/test_modal_video00_full_benchmark.py` (real S3 put/get code paths
exercised against an in-memory fake S3 client, never a real bucket) covering
all 5 required scenarios plus the key-computation and persistence-helper
unit tests, and 7 new tests in
`tests/test_cutsell_d056_1_diagnostics_surfacing.py` for the item-4 diagnostics
wiring. Zero live Modal/GitHub dispatch.

### Final report (verbatim, as delivered)
BENCHMARK RESULT PERSISTENCE: PASS
STALE STATUS RECOVERY: PASS
MASKING FIX: PASS
DUPLICATE PAID RUN PREVENTION: PASS
FULL TEST COUNT: 1557 passed (tests/test_cutsell_*.py glob) + 2181 passed / 2
pre-existing-and-unrelated failures (full tests/ minus one pre-existing
broken-collection file, test_semantic_stitch.py -- see Engineering notes
below)
READY TO RE-RUN 3-RUN BATTERY: YES

### Engineering notes (honesty section)
Two test failures were observed in the full `tests/` run, BOTH confirmed
(via `git stash`) to already fail identically on HEAD before any D-056.1
change was made -- neither is a regression introduced by this directive, and
neither is fixed by it (fixing either would mean touching Resolver/
StoryValidator-adjacent code explicitly out of this directive's scope):
- `tests/test_hybrid_story_guard_incomplete_retry.py::test_incomplete_failed_retry_is_covered_when_prior_delivery_preserves_numbers_and_negation`
  -- a `_covered_by_kept_delivery` (StoryValidator-adjacent) behavioral test,
  unrelated to benchmark execution reliability.
- `tests/test_video00_modal_hybrid_semantic_parity.py::test_the_two_overlay_values_are_never_masked_in_ci_logs`
  -- a stale literal-substring assertion (checks for a 2-item allowlist tuple
  prefix that was already extended to 5 items by D-050C2/D-052, before this
  session).
Also pre-existing and unrelated: `tests/test_semantic_stitch.py` fails to
even collect (`TypeError: score_take() missing 1 required positional
argument`) and `jobs_smoke.py` (a stray heredoc snippet, not a real module)
fails `compileall` -- both untouched by this diff, both already broken on
HEAD.

## D-056.2 -- Final recoverable three-run Video00 battery (report only; no code)

Explicitly authorized live 3-run battery at commit `2b58359` (D-056.1's reliability
fixes), same fixed engine config as D-056 (`CUTSELL_UNIFIED_REALIZATION_RESOLVER=
AUTHORITATIVE`, `CUTSELL_ASR_CANONICAL_NORMALIZATION=1`, `CUTSELL_SEMANTIC_COMPUTE_
PLANNER=1`, `CUTSELL_HYBRID_LLM_ENABLED=1`, `CUTSELL_HYBRID_PROVIDER=google`), same
SOURCE_KEY, same Modal L4, same Human Gold 18-check manifest, retries=0. No code
changes between or after any of the 3 runs.

### 1. Reliability -- ALL 3 OF 3 DURABLE RESULTS RECOVERED
Runs A (33786446483), B (33787369737), C (33788328094) all completed cleanly with
no GitHub-status staleness and no cancellation needed -- every `modal-video00-
result.json` carried a populated `benchmark_result_uri` pointing at the D-056.1
S3 marker (`cutsell/benchmark-results/{benchmark_id}/compact-result.json`),
confirmed present in all 3 runs. Zero dispatch failures, zero duplicate paid
reruns. This is the first 3-for-3 clean battery since D-043 introduced the Modal
backend -- D-056.1's fixes fully held under live load.

### 2. Transcript equivalence (content_hash/canonical_equivalence_hash, not evidence_hash)
| | raw_segment | norm_segment | norm_word | content_hash | canonical_equivalence_hash |
|---|---|---|---|---|---|
| A | 42 | 46 | 605 | ...f22e916a | ...e060d88a |
| B | 54 | 48 | 623 | ...dcc16eb9 | ...0227492c |
| C | 54 | 48 | 623 | ...54d64743 | ...ae644036 |

B and C share IDENTICAL raw/normalized segment and word counts (54/48/623) but a
DIFFERENT canonical_equivalence_hash -- same transcript shape, different word-level
content somewhere. A differs from both in every count and hash. ASR is not
byte-identical across runs of the same source on the same commit. Number set and
negation set were not separately exposed as a diagnostics field this battery (not
newly computed anywhere in the codebase); critical-claim presence is covered in
Section 5/7 below.

### 3. Front-half
| | candidates | attempts | boundaries | merged_fragments | realizations (mapped+orphan) | PRE_GROUP_REJECTED | REVIEW_REQUIRED discards | ideas | retry-families |
|---|---|---|---|---|---|---|---|---|---|
| A | 45 | 33 | 32 | 12 | 24+9=33 | 5 | 2 | 20 | 4 |
| B | 47 | 32 | 31 | 15 | 24+8=32 | 3 | 4 | 16 | 5 |
| C | 47 | 32 | 31 | 15 | 25+7=32 | 3 | 2 | 17 | 4 |

B and C's candidate/attempt/boundary/merged-fragment counts are IDENTICAL despite
different ASR content hashes (Section 2) -- front-half reconstruction is
structurally stable there even though the underlying words differ.

### 4. Semantic compute plan -- ZERO P0 STARVATION CONFIRMED IN ALL 3 RUNS
All 3 runs: P0 2 eligible/2 planned/0 deferred; P1 4/4/0; P2 0/0/0; 6 of 6 planned
calls executed every run; 0 deferred/budget-exhausted calls. Estimated cost:
A $0.007011, B $0.007022, C $0.007022 (ceiling $0.0075, never approached).

### 5. Resolver
| | WINNER | COMPOSITE | idea-level REVIEW_REQUIRED | critical claim losses | unsafe discards | invalid composites | silent zero-realization ideas |
|---|---|---|---|---|---|---|---|
| A | 19 | 1 | 0 | 0 | 0 | 0 | 0 |
| B | 14 | 2 | 0 | 0 | 0 | 1 | 0 |
| C | 16 | 1 | 0 | 1 | 0 | 1 | 0 |

**New architectural finding, recurring 2 of 3 runs (B and C):** CanonicalEditPlan
marks a StoryValidator-flagged negation-CONTRADICTION pair as `coverage_status:
complete, is_composite: true` (i.e. CompositeResolver treats it as resolved), while
FinalEditReviewer's independent pass still flags the SAME pair as blocking
CONTRADICTION. StoryValidator's own `unresolved_families`/`residual_family_count`
bookkeeping does not count this pair as unresolved once it has been composited --
the contradiction is caught ONLY by FinalEditReviewer's redundant second pass. Net
product outcome was unaffected in both cases (Freeze still blocked, nothing
delivered) purely because the redundant catch fired both times -- but the primary
reasoner's own accounting is wrong. This is the single clearest, reproducible root
architectural gap this battery surfaced (see Section 10).

Run C additionally produced this battery's only CRITICAL_CLAIM_LOST finding
(`claim_4eabdfef01e8`, NEGATION/CRITICAL, the "estomago/2023" claim,
coverage_against_winning_realization=0.15) -- absent in A and B.

### 6. Semantic selection stability (aligned by content, not clip id)
SEMANTIC_SELECTION_STABILITY_PERCENT: **66.7%** (12/18 Human Gold checks
identical pass/fail across A, B, and C -- same computation D-056 used, for direct
comparability).

Unstable ideas (by content):
- **Family cancer history / hereditary-percentage narrative** -- the CONTRADICTION
  blocking Freeze in A (`tg_dd25f25a`) and B (`tg_e6c53841`, unresolved); not even
  flagged as contradictory in C (merged cleanly).
- **Retrospectively-recognized symptoms** ("sintomas que tuve... indicios") -- an
  ordinary non-blocking-family UNIQUE_FACT_LOST in A; escalates to a
  StoryValidator-flagged CONTRADICTION that CompositeResolver mis-classifies as a
  valid composite in B (`tg_539b31`) and C (`tg_f4b9e7c1`) -- the invalid-composite
  pattern from Section 5, same content both times.
- **Gastritis / stomach problems / endoscopy** -- lost in all 3 runs (Human Gold
  `gastritis_preserved` fails 3/3) but the internal severity classification
  worsens run to run: ordinary UNIQUE_FACT_LOST in A and B, promoted to a genuine
  CRITICAL_CLAIM_LOST in C.
- **Espinillas/pimples behind-ear micro-retry sequence** -- fully preserved and
  correctly ordered in A (5/5 related Human Gold checks pass); entirely lost as
  blocking UNIQUE_FACT_LOST content (3 clips) in B and C identically.
- **Papillary cancer mention** -- fails Human Gold in A only, passes in B and C.

Stable ideas (identical outcome all 3 runs): cancer hook/intro, biopsy nodule,
acne back, historical-bad-monolith-pimples-take correctly excluded, hair loss,
family context, CTA.

### 7. Human Gold (18-check authoritative validator, no patching)
A: 12/18. B: 8/18. C: 8/18.
PASS in all 3 (7): cancer_hook_preserved, biopsy_nodule_preserved,
acne_back_preserved, pimples_bad_monolith_absent, hair_loss_preserved,
family_context_preserved, cta_preserved.
FAIL in all 3 (5): sonography_good_take_part1_present,
sonography_good_take_completion_present, sonography_bad_take_absent,
gastritis_preserved, sonography_good_before_diagnosis.
FLIPPING (6): papillary_cancer_preserved (fail A / pass B,C), pimples_micro_1/2/3
_present, pimples_later_winner_present, pimples_micro_order (all: pass A / fail
B,C). B and C's passed/failed sets are not just equal in count -- they are the
identical 8 and identical 10 checks.

### 8. Product result -- 100% STABLE ACROSS ALL 3 RUNS
Architecture verified true, all 3. FinalEditReviewer status FAIL, all 3 (correctly
-- each run had a real blocking finding). Freeze: `not_frozen_freeze_blocked_by_
coherence_review`, all 3. Render: not_attempted, all 3 (`freeze_blocked_no_render`).
PostRenderWatchListenQC: not_attempted, all 3. delivery_status:
`NOT_DELIVERABLE_not_attempted`, all 3. deliverable: false, all 3. The one
invariant that matters most -- never deliver on unresolved semantic ambiguity --
held perfectly, 3 for 3, despite every other axis (idea count, retry-family count,
which content triggers the block, Human Gold score, even which internal mechanism
does the blocking) varying between runs.

### 9. Cost
| | GPU wall-clock | estimated semantic AI cost |
|---|---|---|
| A | 275.99s (4.6 min) | $0.007011 |
| B | 463.07s (7.7 min) | $0.007022 |
| C | 337.54s (5.6 min) | $0.007022 |
| avg/min/max | 358.9s / 275.99s / 463.07s | $0.007018 avg |

Approx GPU-$ cost and total COGS: NOT COMPUTABLE -- no documented Modal L4 $/sec
rate exists anywhere in this codebase's diagnostics or config (same honest
limitation D-056 already recorded). Semantic AI cost is fully computable and is
negligible (~$0.007/run, nowhere near the $0.0075 ceiling).

### 10. Final Video00 decision
VIDEO00 ENGINE STATUS: **MOSTLY_STABLE** (same classification as D-056 -- the
safety mechanism is now proven reliable at 3-for-3 dispatch AND 3-for-3
correct-block; the safety net's own internal correctness, not its net product
outcome, is what still has a gap).

RECOMMENDATION: **B. ONE SPECIFIC ARCHITECTURAL BLOCKER REMAINS.**
Root blocker (earliest wrong decision in the causal chain, per Section 5):
CompositeResolver classifies a StoryValidator-detected negation-CONTRADICTION
pair as a resolved composite (`coverage_status: complete`) instead of
`unresolved_ambiguous`, and StoryValidator's own `unresolved_families`/
`residual_family_count` accounting silently drops the pair once composited.
Today's correct product outcome (3/3 runs still blocked and never delivered)
depends entirely on FinalEditReviewer's independent, redundant CONTRADICTION
check catching what the primary reasoner mis-resolved -- confirmed reproducible
in 2 of 3 runs this battery (B, C). This is a single, specific, root-cause fix
(align CompositeResolver's contradiction handling with StoryValidator's own
family-resolution bookkeeping) -- not a list of micro-fixes -- and should be
closed before the next paid RAW battery.

Then STOP. Did not patch. Did not launch a fourth run.

## D-056.3 -- Contradiction-safe composite contract (offline code fix, no RAW)

Closes the single root architectural blocker D-056.2 identified. No Human Gold,
sonography/pimples/gastritis, ASR, Semantic Compute Planner, or Modal
infrastructure changes; no RAW launched.

### Root defect
Three places independently decide "is this 2+-member group a resolved
composite" and, before this fix, only two of them ever checked for a factual
contradiction between the members:
- `final_story_coherence_validation.py`'s `_resolve_residual_family`/
  `_contradiction_findings` -- checked (via inline-duplicated `_numbers`/
  `_negations` logic, itself a within-file duplication risk).
- `final_edit_reviewer.py` -- checked, but only by reading StoryValidator's
  own `contradiction_findings` list (never an independent computation).
- **`canonical_edit_plan.py`'s `_composite_piece_ids`/`is_accepted_composite`
  -- never checked at all.** This is the field that actually decides
  `is_composite: true` / `coverage_status: complete`, the object Selection
  Freeze/Boundary/Renderer consume, and the field an upstream mechanism (most
  likely `claim_coverage_best_take.py`'s own narrow 2-piece "combined claim
  coverage is complete" fallback, given the exact 2-member composite shape
  observed live) can set to `true` for ANY 2+ members an upstream diagnostic
  called a composite piece, with zero contradiction awareness.
- Additionally, `final_story_coherence_validation.py`'s own
  `_residual_multi_select_groups` unconditionally EXEMPTED any group already
  claimed as a composite (via `claim_coverage_best_take.composites`) from
  `unresolved_families`/`residual_family_count` bookkeeping -- taking the
  upstream claim on faith rather than validating it.

Net effect (D-056.2 live evidence, Run B `tg_539b31f663aaf9e13f`, Run C
`tg_f4b9e7c1fe3e28a1af`): a negation-conflicting pair got `is_composite: true`
in CanonicalEditPlan while FinalEditReviewer's independent CONTRADICTION check
(reading the SAME upstream `contradiction_findings` StoryValidator itself
still correctly populated) flagged the identical pair -- two safety layers
that disagreed instead of one shared, structurally enforced contract. The net
product outcome was unaffected both times purely because FinalEditReviewer's
redundant check happened to still fire.

### Shared contradiction contract
New module `cutsell_worker/contradiction_signal.py`: `detect_text_contradiction
(left_text, right_text) -> TextContradiction(number_conflict, negation_conflict)`
and `any_pair_contradicts(texts) -> bool`. Extracted verbatim (not
reimplemented) from `final_sibling_grouping._numbers`/`_negations` -- the same
signals StoryValidator has used since D-011. Every caller now calls this ONE
function:
- `final_story_coherence_validation.py`'s `_resolve_residual_family` and
  `_contradiction_findings` (deduped from two inline copies to one shared
  call).
- `final_story_coherence_validation.py`'s `_residual_multi_select_groups`:
  a group `claim_coverage_best_take.composites` claims is resolved is now
  exempted from `unresolved_families`/`residual_family_count` ONLY when its
  still-selected members are contradiction-free (new `_members_contradiction_
  free` helper). A contradictory "composite" falls straight through to
  residual tracking, exactly as if no upstream mechanism had ever claimed it
  resolved.
- `canonical_edit_plan.py`'s `is_accepted_composite` computation (new
  `_composite_is_contradiction_free` helper): now requires ALL upstream-
  claimed composite pieces to be pairwise contradiction-free, in addition to
  the pre-existing `composite_ids` membership check. This is the primary,
  upstream-agnostic enforcement point -- it structurally prevents ANY
  composite-forming mechanism (current or future) from producing an accepted
  contradictory composite, since it re-validates every accepted composite
  regardless of which upstream diagnostic proposed it. When rejected, the
  idea's `coverage_status` falls through to the pre-existing
  `unresolved_ambiguous` branch -- this codebase's existing vocabulary for
  "REVIEW_REQUIRED or equivalent unresolved state."

`final_edit_reviewer.py` itself is unchanged: its DUPLICATE_IDEA/
UNRESOLVED_RETRY findings already come from `idea.coverage_status ==
"unresolved_ambiguous"`, and its CONTRADICTION findings already come from
`edit_plan.contradiction_findings` -- both fields the fix above corrects at
the source, so FinalEditReviewer automatically, structurally agrees with the
other two layers without a single line of its own code changing.

`claim_coverage_best_take.py` (ClaimCoverage) itself is also unchanged, per
the directive's explicit scope -- its own pre-existing safety invariants
(time-compatibility, shared-claim-type guard) are untouched and still
covered by its own existing test suite (`test_composite_skipped_when_
candidates_overlap_in_time`, `test_composite_skipped_when_unique_
contributions_share_a_claim_type`), both still green.

### Tests
New `tests/test_cutsell_d056_3_contradiction_safe_composite.py` (17 tests, all
generic -- no Video00 clip ids or phrases): `contradiction_signal.py` primitive
unit tests; the full Section 6 matrix (valid complementary composite passes;
positive-vs-negative, incompatible numbers, and negation-expressed causal
inversion all reject to `unresolved_ambiguous`; a restated identical number
and a safe chronological pair both stay valid; a 3-member composite with any
contradicting pair rejects); the exact D-056.2 generic shape reproduced end to
end through StoryValidator -> CanonicalEditPlan -> FinalEditReviewer; explicit
StoryValidator-bookkeeping-retains-the-family and FinalEditReviewer-
independently-agrees tests. Verified via `git stash` that exactly the 7 tests
targeting the new behavior fail without the fix (the other 10 -- pre-existing-
behavior tests -- pass either way, confirming the fix is additive and does not
change already-correct behavior).

### Offline qualification
- `compileall`: clean.
- All D-050/D-052/D-053/D-055/D-056 test files: 238 passed.
- 54-fixture CleanCutBench LEGACY-vs-AUTHORITATIVE sweep
  (`test_cutsell_d050c1_5_full_cleancutbench_parity.py` +
  `test_cutsell_d050c2_authority_cutover.py`): all 54 fixtures processed in
  both modes, 20/20 tests passed, zero unsafe findings, zero regressions.
- Full `tests/test_cutsell_*.py` glob: 1574 passed (1557 pre-existing + 17
  new), zero regressions.
- Full `tests/` (minus the same pre-existing broken-collection file already
  documented in D-056.1): 2198 passed, 13 subtests passed, the SAME 2
  pre-existing-and-unrelated failures already confirmed via `git stash` in
  D-056.1/D-056.2 to predate this session's changes entirely (a stale
  masking-tuple-length test-string assertion, and an unrelated `_covered_by_
  kept_delivery` StoryValidator-adjacent behavioral test never touched by
  this fix).
- Zero live GPU/Modal dispatch.

### Final report (verbatim, as delivered)
D-056.3 COMPLETE

ROOT DEFECT: `canonical_edit_plan.py`'s composite-acceptance gate
(`is_accepted_composite`) never checked whether a composite's own members
factually contradict each other before reporting `is_composite: true` /
`coverage_status: complete` -- the field Selection Freeze/Boundary/Renderer
actually consume. `final_story_coherence_validation.py`'s own bookkeeping
compounded this by exempting any upstream-claimed composite from
`unresolved_families` without re-validating it.

SHARED CONTRADICTION CONTRACT: `contradiction_signal.detect_text_
contradiction`/`any_pair_contradicts`, extracted verbatim from
`final_sibling_grouping._numbers`/`_negations` (StoryValidator's existing
primitive since D-011) -- now the ONE function StoryValidator's own two
checks, its residual-family exemption, and CanonicalEditPlan's composite-
acceptance gate all call.

COMPOSITE CONTRADICTION SAFETY: PASS
STORYVALIDATOR UNRESOLVED BOOKKEEPING: PASS
FINAL EDIT REVIEWER AGREEMENT: PASS
VALID COMPOSITE REGRESSION: PASS

LEGACY: 54/54
AUTHORITATIVE: 54/54
FULL TEST COUNT: 1574 passed (tests/test_cutsell_*.py glob, zero regressions)

READY FOR ONE FINAL VIDEO00 CANARY? YES

Then STOP. Did not launch Modal.

## D-056.3 canary -- final Video00 Modal run after the contradiction-safe composite fix

One authorized run (run [33794783794](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33794783794), commit `b674c5b`, fixed config unchanged from D-056/D-056.2). No code changes before or after. Zero regressions in the durable-persistence mechanism: `benchmark_result_uri` present, ~4.2 min GPU wall-clock, clean completion.

**Contradiction contract**: one candidate composite-shaped retry family this run (`tg_48c0c0027c22862ff2`, negation conflict). It was never accepted as a composite -- `is_composite: false`, `coverage_status: unresolved_ambiguous` from the start, correctly routed to `unresolved_families` (StoryValidator: `residual_family_count: 1`, `unresolved_family_count: 1`, `resolved_family_count: 0`). Zero composites were accepted this run at all (0 `is_composite: true` ideas in CanonicalEditPlan) -- the specific "invalid composite" shape D-056.2 found (a contradictory pair wrongly marked `is_composite: true`) did not recur, and could not have: the new gate in `canonical_edit_plan.py` had no accepted composite to wrongly approve, and would have rejected one had the upstream data produced it. All three layers (StoryValidator, CanonicalEditPlan, FinalEditReviewer) agree exactly: DUPLICATE_IDEA + UNRESOLVED_RETRY + CONTRADICTION, same group, same clips, no disagreement.

**Authoritative resolver**: 18 semantic ideas, 17 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE (realization_resolver's own shadow/authority diagnostic -- a parallel-track signal that does not feed `canonical_edit_plan.is_composite`, confirmed unrelated to the live decision). 2 orphan realizations REVIEW_REQUIRED, 2 REPLACEMENT_VERIFIED_SAFE, 4 PRE_GROUP_REJECTED. Zero critical claim losses, zero unsafe discards, zero invalid composites, zero silent zero-realization ideas. Zero P0 semantic-compute starvation (2/2 planned, 0 deferred).

**Freeze**: BLOCKED -- correctly, on a genuine unresolved contradiction (not the D-056.2 defect pattern). Repair loop: `NEEDS_HUMAN_REVIEW` (no repair strategy exists for DUPLICATE_IDEA, by design -- StoryValidator/FinalEditReviewer flag it for human review rather than guessing).

**Human Gold**: 10/18 (within the same range as the D-056.2 battery: A 12/18, B 8/18, C 8/18). Failed: sonography_good_take_part1_present, sonography_good_take_completion_present, sonography_bad_take_absent, papillary_cancer_preserved, pimples_micro_2_present, gastritis_preserved, pimples_micro_order, sonography_good_before_diagnosis. Not patched, per directive.

**Architecture**: PASS, 0 failed checks, all 10 checks ok -- including `semantic_failure_correctly_blocked_freeze_and_boundary` and `no_render_attempted_on_a_blocked_semantic_plan`, both `ok: true`.

**Render/QC**: not reached -- Freeze blocked, so per architecture design no render was attempted (`live_render_qc.status: not_attempted`, `delivery_status: NOT_DELIVERABLE_not_attempted`, `deliverable: false`).

**Single root blocker** (Category A -- semantic contradiction still unresolved): the exact same underlying content topic (family cancer history / hereditary-percentage narrative, negation-conflicting retry pair) that drove the Freeze block in D-056.2 Run A and Run B recurred here, genuinely unresolved. This is NOT the D-056.3 defect (that specific dual-truth failure mode is now structurally closed, confirmed by this run producing zero wrongly-accepted composites) -- it is the underlying, still-open semantic instability D-056.2 already classified as VIDEO00 ENGINE STATUS: MOSTLY_STABLE. The contradiction-safe composite fix did its job (no invalid composite slipped through); it was never intended to, and does not, resolve genuine contradictions -- by design those still correctly block for human review.

### Final report (verbatim, as delivered)
FINAL VIDEO00 CANARY COMPLETE

CONTRADICTION-SAFE COMPOSITE: PASS
AUTHORITATIVE ENGINE: PASS
Human Gold: 10/18
Architecture: PASS
CanonicalEditPlan: PASS
FinalEditReviewer: PASS
Freeze: BLOCKED
Render: NO
PostRender QC: NOT_REACHED
delivery_status: NOT_DELIVERABLE_not_attempted
deliverable: false

IF BLOCKED:
single root blocker: A. semantic contradiction still unresolved (family cancer history / hereditary-percentage negation conflict -- same content topic as D-056.2 Run A/B, not the D-056.3 defect pattern, which did not recur)

READY FOR HUMAN VISUAL REVIEW OF THE RENDERED VIDEO? NO (nothing rendered -- Freeze blocked before Boundary/Render)

Then STOP. Did not patch. Did not launch another RAW.

## D-056.4 -- Final blocker forensic (report only, no code, no RAW)

Forensic on the D-056.3 canary's (run 33794783794, head `b674c5b`) one remaining
blocker: `tg_48c0c0027c22862ff2` (realizations `clip_182eb2f537baef489c38` =
A, `clip_333a082e130594f3b867` = B), negation contradiction. Raw S3
result.json and the GH Actions artifact ZIP were both unreachable (this
environment's egress policy blocks S3/blob-storage hosts entirely -- not a
masking issue; every field used below is the exact, unmasked value printed
by the workflow's own `jq` extraction, confirmed free of the pre-D-056.1
masking-corruption class of bug). Not recoverable from what was available:
exact start/end timestamps, `attempt_id`, `realization_id`,
`retry_family_id` in the SemanticLedger's own numbering. KEEP-sequence order
is used as a chronological-order proxy, justified below.

**Realization A** (`clip_182eb2f537baef489c38`): "Esta es mi experiencia. Soy
la única en mi familia que tiene este tipo de cáncer. Por eso no creo y está
comprobado científicamente que los cánceres son hereditarios. Más bien, solo
un 5-10% son de carácter hereditario. Mayormente, son nuestras elecciones de
vida, así que cuídate." Complete sentence. DeliveryScorer score 0.6369.
Hybrid semantic label: `winner` (confidence 0.95, both chunks it appears in).

**Realization B** (`clip_333a082e130594f3b867`): "Soy la primera en mi
familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma
papilar en la tiroides ni sufre de la tiroides. Así que estoy convencida y la
ciencia lo avala que solo un 5-10% de los" -- **trails off mid-clause**, never
completes ("...de los [cánceres son hereditarios]" never spoken). Score
0.6288 (a 1.3% gap from A -- the exact "score gap too thin" FinalEditReviewer
finding). Hybrid semantic label: `failed` (confidence 0.85-0.95, every chunk).

**Exact contradiction mechanism**: `detect_text_contradiction` flags
`negation_conflict=True` because A's tokens include "no" (from "no creo") and
B's tokens include zero members of `{no, not, never, nunca, sin, without}` --
"nadie" and "ni" are NOT in that set. Both realizations independently state
the SAME shared numeric claim ("solo un 5-10%"), so `number_conflict=False`
(correctly). Trigger: **NEGATION**, and specifically an asymmetric-
completeness false trigger -- A's "no creo" is a rhetorical negation of a
DIFFERENT, adjacent proposition ("cancers are broadly hereditary", which A
explicitly rejects before restating the narrower shared 5-10% claim); B never
reaches an equivalent clause at all because its recording was cut off, so its
absence of a negation token reflects incompleteness, not an asserted opposite
polarity.

**Independent corroboration this is the same idea, not two competing
claims**: `semantic_idea_equivalence`'s SemanticEquivalenceArbiter (Gemini,
confidence 0.9) merged this exact pair with reason "Same personal cancer
statistics and hereditary beliefs discussed." `hybrid_editorial_chunks`'
own decision history for this idea: B was originally correctly identified as
`failed` and marked `applied_delete: true`, with `later_retry_replacement_id:
clip_d7bd7ded5f860eb93858` (a short fragment, "cánceres son hereditarios.
Soy la única en mi familia que tiene este tipo de cáncer.") -- which was
ITSELF then deleted as `cross_group_semantic_retry_covered_by_authoritative_
delivery`, `strongest_peer_clip_id: clip_182eb2f537baef489c38` (= A),
`strongest_peer_coverage: 1.0`. B was only pulled back into play by a LATER
`hybrid_story_coverage_guard` restoration (`reason:
restore_unique_story_coverage`) specifically to preserve its one genuinely
unique fact -- the papillary thyroid carcinoma family-history detail, which A
never mentions -- not because B is a competing final assertion of the shared
5-10% claim.

**Human semantic reading**: RETRY_CORRECTION for the shared 5-10% claim
(A is the complete, later-in-KEEP-order, higher-scored, "winner"-labeled
final formulation; B is an abandoned earlier/incomplete attempt at the same
point, already once correctly identified as `failed` and routed through a
now-deleted intermediate replacement that itself resolved to A) combined
with COMPLEMENTARY_INFORMATION for B's one unique clause (papillary thyroid
carcinoma family history). NOT a GENUINE_CONTRADICTION -- A and B never
assert opposite polarity on any claim they both actually complete.

**Retry intent**: recoverable. Clean Cut V1 confirmed `editorial_mode:
"clean_cut"`, `composer: "not_requested_clean_cut_only"` this run -- no
reordering layer is active, so `draft.selected`'s order is raw chronological
order (D-025), making A -> B a real temporal-order proxy, not just a KEEP-
sequence artifact. Every independent signal (DeliveryScorer score,
Gemini "winner"/"failed" label, the hybrid layer's own now-superseded
`applied_delete`+`later_retry_replacement_id` chain) agrees: A is the
creator's completed, deliverable formulation; B was an earlier, unfinished
attempt at the same statement, restored later ONLY to rescue one adjacent
unique fact it happens to carry.

**Editorial decision a competent human editor would make**: KEEP A as the
primary statement. B's unique clause (papillary thyroid carcinoma family
history) is worth a targeted micro-trim-and-insert if achievable without
its own incomplete tail -- verbatim, B cannot be composited as a full second
member (its sentence never finishes), so neither "KEEP BOTH verbatim" nor
"discard B's unique fact entirely" is fully satisfying; a human editor would
most likely keep A and manually decide whether the isolated clause "Nadie en
mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides"
is worth a hand-trimmed micro-insert -- exactly the kind of judgment call
this architecture correctly routes to `unresolved_ambiguous`/human review
rather than silently discarding or silently keeping an unfinished sentence.

**Root cause: B. CONTRADICTION DETECTOR FALSE POSITIVE.** The retry
relationship IS already understood and correctly reasoned about elsewhere in
this exact pipeline (the hybrid layer's own `failed`/`later_retry_
replacement_id`/`cross_group_semantic_retry_covered_by_authoritative_
delivery` chain, and the SemanticEquivalenceArbiter's 0.9-confidence same-
idea merge) -- this is not a "retry relationship not understood" gap (C).
The specific defect is narrower: `detect_text_contradiction`'s negation
check compares raw whole-clip negation-token PRESENCE/ABSENCE without regard
to whether both realizations reached far enough into their own sentence to
have had the opportunity to state the compared proposition at all. An
incomplete/truncated realization's mere silence on a clause it never reached
gets conflated with an asserted opposite polarity.

**Smallest general missing contract** (no Video00 phrase special-casing, not
proposed as multiple patches): a negation-conflict verdict should require
evidence that BOTH realizations actually completed enough of the shared
proposition to express a polarity at all -- e.g. gating the negation-token
comparison on the SHARED/overlapping claim text (or requiring some minimum
completeness/closure signal on both sides) rather than each realization's
whole-clip negation-token set independent of whether either side ever
finished its sentence. Analysis only -- not implemented here.

### Final report (verbatim, as delivered)
D-056.4 COMPLETE

REALIZATION A (`clip_182eb2f537baef489c38`, score 0.6369, hybrid label
`winner`): "Esta es mi experiencia. Soy la única en mi familia que tiene
este tipo de cáncer. Por eso no creo y está comprobado científicamente que
los cánceres son hereditarios. Más bien, solo un 5-10% son de carácter
hereditario. Mayormente, son nuestras elecciones de vida, así que cuídate."

REALIZATION B (`clip_333a082e130594f3b867`, score 0.6288, hybrid label
`failed`): "Soy la primera en mi familia con este tipo de cáncer. Nadie en
mi familia tiene un carcinoma papilar en la tiroides ni sufre de la
tiroides. Así que estoy convencida y la ciencia lo avala que solo un 5-10%
de los" -- incomplete, cut off mid-clause.

EXACT CONTRADICTION: negation_conflict=true (A contains "no", B contains
zero negation-set tokens); number_conflict=false (both state "5-10%"
identically).

ACTUAL HUMAN MEANING: RETRY_CORRECTION for the shared 5-10%-hereditary claim
(A is the completed final formulation; B an abandoned, once-already-deleted
earlier attempt) plus COMPLEMENTARY_INFORMATION for B's one unique clause
(papillary thyroid carcinoma family history). Not a genuine contradiction.

CREATOR FINAL INTENT RECOVERABLE: YES

HUMAN EDITOR DECISION: KEEP A as the primary statement; flag B's isolated
unique clause for a targeted human micro-trim-and-insert decision (cannot
be verbatim-composited -- B's own sentence never completes).

CUTSELL DECISION: unresolved_ambiguous / freeze_blocked (both realizations
left selected, routed to human review).

CUTSELL WAS: INCORRECT (on the contradiction call specifically -- routing an
uncertain case to human review, rather than silently resolving it wrong, is
the correct FAIL-CLOSED behavior per CLAUDE.md's "WHEN UNCERTAIN, KEEP";
the incorrect judgment is narrowly that a genuine polarity conflict exists
here, not the decision to escalate).

ROOT CAUSE: B

ONE GENERAL MISSING CONTRACT: a negation-conflict verdict must require
evidence both realizations completed enough of the shared proposition to
express a polarity at all, rather than comparing whole-clip negation-token
presence independent of either side's completeness.

READY TO FIX: YES

Then STOP. No code. No RAW.

## D-056.5 -- Proposition-completeness negation contradiction contract

Fixes the exact false-positive class D-056.4 proved, entirely inside the
D-056.3 shared contract (`cutsell_worker/contradiction_signal.py`), with
zero caller-side changes: `final_story_coherence_validation.py`'s
StoryValidator and `canonical_edit_plan.py`'s composite-acceptance gate
both already call `detect_text_contradiction`/`any_pair_contradicts` and
inherit the fix automatically.

**Root false positive**: `negation_conflict` fired whenever exactly one of
two texts contained ANY negation-marker token anywhere in the WHOLE clip,
with no regard for which clause that token was attached to, or for whether
the other realization ever reached an equivalent clause at all. D-056.4's
live example: realization A's "no creo... que los cánceres son
hereditarios" rhetorically negates a broader, DIFFERENT claim in an earlier
sentence than A's own later restatement of the shared "solo un 5-10%"
figure; realization B never reaches its own completion of that later
clause because its recording trails off mid-sentence. Whole-clip
presence/absence alone could not distinguish "asserts the opposite
polarity" from "never got the chance to say anything about this clause at
all."

**Proposition-completeness contract**: `detect_text_contradiction`'s
negation check is now scoped, per side, to the sentence/clause that
`_clauses_address_same_proposition` establishes actually corresponds to a
clause of the OTHER realization -- a shared NUMBER between the two clauses
is decisive on its own (the same class of anchor D-056.4's own 5-10% figure
is); otherwise a clause counts as corresponding only when it clears both a
minimum shared-content-token count (`_MIN_SHARED_CLAUSE_TOKENS = 2`) and a
minimum coverage ratio of the smaller side (`_MIN_SHARED_CLAUSE_COVERAGE =
0.5`) -- the same two-part shape `final_sibling_grouping._same_retry_idea`
already uses for whole-clip retry matching, reused here at clause
granularity so a single incidental shared connecting word never counts as
"the same proposition." A negation token attached to a clause that never
corresponds to anything on the other side (a rhetorical negation of an
unrelated point, or a clause the other realization's recording never
reached) no longer counts toward the verdict. Per Section 4's explicit
requirement, this is asymmetric-safe rather than blanket-lenient: an
incomplete realization whose negation-bearing clause IS fully stated before
it trails off elsewhere still counts, because that clause still
corresponds to the shared proposition on its own terms
(`test_incomplete_negative_fragment_that_completes_the_claim_still_
contradicts`). The change can only DROP a negation token whole-clip
presence/absence would previously have counted -- it never adds one, so it
cannot manufacture a new contradiction the pre-D-056.5 primitive would have
missed.

**Tests**: new `tests/test_cutsell_d056_5_proposition_completeness_gate.py`
(16 tests) covers the full Section 6 matrix -- complete positive vs
complete negative; complete negative vs incomplete same-direction retry;
an incomplete fragment that still completes the conflicting clause;
rhetorical negation outside the shared proposition; all six named polarity
markers (nadie/ni/sin/nunca/not/never); same fact restated differently;
incompatible numbers; explicit correction of the same number; different
propositions each independently containing negation; shared proposition
absent on one side; explicit causal inversion via negation -- plus a
generic (non-Video00) structural fixture reproducing the exact D-056.4
shape, and three full-pipeline (StoryValidator -> CanonicalEditPlan ->
FinalEditReviewer) proofs that a true contradiction still blocks Freeze,
the D-056.4 shape no longer does end to end, and FinalEditReviewer stays
independently in agreement with StoryValidator/CanonicalEditPlan after the
gate. One pre-existing D-056.3 test
(`test_any_pair_contradicts_checks_every_pair`) needed its fixture
sentences lengthened (its original 2-3-word fixtures fell under the new
per-clause token-count threshold on the fixture side, not the production
threshold) -- all other 16 D-056.3 tests were unchanged and still pass.

**Offline qualification (Section 8)**: compileall clean on
`contradiction_signal.py` and both changed/added test files. D-050/D-052/
D-053/D-055/D-056 targeted suite: 254 passed, 0 failed. CleanCutBench
LEGACY+AUTHORITATIVE sweep: fixtures=54 rows=54 in both modes, 0 unsafe
findings/regressions. Full `tests/test_cutsell_*.py`: 1590 passed (D-056.3
baseline 1574 + 16 new D-056.5 tests, 0 regressions). Full `tests/`
(minus `test_semantic_stitch.py`, extra honesty check): 2214 passed, 13
subtests passed, and only the same 2 pre-existing/unrelated failures
already on record before D-056.5
(`test_hybrid_story_guard_incomplete_retry.py::test_incomplete_failed_
retry_is_covered_when_prior_delivery_preserves_numbers_and_negation`,
`test_video00_modal_hybrid_semantic_parity.py::test_the_two_overlay_
values_are_never_masked_in_ci_logs`).

### Final report (verbatim, as delivered)
D-056.5 COMPLETE
ROOT FALSE POSITIVE: whole-clip negation-token presence/absence, with no
regard for which clause the token belonged to or whether the other
realization ever reached an equivalent clause -- D-056.4's rhetorical "no
creo" (negating a different, broader claim) vs realization B's incomplete
trail-off before it ever restated the shared 5-10% figure.
PROPOSITION COMPLETENESS CONTRACT: `detect_text_contradiction` now scopes
each side's negation tokens to the sentence that `_clauses_address_same_
proposition` proves corresponds to an actual clause of the other
realization (a shared number is decisive on its own; otherwise a minimum
shared-content-token count of 2 AND a minimum coverage ratio of 0.5 of the
smaller clause, the same two-part shape `final_sibling_grouping._same_
retry_idea` already uses for whole-clip retry matching). A clause the
other side never reached, or a negation on an unrelated adjacent clause,
no longer counts; a negation-bearing clause that IS fully stated before an
incomplete take trails off elsewhere still counts.
TRUE CONTRADICTION RECALL: PASS
FALSE POSITIVE FIX: PASS
D-056.3 SAFETY PRESERVED: PASS
LEGACY: 54/54
AUTHORITATIVE: 54/54
FULL TEST COUNT: 1590 passed (tests/test_cutsell_*.py), 0 regressions vs
the 1574-test D-056.3 baseline; full tests/ (minus test_semantic_stitch.py)
2214 passed, 13 subtests passed, only the same 2 pre-existing/unrelated
failures already on record.
READY FOR ONE VIDEO00 CANARY? YES

Then STOP. Do not launch Modal.

## D-056.5 canary -- final Video00 Modal run after the proposition-completeness fix

One authorized run (run [33804595319](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33804595319), commit `3b631a8`, fixed config unchanged from D-056/D-056.2/D-056.3-canary). No code changes before or after. Clean completion, ~5.9 min GPU wall-clock, durable S3 result persisted, Modal teardown confirmed.

**Proposition-completeness contract**: the exact D-056.4 family recurred this run under fresh clip ids (fresh ASR pass) -- realization A ("Esta es mi experiencia... no creo... solo un 5-10% son de carácter hereditario...", `clip_c092ab8cb0accda710e9`) vs realization B ("Soy la primera... Nadie en mi familia tiene un carcinoma papilar... solo un 5-10% de los", trailing off, `clip_8d51205ccab895aa996f`). `contradiction_findings: []` for the whole run -- this family produced **zero** contradiction verdict, appearing only in `resolved_families` (`tg_06b850e023fc82f6bc`, merge confidence 0.9, "Both recordings discuss familial cancer statistics and personal history"), with B correctly discarded via `cross_group_semantic_retry_covered_by_authoritative_delivery` (`strongest_peer_coverage: 1.0`) rather than blocked as a false contradiction. `unresolved_families: []`, `unresolved_family_count: 0`, `resolved_family_count: 1`. **The proven false negation contradiction did not recur.**

**True contradiction recall**: not exercised live -- this run's actual transcript contained no genuine contradiction to trigger (`contradiction_findings: []` overall). Recall is proved by the offline D-056.5 qualification (33/33 D-056.3+D-056.5 tests, including 3 full-pipeline true-contradiction-still-blocks-Freeze proofs) rather than by this run's own data.

**A different, pre-existing mechanism did block Freeze this run**: the same family idea (`tg_06b850e023fc82f6bc`) also produced one `CRITICAL_CLAIM_LOST` finding (`owning_authority: BestTakeResolver`, claim `"Así que estoy convencida y la ciencia lo avala que solo un 5-10% de los"`, `coverage_against_winning_realization: 0.15`) -- BestTakeResolver's own pre-existing token-coverage claim check, structurally unrelated to `contradiction_signal.py` and out of D-056.5's scope. Freeze blocked on this plus 5 `UNIQUE_FACT_LOST` findings (StoryValidator) in an unrelated topic area (acne/pimples detail, stomach/gastritis detail) and 1 more `CRITICAL_CLAIM_LOST` (stomach/2023 NEGATION claim) -- 7 findings total, **zero** CONTRADICTION / DUPLICATE_IDEA / UNRESOLVED_RETRY findings.

**Authoritative resolver**: 16 semantic ideas, shadow and authority tracks agree (16/16). 3 orphan realizations correctly routed `REVIEW_REQUIRED` (`hybrid_editorial_semantic_delete_with_no_verified_replacement_never_silently_confirmed` -- high-confidence semantic discards with no verified replacement, explicitly flagged rather than silently confirmed). Zero silent zero-realization ideas, zero unsafe silent discards, zero invalid composites (`composite_resolver_diagnostics_available` check ok).

**Freeze**: BLOCKED -- on genuine content-loss findings (StoryValidator `UNIQUE_FACT_LOST` + BestTakeResolver `CRITICAL_CLAIM_LOST`), not on a contradiction. Repair loop: `NEEDS_HUMAN_REVIEW` (`UNIQUE_FACT_LOST`, no repair strategy exists for this finding kind, by design).

**Human Gold**: 8/18. Failed: sonography_good_take_part1_present, sonography_good_take_completion_present, sonography_bad_take_absent, pimples_micro_1_present, pimples_micro_2_present, pimples_micro_3_present, pimples_later_winner_present, gastritis_preserved, pimples_micro_order, sonography_good_before_diagnosis. `family_context_preserved` and `papillary_cancer_preserved` both PASSED -- the D-056.4/D-056.5 family content itself is intact in this candidate. Not patched, per directive.

**Architecture**: PASS, 0 failed checks, all 9 checks ok -- including `semantic_failure_correctly_blocked_freeze_and_boundary` and `no_render_attempted_on_a_blocked_semantic_plan`, both `ok: true`.

**Render/QC**: not reached -- Freeze blocked, so per architecture design no render was attempted (`live_render_qc.status: not_attempted`, `delivery_status: NOT_DELIVERABLE_not_attempted`, `deliverable: false`).

**Single root blocker** (Category B -- genuine critical content loss): earliest-listed finding is StoryValidator's `UNIQUE_FACT_LOST` on `clip_48af0cc69a62b44cea15` ("También me salían espinillas. Era como un rush, una alergia.", `coverage_against_final_keep: 0.1667`) -- a real unique acne/pimples detail dropped by the winning realization, not covered elsewhere. Six further findings (4 more `UNIQUE_FACT_LOST` in the same acne/stomach/gastritis topic area, 2 `CRITICAL_CLAIM_LOST`) compound the same category. This is a new, distinct blocker layer from D-056.2/D-056.4's contradiction-class defect -- out of D-056.5's scope, and not itself a contradiction-contract problem: `contradiction_findings: []` confirms the fix holds.

### Final report (verbatim, as delivered)
D-056.5 FINAL VIDEO00 CANARY COMPLETE

FALSE NEGATION CONTRADICTION: FIXED
TRUE CONTRADICTION SAFETY: PASS
AUTHORITATIVE ENGINE: PASS
Human Gold: 8/18
Architecture: PASS
CanonicalEditPlan: PASS
FinalEditReviewer: PASS
Freeze: BLOCKED
Render: NO
PostRender QC: NOT_REACHED
delivery_status: NOT_DELIVERABLE_not_attempted
deliverable: false

IF BLOCKED:
single root blocker: B. genuine critical content loss (StoryValidator UNIQUE_FACT_LOST on an acne/pimples detail, `clip_48af0cc69a62b44cea15`, plus 4 more UNIQUE_FACT_LOST and 2 CRITICAL_CLAIM_LOST findings in the same acne/stomach/gastritis topic area -- a new blocker layer, not a contradiction, and not the D-056.4 defect recurring)

READY FOR HUMAN VISUAL REVIEW? NO (nothing rendered -- Freeze blocked before Boundary/Render)

Then STOP. Did not patch. Did not launch another RAW.

## D-056.6 / D-057 -- content-loss and Resolver-authority forensics (report only, no code)

Two report-only directives, delivered in chat, not code checkpoints on their own -- superseded by D-058's actual fixes below, summarized here so the D-058 entry has context without re-deriving it:

D-056.6 collected the D-056.5 canary's 7 content-loss findings and traced them to 3 root decisions (not 7 independent losses): (A) pre-Resolver deterministic grouping wrongly merged two distinct symptom beats ("back acne" / "hormonal pimples behind the ear-neck") into one mutually-exclusive retry contest; (B) a downstream authority reversed a correct, twice-independently-confirmed semantic winner for a genuine 3-way stomach/gastritis retry family back to a lower-confidence, incomplete take; (C) a validator-side coverage/alignment false positive on a genuinely-preserved, differently-worded 5-10%-hereditary claim.

D-057 traced (B) to its exact source: `realization_resolver.py`'s `_pick_winner`, inside the Unified Resolver itself (not a downstream mutator -- confirmed 0 SEMANTIC_MUTATORs exist anywhere after `apply_authoritative_realization_resolution`), ranked candidates that already satisfied full critical-claim coverage by raw DeliveryScorer score first, never consulting the Semantic Ledger's own already-recorded `SEMANTIC_WINNER_OVERRIDE` evidence. (A) was confirmed strictly pre-Resolver (an Idea Formation problem). (C) was confirmed a genuine VALIDATOR false positive, not a Selection failure.

## D-058 -- grouping + Resolver decision-quality targeted fixes

Three independent, targeted fixes for D-057's findings -- no new authority layer, no post-Resolver semantic mutator added (confirmed still 0 after this directive).

**Phase 1 (distinct-idea grouping safety)**: new `take_grouping_provider.split_incohesive_retry_groups`, called in `pipeline.py` immediately after `reconcile_semantic_idea_equivalence` (the single choke point that function's own docstring already establishes), before Best Take ranking ever treats a group as one mutually-exclusive contest. Every pair inside an already-multi-member group must now show either strong deterministic lexical evidence of being the same retry (`_provider_members_compatible`'s existing complete-link threshold, reused verbatim) or explicit arbiter confirmation for that specific pair; neither temporal proximity nor shared vocabulary/topic alone is ever sufficient. A pair with neither is split into its own connected component, so a genuinely distinct beat gets an independent chance to survive Selection instead of losing a contest it was never actually part of. `protected_ids` (accepted composite pieces) are never re-examined, same contract as the arbiter merge step.

**Phase 2 (Resolver evidence hierarchy)**: `realization_resolver.py`'s `_pick_winner` (inside `_resolve_one_idea`) now ranks candidates that already satisfy full critical coverage by an explicit hierarchy -- (1) semantic validity/completeness (`RealizationRecord.complete_idea is not False`, D-050C1.6's own field, never guessed at as incomplete when unknown), (2) high-confidence semantic winner evidence (a new `_semantic_winner_confidence_by_realization` helper reads the Ledger's own `SEMANTIC_WINNER_OVERRIDE` decisions, at or above the same 0.85 confidence floor `pipeline.py`'s `_semantic_best_take` already uses), (3) critical claim coverage quality (individual critical claims covered, a finer signal than the boolean "covers all critical groups" gate every pool member already passes), (4) delivery quality (DeliveryScorer/watch-listen score, now a tiebreaker only), (5) contextual richness (unchanged). `semantic_ledger.py`'s own `SEMANTIC_WINNER_OVERRIDE` recording now carries the arbiter's confidence in its evidence (previously unrecorded). When two candidates both carry conflicting high-confidence semantic winner evidence, the idea resolves `REVIEW_REQUIRED` (`conflicting_high_confidence_semantic_winner_evidence`) rather than guessing.

**Phase 3 (validator claim-paraphrase alignment, separate code path from Phases 1-2 as directed)**: `semantic_claims.py`'s `AMBIGUOUS_COVERAGE_FLOOR` lowered from 0.3 to 0.10 (the live D-056.6 false positive's own coverage was 0.15 -- a genuine paraphrase that never reached the bounded claim-equivalence arbiter under the old floor). A new, floor-independent `_DEFINITIVE_MISMATCH_COVERAGE_CAP` (0.05) keeps three deterministic "definitely not the same claim" guards inside `claim_coverage` -- negation flip (pre-existing, now using the fixed cap instead of a floor-relative one), number/percentage change (new), and connector-based causal-direction inversion (new, scoped to `_CAUSE_EFFECT_MARKERS`, the existing marker vocabulary) -- confidently below the new, wider ambiguous band regardless of how low the floor is tuned, so none of the three can ever drift into arbiter-eligible territory. A bare, connector-less causal-verb inversion ("X triggers Y" vs "Y triggers X") remains an honest, documented gap in bag-of-words coverage -- the same class `contradiction_signal.py`'s own module docstring already declares out of scope for that primitive; fixing it would require preserving token order/dependency roles, a materially new representation this directive's "no new architecture" scope excludes.

**Tests**: 3 new files, 22 tests total -- `test_cutsell_d058_phase1_grouping_safety.py` (10, including a generic reproduction of the exact D-057 three-clip shape: two true back-acne retries plus one genuinely distinct beat, arbiter confirms only the true retry pair), `test_cutsell_d058_phase2_resolver_evidence_hierarchy.py` (5, including the exact D-057 gastritis shape: higher-score-but-incomplete vs lower-score-but-complete-and-semantically-confirmed), `test_cutsell_d058_phase3_claim_paraphrase_alignment.py` (7, covering paraphrase-now-covered, number/negation/entity/causal-still-not-covered, and fail-open-without-an-arbiter unchanged).

**Offline qualification**: compileall clean. D-050 through D-058 targeted suite: 276 passed (254 baseline + 22 new). CleanCutBench LEGACY+AUTHORITATIVE: 54/54 fixtures both modes, identical 23-same/31-different/3-review_required split as the pre-D-058 baseline, 0 regressions. Full `tests/test_cutsell_*.py`: 1612 passed (1590 baseline + 22 new), 0 regressions. Full `tests/` (minus `test_semantic_stitch.py`): 2236 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record since D-056.1.

### Final report (verbatim, as delivered)
D-058 COMPLETE

GROUPING DISTINCT-IDEA SAFETY: PASS
RESOLVER EVIDENCE HIERARCHY: PASS
GASTRITIS GENERIC FIX: PASS
PIMPLES GENERIC FIX: PASS
CLAIM PARAPHRASE ALIGNMENT: PASS
POST-RESOLVER SEMANTIC MUTATORS: 0
LEGACY: 54/54
AUTHORITATIVE: 54/54
FULL TEST COUNT: 1612 passed (tests/test_cutsell_*.py), 0 regressions vs the 1590-test D-056.5 baseline; full tests/ (minus test_semantic_stitch.py) 2236 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record.
READY FOR ONE VIDEO00 CANARY? YES

Then STOP. Do not launch Modal.

## D-058 canary -- final Video00 Modal run validating the three structural fixes live

One authorized run (run [33814961471](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33814961471), commit `7be38e4`, fixed config unchanged, ~4.4 min GPU wall-clock, clean completion, teardown confirmed). No code changes before or after.

**Pimples/acne (Phase 1)**: the exact three-clip shape recurred under fresh clip ids -- "Por temporada, me salía un acné en la espalda con la que yo resolvía con resorcina." plus THREE separate hormonal-pimples-behind-the-ear/neck clips. All four now survive to the final KEEP sequence as independent beats -- zero contradiction/duplicate/retry findings on any of them. Human Gold confirms directly: `pimples_micro_1_present`, `pimples_micro_2_present`, `pimples_micro_3_present`, `pimples_bad_monolith_absent`, `pimples_later_winner_present`, `pimples_micro_order`, and `acne_back_preserved` all **PASSED** (all FAILED in the pre-D-058 canary). **GROUPING DISTINCT-IDEA SAFETY: PASS.**

**Gastritis (Phase 2)**: the complete realization ("...dijeron que tenía gastritis. Nada severo pero tenía gastritis y me mandaron tres meses con pastillas.") is the one that survived to KEEP, verbatim-matching Human Gold's own gold text exactly. The two competing incomplete/thinner takes (the "...me diagnosticaron con..." cutoff and the bare "2023" mention) were correctly discarded. Human Gold confirms: `gastritis_preserved` **PASSED** (failed in the pre-D-058 canary). **RESOLVER EVIDENCE HIERARCHY: PASS.**

**Hereditary 5-10% (Phase 3)**: the winning realization (the full "Esta es mi experiencia..." statement) still carries the same "5-10%" fact verbatim, and `contradiction_findings: []` confirms the D-056.5 fix still holds. However, `_lost_critical_claims` still reports this exact claim `CRITICAL_CLAIM_LOST` (`coverage_against_winning_realization: 0.05`) -- a **newly discovered, different residual bug** in the same mechanism, not the one D-058 Phase 3 targeted: the claim's own content tokens (`{"convencida","avala","ciencia","solo","así","estoy"}`, since digits are excluded from `_content()`) never reach `_MIN_SHARED_FOR_RELEVANCE` (2 tokens) against any single sentence of the winning realization -- each candidate sentence shares at most 1 token ("solo" or "así") with the claim. `claim_coverage`'s relevance-scoping therefore falls back to the WHOLE candidate text (`relevant_sentences if relevant_sentences else candidate_text`), which contains an earlier, unrelated negation ("Por eso no creo... son hereditarios") that then trips the negation-flip guard and caps coverage at `_DEFINITIVE_MISMATCH_COVERAGE_CAP` (0.05) -- reproducing, through a different code path, the same class of whole-text-scope false positive D-056.4/D-056.5 already fixed once for `contradiction_signal.py`. Confirmed via local reproduction of the exact live claim/candidate text pair. **CLAIM PARAPHRASE ALIGNMENT: FAIL** (the deterministic number/negation/causal mismatch guards themselves are confirmed correct and unregressed by the 22-test D-058 offline suite; this is a distinct, not-yet-fixed relevance-scoping gap in the same function).

**Authoritative resolver**: 19 semantic ideas -- 18 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE, shadow and authority tracks agree (19/19). 2 orphan realizations correctly routed `REVIEW_REQUIRED` (no verified replacement, explicitly flagged not silently confirmed). Zero invalid composites, zero silent zero-realization ideas.

**Human Gold**: 16/18 (up from 8/18 pre-D-058). Failed: `papillary_cancer_preserved`, `sonography_good_before_diagnosis` -- both unrelated to any of the three D-058 shapes (a separate thyroid-nodule/biopsy-sequencing area).

**Architecture**: PASS, 0 failed checks.

**Freeze**: BLOCKED -- on the single residual `CRITICAL_CLAIM_LOST` finding above, plus 2 pre-existing, unrelated `UNIQUE_FACT_LOST` findings (a stray "Síntomas que tuve..." aside and an incidental "2023" date) that were never part of D-058's scope.

**Render/QC**: not reached -- Freeze blocked, no render attempted, `delivery_status: NOT_DELIVERABLE_not_attempted`, `deliverable: false`.

**Single root blocker**: Category C -- coverage/alignment false positive, in `claim_coverage`'s relevance-scoping fallback (see above). Not a grouping problem, not a resolver-winner problem, not genuine content loss, not a genuine contradiction.

### Final report (verbatim, as delivered)
D-058 VIDEO00 CANARY COMPLETE

GROUPING DISTINCT-IDEA SAFETY: PASS
RESOLVER EVIDENCE HIERARCHY: PASS
CLAIM PARAPHRASE ALIGNMENT: FAIL
AUTHORITATIVE ENGINE: PASS
Human Gold: 16/18
Architecture: PASS
CanonicalEditPlan: PASS
FinalEditReviewer: PASS (correctly caught the residual finding -- functioned correctly; the underlying claim-coverage computation it reads from is what's wrong)
Freeze: BLOCKED
Render: NO
PostRender QC: NOT_REACHED
delivery_status: NOT_DELIVERABLE_not_attempted
deliverable: false

IF BLOCKED:
single root blocker: C. coverage/alignment false positive -- `claim_coverage`'s relevance-scoping falls back to the whole candidate text when no single sentence shares >=2 tokens with a heavily-paraphrased claim, and the whole-text fallback picks up an unrelated earlier negation, capping coverage at the fixed mismatch value. A newly discovered gap, distinct from the one D-058 Phase 3 fixed (which is confirmed correct and unregressed offline).

READY FOR HUMAN VISUAL REVIEW: NO (nothing rendered -- Freeze blocked before Boundary/Render)

Then STOP. Do not patch. Do not launch another RAW.

## D-059 -- claim-coverage proposition scope fix

Targeted validator fix for the D-058 canary's own residual finding -- no new architecture, no changes to Unified Resolver, grouping, BestTake, ASR, Human Gold, Semantic Ledger, Freeze, Boundary, or Render/QC.

**Root scope bug**: `semantic_claims.claim_coverage`'s relevance-scoping required a candidate sentence to clear a general content-token bar (`min(2, claim_token_count)` shared tokens) before treating it as the sentence the claim is actually about. When a claim's own surrounding scaffolding was paraphrased heavily enough (near-zero shared ordinary words) that no sentence cleared that bar, the function fell back to the WHOLE candidate text for its negation/number/causal mismatch guards -- so an unrelated clause's own negation could poison a claim it was never actually about. Live shape (D-058 canary): an earlier "no creo... son hereditarios" clause (rejecting a broader, unrelated claim) capped coverage for a completely separate, later, correctly-restated "5-10%" quantitative claim whose own phrasing barely overlapped the claim's own scaffolding words.

**Fix (two parts, both inside `claim_coverage`)**:
1. When some candidate sentence shares an EXACT number with the claim, that sentence takes PRIORITY over the general content-token test as the relevance signal -- reuses `contradiction_signal._clauses_address_same_proposition`'s own D-056.5 anchor ("a shared number is decisive on its own"), not reinvented. Priority, not union: unioning would have let a coincidentally-overlapping unrelated sentence (sharing generic scaffolding like "science backs this up") merge into scope alongside the real match, reproducing the same bug. This is a match-only signal -- a genuinely different number never "shares" one, so a real number mismatch still falls through to the general content-token test untouched.
2. When truly no sentence is relevant under either test, the negation/number/causal guards are **skipped entirely** -- never fall back to the whole candidate text. The plain overlap ratio (already computed whole-text-scoped, independent of this per-sentence check) stands alone: a genuine low-overlap paraphrase still reaches the ambiguous band and gets a real chance at the bounded claim-equivalence arbiter (`resolve_ambiguous_coverage`, unchanged); a claim with no credible matching content anywhere is honestly reported not covered via low raw overlap, never via a fabricated mismatch cap -- the distinction is provable: the resulting coverage value is never exactly `_DEFINITIVE_MISMATCH_COVERAGE_CAP` in this case.

**Verified against the exact live D-058 canary claim/candidate text pair**: coverage now 0.333 (ambiguous band, arbiter-confirmable) -- was 0.05 (falsely capped).

**Tests**: `tests/test_cutsell_d059_claim_coverage_proposition_scope.py`, 9 new tests -- unrelated negation before/after a valid paraphrased claim (covered, never poisoned), same-proposition negation/number/causal mismatch (still not covered), low-overlap clear paraphrase reaching the arbiter, no credible matching proposition (honestly not covered via low overlap, not a fabricated mismatch), multi-sentence clip with only one relevant clause, and a generic reproduction of the exact D-058 canary shape (a rhetorical negation of a broader unrelated claim, followed by a heavily-paraphrased restatement of the shared number).

**No safety weakening**: true negation/number/causal mismatch on the SAME matched proposition are all still confidently capped and unreachable by the arbiter (proven by the 3 same-proposition mismatch tests above). D-056.3's 17 contradiction-safe-composite tests and D-056.5's 16 proposition-completeness tests remain green unchanged (33/33) -- `contradiction_signal.py` itself was never touched.

**Offline qualification**: compileall clean. D-050 through D-059 targeted suite: 285 passed (276 baseline + 9 new). CleanCutBench LEGACY+AUTHORITATIVE: 54/54 fixtures both modes, identical 23/31/3 split, 0 regressions. Full `tests/test_cutsell_*.py`: 1621 passed (1612 baseline + 9 new), 0 regressions. Full `tests/` (minus `test_semantic_stitch.py`): 2245 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record since D-056.1.

### Final report (verbatim, as delivered)
D-059 COMPLETE

ROOT SCOPE BUG: `claim_coverage`'s relevance-scoping fell back to the whole candidate text for its negation/number/causal mismatch guards whenever no single sentence cleared the general content-token relevance bar, letting an unrelated clause's own negation poison a claim it was never about.
PROPOSITION-LEVEL COVERAGE: PASS
UNRELATED NEGATION FALSE POSITIVE: FIXED
TRUE NEGATION SAFETY: PASS
NUMBER SAFETY: PASS
CAUSAL SAFETY: PASS
Human Gold fixtures: PASS (offline; the exact live claim/candidate pair now resolves to the ambiguous band, arbiter-confirmable, matching the D-058 canary's own Human Gold-passing content)
LEGACY: 54/54
AUTHORITATIVE: 54/54
FULL TEST COUNT: 1621 passed (tests/test_cutsell_*.py), 0 regressions vs the 1612-test D-058 baseline; full tests/ (minus test_semantic_stitch.py) 2245 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record.
READY FOR ONE FINAL VIDEO00 CANARY? YES

Then STOP. Do not launch Modal.

## D-059 canary -- final Video00 Modal run validating the proposition-scope fix live

One authorized run (run [33819516264](https://github.com/AutomatedRetailServices/EditDNA-worker/actions/runs/33819516264), commit `db8a32d`, fixed config unchanged, retries=0, clean completion). No code changes before or after.

**Hereditary 5-10% claim (D-059's own target)**: `coverage_against_winning_realization` for `claim_9387121066f1` (idea `tg_408fb4ac93d8dffb4f`, winning realization `clip_50308b2721b5fee0d949`) is now **0.3333** -- up from the D-058 canary's fabricated-mismatch value of 0.05. This is direct, confirmed proof the fix works exactly as designed: the shared-number-anchor proposition scope excluded the winning realization's own unrelated earlier negation clause, and the score landed in the genuine ambiguous band (`0.10 <= 0.3333 < 0.6`) instead of the fixed mismatch cap. **D-059 CLAIM SCOPE: PASS.**

**Why Freeze still blocks on this claim anyway**: `resolve_ambiguous_coverage`'s ambiguous band requires a live `claim_equivalence_arbiter` to confirm the paraphrase; grep of `flow_b.py` (the real production entry point) confirms only `semantic_equivalence_arbiter` is ever instantiated and passed through the call chain (`universal_clean_cut.py` -> `final_story_coherence_validation.py` -> `realization_resolver.py`) -- `claim_equivalence_arbiter` is a fully-wired, never-instantiated dead parameter. So the ambiguous band always receives `arbiter=None` and fails open to "not covered" per the pre-existing, unchanged "WHEN UNCERTAIN, KEEP [the finding]" design. This is a **separate, newly discovered, out-of-scope structural gap** -- not a flaw in D-059's actual code, and not a regression of anything D-059 touched.

**Safety preservation**: `contradiction_findings: []` (D-056.5 holds); `unresolved_families: []` (the hereditary retry family itself merged cleanly, confidence 0.9); all 3 same-proposition mismatch guards (negation/number/causal) remain green in the unchanged offline suite; no genuine mismatch case appeared live this run to additionally exercise on real data.

**Authoritative engine**: 18 semantic ideas -- 17 RESOLVED_WINNER + 1 RESOLVED_COMPOSITE, 3 unresolved orphans correctly routed `REVIEW_REQUIRED` (not guessed). Zero unsafe discards, zero invalid composites, zero silent zero-realization ideas (`missing_idea_coverage: []`).

**Human Gold**: 14/18 (down from the D-058 canary's 16/18). All D-058-targeted checks still PASS (`gastritis_preserved`, `family_context_preserved`, `acne_back_preserved`, `pimples_micro_1_present`, `pimples_micro_3_present`, `pimples_bad_monolith_absent`, `pimples_later_winner_present`). Failed: `papillary_cancer_preserved`, `pimples_micro_2_present`, `pimples_micro_order`, `sonography_good_before_diagnosis` -- the 2 new failures beyond the D-058 canary's original 2 look like run-to-run ASR/transcript nondeterminism, not investigated further per directive (report only, do not patch).

**Architecture**: PASS, 0 failed checks. **CanonicalEditPlan** (`plan_8c19e0575518de38`, v1): created, 6 top-level ideas, 0 composite, all `coverage_status: complete`; `validation_state: freeze_blocked_pending_review`.

**FinalEditReviewer**: `status: "FAIL"`, 2 blocking findings -- (1) `UNIQUE_FACT_LOST` on `clip_e065a01173052bbf8c78` ("Síntomas que tuve...", coverage 0.4), entirely unrelated to D-058/D-059; `repair_loop` reports `reason: "no_repair_strategy_exists_for_this_finding_kind"` for it -- a structural repair-loop coverage gap, not investigated further as it predates and is outside D-059's scope; (2) `CRITICAL_CLAIM_LOST` on the hereditary claim (above). 1 non-blocking warning (an incidental "2023" date atom, classified CONTEXTUAL).

**Freeze**: BLOCKED on both findings above. **Render/QC**: not reached, `delivery_status: NOT_DELIVERABLE_not_attempted`, `deliverable: false`.

**Single root blocker (earliest in FinalEditReviewer's findings array)**: Category F -- `UNIQUE_FACT_LOST` on `clip_e065a01173052bbf8c78`, for which `repair_loop` has no repair strategy for this finding kind at all (structural gap). The directive's primary-interest blocker, `CRITICAL_CLAIM_LOST` on the hereditary claim, is also Category F -- D-059's fix is proven correct; the residual gap is the missing live `claim_equivalence_arbiter` wiring in `flow_b.py`, not a grouping, resolver-winner, coverage-math, content-loss, or contradiction defect.

### Final report (verbatim, as delivered)
D-059 FINAL VIDEO00 CANARY COMPLETE
D-059 CLAIM SCOPE: PASS
TRUE NEGATION SAFETY: PASS
NUMBER SAFETY: PASS
CAUSAL SAFETY: PASS
AUTHORITATIVE ENGINE: PASS
Human Gold: 14/18
Architecture: PASS
CanonicalEditPlan: PASS
FinalEditReviewer: FAIL
Freeze: BLOCKED
Render: NO
PostRender QC: NOT_REACHED
delivery_status: NOT_DELIVERABLE_not_attempted
deliverable: false
IF BLOCKED: single root blocker: UNIQUE_FACT_LOST on clip_e065a01173052bbf8c78 (StoryValidator) -- repair_loop has no repair strategy for this finding kind (structural gap, category F), unrelated to D-058/D-059. Secondary, D-059-relevant blocker: CRITICAL_CLAIM_LOST on the hereditary claim -- D-059's fix is proven correct (coverage moved from fabricated mismatch 0.05 to genuine ambiguous 0.3333), but no live claim_equivalence_arbiter is wired in flow_b.py to resolve the ambiguous band (structural gap, category F).
READY FOR HUMAN VISUAL REVIEW: NO

Then STOP. Did not patch. Did not launch another RAW.

## D-061 -- semantic-equivalence-aware validation (D-060 final Freeze-blocker integration)

Fixes D-060's two proven false positives -- no new semantic authority, no new provider/model, no weakening of Freeze, no changes to Grouping/BestTake/Unified Resolver/Semantic Ledger/ASR/Human Gold/Freeze policy/Boundary/Render-QC.

**Phase 1 (same-idea paraphrase credit)**: new `_same_idea_paraphrase_credit` in `final_story_coherence_validation.py`, consulted only inside `_lost_semantic_atoms`'s existing content-loss trigger. A POST-GROUPING discarded clip (a member of a genuine 2+-member `take_judge_groups` retry contest) whose own family has a SELECTED sibling is credited -- and the finding relabeled `SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION`, `blocking: false` -- only when an EXISTING `semantic_idea_equivalence` merge record for that exact pair already confirms it at >=0.85 confidence (D-058 Phase 2's own bar, reused). No new paid call: this reuses evidence the arbiter already produced during grouping. A deterministic idea-scoped word-overlap fallback was designed, proven mathematically inert (idea-scoped coverage can never exceed whole-video coverage, since the winner's own text is already part of the whole-video comparison -- a genuine near-duplicate is *already* handled correctly by the existing, unmodified check with no fix needed, so this fallback would never once fire), and deliberately removed rather than shipped as dead code. Pre-group-rejected/ungrouped/hybrid-deleted clips (no entry in the group map) and genuinely additive facts (no matching pairwise evidence) are completely unaffected -- current fail-closed behavior unchanged.

**Phase 2 (claim-equivalence arbiter wiring)**: new `claim_equivalence_google.py` (`GoogleClaimEquivalenceArbiter`, implementing `semantic_claims.ClaimEquivalenceArbiter.claim_covered`), mirroring `semantic_idea_equivalence_google.py`'s existing Gemini-transport pattern exactly -- same provider (`google`), same model (`gemini-3.5-flash-lite`), its own independent `DollarBudgetLedger`/cost ceiling (`max_cost_per_claim_equivalence_call_usd`, new `hybrid_provider_settings.py` field, default $0.003, own env var `CUTSELL_HYBRID_MAX_CLAIM_EQUIVALENCE_USD`). `BrainRuntime` gains a `claim_equivalence_arbiter` field + `_build_claim_equivalence_arbiter` factory, gated the same way `semantic_equivalence_arbiter` already is (`requested_hybrid` + a new independent rollback flag `CUTSELL_CLAIM_EQUIVALENCE_ARBITER`, default on). **The actual root fix**: `universal_clean_cut_validation.py` -- the real production/RAW-harness call site, confirmed by direct trace to be the exact gap D-059/D-060 identified -- now passes `claim_equivalence_arbiter=brain.claim_equivalence_arbiter` into `process_universal_clean_cut_sources`, which already correctly threaded the parameter through `ClaimCoverageBestTake` and `StoryValidator` but never received a live instance before. No changes to `claim_coverage`/`resolve_ambiguous_coverage` themselves (D-059's proposition-scoping fix and D-038's deterministic mismatch guards are untouched) -- the arbiter is structurally unreachable for any confidently-covered or confidently-mismatched claim (`resolve_ambiguous_coverage` only ever calls it inside the genuinely ambiguous band), so it can never override a number/negation/entity/causal-direction mismatch; the prompt additionally, defensively instructs the model to answer NOT covered on any such difference or genuine uncertainty, belt-and-braces on top of the structural guarantee.

Noted side effect, in scope, not new: `claim_coverage_best_take.py` (D-048's BestTake-level claim-criticality override) already accepted this same `claim_equivalence_arbiter` parameter and already called `resolve_ambiguous_coverage` through it -- wiring one real instance at the `BrainRuntime` root activates BOTH consumers simultaneously, since they share the one instance. This is not a new authority (D-048 already existed, pre-Freeze, part of BestTakeResolver); it lets an ambiguous paraphrase get recognized earlier, before StoryValidator would otherwise need to flag it. 2 new targeted tests confirm the pre-D-061 fail-open behavior is unchanged without an arbiter, and the new behavior is safe with one.

**Phase 3 (validator diagnostics consistency)**: `_lost_semantic_atoms` rows now carry a `classification` field (`REAL_CONTENT_LOSS` / `SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION`) and, when Phase 1's credit suppressed the finding, `content_loss_suppressed_by`. `_lost_critical_claims` now returns `(findings, confirmations)` -- `confirmations` is a new, additive, observability-only diagnostics list (`claim_coverage_confirmations`) recording every claim the ambiguous band resolved to covered via the arbiter (the *only* way that band can ever resolve to covered, by `resolve_ambiguous_coverage`'s own construction) with `resolution: "claim_equivalence_arbiter_confirmed"` -- deliberately kept OUT of `lost_critical_claims` itself (every row there is unconditionally blocking by construction; adding confirmed-covered rows to it would falsely block Freeze) and never wired into `freeze_blocked`. Nothing is ever silently hidden: a suppressed or confirmed finding is always visible in diagnostics, just correctly no longer blocking. The RAW workflow's diagnostic printing was updated to surface both new fields directly.

**Tests**: 40 new -- `test_cutsell_d061_phase1_same_idea_paraphrase_credit.py` (7: explicit-evidence credit, genuinely-additive-still-blocks, different-idea-still-blocks, pre-group-reject fail-closed, hybrid-delete-no-replacement fail-closed, exact-duplicate/semantic-paraphrase regression-confirmed-unaffected), `test_cutsell_d061_claim_equivalence_google.py` (13, transport-reliability, mirrors the semantic-equivalence arbiter's own test depth: request/response contract, retry, cost-ledger, malformed-response, model-policy, missing-key), `test_cutsell_d061_phase2_claim_equivalence_wiring.py` (15: BrainRuntime construction/gating/rollback/cost-ceiling + a call-counting-arbiter safety matrix proving number/negation/causal-inversion mismatches never even reach the arbiter, plus 2 end-to-end `apply_final_story_coherence_validation` integration tests), `test_cutsell_d061_phase3_validator_diagnostics_consistency.py` (3), plus 2 added to the pre-existing `test_cutsell_claim_coverage_best_take.py` covering the newly-activated second consumer.

**Offline qualification**: compileall clean across `cutsell_worker/`. `tests/test_cutsell_*.py`: 2285-baseline-consistent count with 0 regressions (see final report below for the exact number). CleanCutBench LEGACY+AUTHORITATIVE: 54/54 fixtures both modes (`test_cutsell_clean_cut_core_evaluation_suite.py` + `test_cutsell_d050c1_5_full_cleancutbench_parity.py`, 55 passed together). D-056.3/D-056.5/D-058/D-059-specific test files: 64/64 passed, explicitly re-confirmed green. Full `tests/` (minus `test_semantic_stitch.py`): 2285 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record since D-056.1 (`test_hybrid_story_guard_incomplete_retry.py::test_incomplete_failed_retry_is_covered_when_prior_delivery_preserves_numbers_and_negation`, `test_video00_modal_hybrid_semantic_parity.py::test_the_two_overlay_values_are_never_masked_in_ci_logs`).

**QA_ENGINE**: no dedicated `QA_ENGINE` tool, skill, or script exists in this session or repository (confirmed by search). Ran as a self-conducted independent adversarial review pass instead (explicitly disclosed as such, not represented as a separate system): re-derived the live `take_judge_groups`/`semantic_idea_equivalence` diagnostics shape directly from `pipeline.py` to confirm Phase 1's assumptions hold against real pipeline output (not just fixtures); traced the second-consumer (`claim_coverage_best_take.py`) side effect of Phase 2's wiring, found it had zero existing arbiter-path tests, and added 2; confirmed zero interaction with CompositeResolver/accepted-composite pieces (Phase 1 only ever inspects `draft.discarded`, composite pieces are always in `draft.selected`); confirmed the mathematically-inert idea-scoped-overlap fallback design flaw myself before shipping it, and removed it rather than leaving dead/misleading code. One informational, pre-existing, out-of-scope observation surfaced (not introduced by D-061, not fixed by D-061): `resolve_ambiguous_coverage` (D-038) ignores the arbiter's own returned confidence value entirely, relying solely on prompt-level instructions to fold low-confidence uncertainty into `covered=false` -- a real gap, but in code this directive was explicitly told not to modify (`claim_coverage`/`resolve_ambiguous_coverage` themselves), and not a regression.

### Final report (verbatim, as delivered)
D-061 COMPLETE

SAME-IDEA PARAPHRASE CREDIT: PASS
PRE-GROUP CONTENT SAFETY: PASS
CLAIM EQUIVALENCE ARBITER: WIRED
AMBIGUOUS CLAIM RESOLUTION: PASS
NUMBER SAFETY: PASS
NEGATION SAFETY: PASS
ENTITY/DIAGNOSIS SAFETY: PASS
CAUSAL SAFETY: PASS
POST-RESOLVER SEMANTIC MUTATORS: 0
FREEZE POLICY CHANGED: NO
LEGACY: 54/54
AUTHORITATIVE: 54/54
FULL TEST COUNT: tests/test_cutsell_*.py 1661 passed (1621 D-059 baseline + 40 new), 0 regressions; full tests/ (minus test_semantic_stitch.py) 2285 passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures already on record.
QA_ENGINE VERDICT: PASS_WITH_KNOWN_ISSUES
P0: 0
P1: 0
P2: 0
P3: 1 (pre-existing, out-of-scope: resolve_ambiguous_coverage ignores arbiter confidence value, D-038, not modified or introduced here)
READY FOR ONE VIDEO00 CANARY: YES

Then STOP. Do not launch Modal.

## D-062 -- CutSell Commercial Engineering Operating Model (governance, no engine change)

Canonical governance checkpoint, not an engineering directive: adds
`docs/CUTSELL_COMMERCIAL_ENGINEERING_OPERATING_MODEL.md`, the permanent
operating model for CutSell as it moves from benchmark-driven engine
development toward a commercial product. Registers 11 canonical roles
(Product Owner/MVP Authority, Platform/Application Engineer, AI/Video Engine
Engineer, QA/Release Engineer, Security & Privacy Engineer, SRE/DevOps
Engineer, Product/UX Engineer, FinOps/AI Cost Engineer, Data/Product
Analytics, Trust/Compliance Owner, Human Editorial Acceptance), the
separation-of-duties doctrine (no single role self-certifies the complete
product -- the AI/Video Engine Engineer that implements a fix is never the
same authority that certifies it release-ready), 7 canonical gates
(ENGINE/QA/SECURITY/RELIABILITY/EDITORIAL_ACCEPTANCE/ECONOMICS/RELEASE),
P0-P3 defect severity, a commercial QA coverage matrix (AUTH through
PERFORMANCE), development checkpoints tying each gate to a project stage,
12 reusable canonical review command definitions (contracts only, not
implemented as runnable tools by this checkpoint), the Standard Review
Report template, a release-readiness contract JSON Schema (contract only,
no UI, no runtime validator), and a registry of existing QA assets
(CleanCutBench, Human Gold, Architecture Validator, StoryValidator,
FinalEditReviewer, Selection Freeze, PostRenderWatchListenQC, Semantic
Ledger/Unified Resolver tests, LEGACY/AUTHORITATIVE regression, Modal
canaries, canonical identity/provenance tests) placed under those roles and
gates by reference, not duplicated.

**No engine file changed.** No code was modified, no test was run, no Modal
or RunPod infrastructure was touched. D-061 (already committed at `9919ae2`
on `feature/runpod-pod-on-demand`, working tree clean at the time of this
checkpoint) was neither modified nor rerun.

**Handoff**: per this operating model's Section 11, the correct next formal
action against the existing D-061 build is `RUN QA_ENGINE` -- an
independent review of D-061's implementation, tests, negative cases,
semantic-safety guards, Freeze behavior, post-Resolver immutability,
regression evidence, claim-equivalence arbiter wiring, and same-idea
paraphrase-credit behavior. Only after that verdict is recorded may a paid
Modal canary for D-061 be authorized. This checkpoint does not itself run
that review.

### Final report (verbatim, as delivered)
CUTSELL COMMERCIAL ENGINEERING OPERATING MODEL ADDED

CANONICAL ROLES: 11
CANONICAL GATES: 7
QA MODES: QA_COMPONENT, QA_ENGINE, QA_INTEGRATION, QA_E2E, QA_MOBILE, QA_REGRESSION, QA_EXPLORATORY, QA_PERFORMANCE, QA_RELEASE
REUSABLE REVIEW COMMANDS: RUN ENGINE_GATE, RUN QA_COMPONENT, RUN QA_ENGINE, RUN QA_INTEGRATION, RUN QA_E2E, RUN QA_MOBILE, RUN QA_REGRESSION, RUN QA_EXPLORATORY, RUN QA_PERFORMANCE, RUN QA_RELEASE, RUN SECURITY_REVIEW, RUN SRE_READINESS, RUN COST_REVIEW, RUN EDITORIAL_ACCEPTANCE, RUN BETA_READINESS, RUN RELEASE_GATE
SECURITY MODEL: REGISTERED
SRE MODEL: REGISTERED
FINOPS MODEL: REGISTERED
DATA/ANALYTICS MODEL: REGISTERED
TRUST/COMPLIANCE MODEL: REGISTERED
HUMAN EDITORIAL ACCEPTANCE: REGISTERED
COMMERCIAL QA MATRIX: REGISTERED
RELEASE READINESS CONTRACT: CREATED
D-061 MODIFIED: NO
D-061 RERUN: NO
CURRENT DEVELOPMENT BLOCKED: NO
NEXT FORMAL ACTION: RUN QA_ENGINE against existing D-061 implementation

Then STOP. No Modal. No RunPod. No paid tests. No engine behavior changes.

## D-062.1 -- UNRESOLVED_RETRY final blocker forensic (report only, no code, no RAW)

Forensic trace of the post-D-061-canary Freeze blocker on idea
`tg_c7c1ae9f22e6c10986` (clips `clip_5e34ebd314daf98ff730` [A] and
`clip_0b90ef2e20d08e8e7ec5` [B]). A contains a distinct biopsy/diagnosis-
confirmation fact plus the shared reflective statement that B also carries;
B has only the reflective statement, with slightly better delivery. A should
have auto-won via critical-content completeness, but the resolver's
"conflicting high-confidence semantic-winner labels -> REVIEW_REQUIRED"
branch fired without first checking claim-coverage dominance between the two
candidates.

**Root cause**: resolver too conservative -- a missing check, not a missing
signal. The evidence needed to resolve this automatically (A's superset
critical-claim coverage over B) was already available; the resolver simply
never consulted it before falling to `REVIEW_REQUIRED` on the conflicting
semantic-winner labels.

**Smallest missing contract identified**: check critical-claim-coverage
dominance between conflicting semantic-winner candidates before falling to
`REVIEW_REQUIRED`. No code changed under this forensic; the fix itself is
deferred to a future directive (see D-062.2's `CRITICAL_COVERAGE_DOMINANCE`
concept, which generalizes this specific finding).

No RAW run. No code modified.

## D-062.2 -- CutSell Editorial Resolution & Human Escalation Contract (doctrine, no engine change)

Canonical doctrine checkpoint, not an engineering directive: adds
`docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md`.
Registers the Automatic Editor Doctrine (CutSell is the editorial
decision-maker; human choice is a last resort, never a convenience valve for
resolver uncertainty); Semantic Dominance Before Performance; the
`CRITICAL_COVERAGE_DOMINANCE` concept (strict-superset critical-claim
coverage settles a comparison before delivery quality is consulted); a
16-layer Automatic Resolution Hierarchy (explicitly conceptual precedence,
not hard-coded weights or a scoring formula -- higher layers cannot be
silently overridden by lower layers); `CONTEXTUAL_DELIVERY_FIT` (delivery
quality consulted only once no higher-precedence semantic signal already
decided the outcome); three escalation classes -- `REVIEW_REQUIRED_SEMANTIC`
(irreducible, evidence-symmetric ambiguity, no dominance signal available),
`AUTO_RESOLVED_LOW_MARGIN` (a winner was produced, but narrowly, recorded for
observability), `HUMAN_CHOICE_ELIGIBLE` (the rare, affirmatively-determined
subset of `REVIEW_REQUIRED_SEMANTIC` presentable to a user as a plain A/B
choice); D-062.1 registered as the canonical example of a
NOT-human-choice-eligible case (a strict resolver defect -- coverage
dominance should have auto-resolved it -- generic pattern only, no
Video00-specific text/ids encoded into production logic); the Human Choice
Contract (uncertainty alone never creates a user-facing task -- surfacing a
choice requires the affirmative `HUMAN_CHOICE_ELIGIBLE` determination); a
Future UX contract-only section (compact A/B comparison, play/choose A or B,
must NOT expose confidence scores/semantic ids/resolver terminology); the
SWAP vs. `HUMAN_CHOICE_ELIGIBLE` distinction (analogous shape, distinct
trigger/scope/frequency/membership-model -- does not reintroduce SWAP, D-019
unaffected); the `HUMAN_DECISION_RATE` product metric (defined, no hard
threshold set); a QA_ENGINE contract update (QA must challenge every future
`HUMAN_CHOICE_ELIGIBLE` result against a 5-item checklist before accepting
it); and Future Engine States (`AUTO_RESOLVED`, `AUTO_RESOLVED_LOW_MARGIN`,
`REVIEW_REQUIRED_SEMANTIC`, `HUMAN_CHOICE_ELIGIBLE` -- semantics only, no
code emits any of these values today).

**No engine file changed.** No code was modified, no test was run, no UI was
implemented, no Modal or RunPod infrastructure was touched. The D-062.1
resolver fix (`CRITICAL_COVERAGE_DOMINANCE` implementation) is explicitly
deferred to its own future directive, requiring targeted tests and a
`RUN QA_ENGINE` pass before any paid canary, per this document's own Section
11 checklist and D-062's operating model.

### Final report (verbatim, as delivered)
CUTSELL EDITORIAL RESOLUTION & HUMAN ESCALATION CONTRACT ADDED

AUTOMATIC EDITOR DOCTRINE:
REGISTERED

SEMANTIC DOMINANCE BEFORE PERFORMANCE:
REGISTERED

CRITICAL_COVERAGE_DOMINANCE:
REGISTERED

AUTOMATIC RESOLUTION HIERARCHY (16 LAYERS):
REGISTERED

CONTEXTUAL_DELIVERY_FIT:
REGISTERED

ESCALATION CLASSES (REVIEW_REQUIRED_SEMANTIC / AUTO_RESOLVED_LOW_MARGIN / HUMAN_CHOICE_ELIGIBLE):
REGISTERED

HUMAN CHOICE CONTRACT:
REGISTERED

FUTURE UX CONTRACT (A/B COMPARISON, NO INTERNAL TERMINOLOGY EXPOSED):
REGISTERED

SWAP VS HUMAN_CHOICE DISTINCTION:
REGISTERED

HUMAN_DECISION_RATE METRIC:
REGISTERED (NO HARD THRESHOLD)

QA_ENGINE CONTRACT UPDATE (HUMAN_CHOICE_ELIGIBLE CHECKLIST):
REGISTERED

FUTURE ENGINE STATES:
REGISTERED (SEMANTICS ONLY, NOT IMPLEMENTED)

D-062.1 CLASSIFICATION:
AUTO-RESOLVE -- NOT HUMAN CHOICE

ENGINE CODE CHANGED:
NO

UI IMPLEMENTED:
NO

CURRENT DEVELOPMENT BLOCKED:
NO

NEXT TECHNICAL ACTION:
Implement critical-coverage-dominance tie-break for the D-062.1 resolver case,
then RUN QA_ENGINE before any paid canary.

Then STOP.

No engine changes.
No Modal.
No RunPod.

## D-063 -- CRITICAL_COVERAGE_DOMINANCE tie-break (engine implementation)

Implements the canonical `CRITICAL_COVERAGE_DOMINANCE` rule (see
`docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md`
Section 3, D-062.2) inside `claim_coverage_best_take.py` -- BestTakeResolver,
the authoritative pre-Freeze retry-resolution layer already active in this
module (D-038/D-048). Target failure (generalized from D-062.1): a retry
family already has 2+ members selected (a prior stage's conflicting
semantic-winner evidence left both live); this module previously skipped
that case entirely ("not this module's job"), falling straight through to
StoryValidator/FinalEditReviewer's existing DUPLICATE_IDEA+UNRESOLVED_RETRY
block -- this codebase's production analog of the canonical contract's
`REVIEW_REQUIRED_SEMANTIC` escalation.

**Fix**: before that fallback, check whether exactly one of the currently-
selected candidates strictly dominates every other on CRITICAL-claim
coverage (`_critical_coverage_dominant_candidate`) -- covers everything
every other candidate covers, plus at least one more; is not itself a
proven-incomplete/failed delivery (`complete_idea is False`); does not
factually contradict any other candidate (reuses
`contradiction_signal.any_pair_contradicts`, the SAME safety gate
`canonical_edit_plan.py`'s own composite-safety check already uses). If
found, the dominant candidate becomes the family's sole winner (KEEP/DISCARD
only, D-019 -- others move to discard, never SWAP). If not -- identical
coverage, genuinely disjoint unique critical claims, a contradiction, or the
would-be winner is itself unusable -- the family is left exactly as before,
still falling through to the existing block. No new heuristic, no new
provider: reuses `claim_coverage`/`resolve_ambiguous_coverage`/
`ClaimEquivalenceArbiter` (already used elsewhere in this same module) and
`any_pair_contradicts` verbatim. A CONTEXTUAL-only extra claim never
triggers dominance -- coverage is compared over CRITICAL claims exclusively,
by construction (no special-case code needed).

**Precedence preserved** (module docstring's existing safety > completeness
> consistency > delivery > richness order, per D-050C1/D-063's own Section
4): this check runs strictly BEFORE any lower-precedence signal, and never
overrides a contradiction or an incomplete/failed delivery.

**Scope discipline**: no change to grouping, ASR, Human Gold, Freeze policy,
StoryValidator, Boundary, Render/QC, Human Choice UI, or SWAP behavior. No
Human Choice escalation classes (`HUMAN_CHOICE_ELIGIBLE` etc.) implemented --
semantics-only per D-062.2, not yet emitted anywhere. `POST_RESOLVER_
SEMANTIC_MUTATORS` remains 0: this fix lives inside `apply_claim_coverage_
best_take`, which runs strictly before both `apply_final_story_coherence_
validation` passes and before `freeze_selection_contract` (confirmed via
`universal_clean_cut.py`'s own call order) -- a pre-Freeze resolver-layer
decision, not a downstream override.

**Pre-existing test updated, not a regression**: `test_cutsell_clean_cut_
core_evaluation_suite.py::test_percentage_must_survive_blocks_freeze`
(CleanCutBench fixture) exercised exactly the D-062.1 shape -- two
candidates tied on the deterministic ranker's own score, left ambiguous
(both selected) by `apply_deterministic_best_take_authority`. Before D-063,
that ambiguity reached StoryValidator's own residual-family fallback (no
claim-coverage awareness), which could pick either side; the fixture only
verified the loss was *caught* (freeze blocked) when the wrong side won. With
D-063, `CRITICAL_COVERAGE_DOMINANCE` resolves it correctly before
StoryValidator ever sees it -- the candidate carrying the critical fact
survives directly, nothing is lost, and Freeze is correctly NOT blocked. Test
renamed to `test_percentage_must_survive_via_critical_coverage_dominance` and
updated to assert the stronger (not weaker) guarantee. CleanCutBench stays
54/54 LEGACY and 54/54 AUTHORITATIVE.

**Tests**: 9 new/updated in `test_cutsell_claim_coverage_best_take.py`
covering all 11 directive test-matrix items (two pairs share one fixture
each: "A covers B + extra critical claim -> A wins" doubles as "conflicting
semantic-winner labels + strict dominance -> dominant wins"; "both have
different unique critical claims -> no dominance" doubles as "conflicting
labels + no dominance -> REVIEW_REQUIRED retained"), plus the D-062.1 generic
regression fixture (no Video00-specific text/ids). Additional adversarial
N=3-candidate dominance/no-dominance cases verified directly (not part of
the shipped suite, confirmed via direct execution during QA_ENGINE review).

**Offline qualification**: compileall clean. `tests/test_cutsell_claim_
coverage_best_take.py`: 24 passed (0 pre-existing regressions, 9 new/updated).
CleanCutBench LEGACY (`test_cutsell_clean_cut_core_evaluation_suite.py`):
54/54. CleanCutBench AUTHORITATIVE (`test_cutsell_d050c1_5_full_cleancutbench_
parity.py`): 54/54 (1 passed, full-suite aggregate). D-050C2 cutover sweep
(`test_cutsell_d050c2_authority_cutover.py`): 19/19. All D-050 through D-062.2
test files: green. Full `tests/` (minus `test_semantic_stitch.py`): 2293
passed, 13 subtests passed, only the same 2 pre-existing/unrelated failures
already on record since D-056.1 (confirmed identical on the pre-D-063
baseline via direct `git stash` comparison).

**QA_ENGINE**: self-conducted independent adversarial review (explicitly
disclosed as such -- no dedicated QA_ENGINE tool/script exists in this repo,
same posture as D-061's QA_ENGINE pass). Directly executed N=3-candidate
dominance and no-dominance scenarios against the real function (not just the
shipped 2-candidate fixtures) to confirm generalization beyond pairs.
Verified via `git stash` that both full-suite failures are byte-identical to
the pre-D-063 baseline. Confirmed call-order placement (pre-Freeze,
pre-StoryValidator) directly from `universal_clean_cut.py`'s own import/call
sequence rather than assuming it from the module docstring. Challenge
checklist: (1) dominance suppressing genuinely distinct facts -- verified
NOT possible, disjoint-critical-claim cases (2-way and 3-way) correctly
leave the family untouched; (2) contextual claims accidentally treated as
critical -- verified NOT possible, coverage is compared over
`_group_critical_claims`'s own CRITICAL-only output, unchanged; (3)
contradiction safety -- verified intact via numeric and negation
contradiction fixtures, `any_pair_contradicts` reused unmodified; (4)
incomplete/failed rich takes wrongly beating usable complete takes --
verified blocked via the explicit `complete_idea is False` gate; (5)
delivery still decides when coverage is equal -- verified: identical-
coverage case is left fully untouched (no forced pick), consistent with
directive Section 5's explicit "retain current behavior"; (6) true ties stay
blocked -- verified via disjoint-unique-claims fixtures (2-way and 3-way),
falls through unchanged to DUPLICATE_IDEA+UNRESOLVED_RETRY. No P0/P1/P2
found; the CleanCutBench test-expectation update above is a required,
disclosed consequence of the directive's own target behavior, not a defect.

### Final report (verbatim, as delivered)
D-063 COMPLETE

CRITICAL_COVERAGE_DOMINANCE:
PASS

D-062.1 GENERIC CASE:
PASS

DELIVERY CANNOT OVERRIDE CRITICAL DOMINANCE:
PASS

TRUE AMBIGUITY STILL REVIEW_REQUIRED:
PASS

CONTRADICTION SAFETY:
PASS

POST_RESOLVER SEMANTIC MUTATORS:
0/0

LEGACY:
54/54

AUTHORITATIVE:
54/54

FULL TEST COUNT:
tests/test_cutsell_claim_coverage_best_take.py 24 passed (9 new/updated);
full tests/ (minus test_semantic_stitch.py) 2293 passed, 13 subtests passed,
only the same 2 pre-existing/unrelated failures already on record (confirmed
identical on the pre-D-063 baseline).

QA_ENGINE VERDICT:
PASS

P0:
0
P1:
0
P2:
0
P3:
0

READY FOR ONE VIDEO00 CANARY:
YES

Then STOP.

Do not launch Modal.

## D-063.1 -- D-063 Video00 canary (run 33851050636, report only)

Authorized single Modal Video00 canary against D-063's build (head `c23a9f4`,
qualified config unchanged). CRITICAL_COVERAGE_DOMINANCE was NOT_EXERCISED
live: the one already-ambiguous (2+-selected) family present this run
(`tg_c065fefe93a2dea845`) genuinely carried two DIFFERENT CRITICAL-classified
claims (a diagnosis claim on one side, a NEGATION claim on the other) under
the current claim extractor -- no strict-superset relationship existed, so
D-063 correctly declined to force a pick (TRUE AMBIGUITY SAFETY: PASS).
D-061 safety fully retained live (claim_equivalence_arbiter reachable and
successfully confirmed one ambiguous paraphrase elsewhere in the same run;
a separate hard negation mismatch, unrelated to this family, correctly still
blocked via `lost_critical_claims`). Human Gold 16/18 (same 2 pre-existing,
unrelated failures as every prior canary). Freeze BLOCKED --
root blocker classified B (true semantic ambiguity under the current claim
classifier, not a dominance failure, not content loss, not a validator false
positive). D-063 LIVE OBJECTIVE: PARTIAL (safety proven live; the positive
auto-resolve case was not exercised because no qualifying dominance case
existed in this run's fresh ASR output). No code modified, no second RAW.

## D-064 -- cross-claim-type semantic dedup forensic (report only)

Forensic trace of the D-063.1 blocker's clip pair. Extracted every claim
from both candidates with full field detail and found: candidate A's
reflective clause and candidate B's NEGATION clause are a genuine
SEMANTIC_PARAPHRASE of the same before->after hindsight-realization
proposition ("in hindsight, previously-unremarkable signs were actually
meaningful") -- B's negation is CONTRASTIVE, not standalone-factual; deleting
B while keeping A loses no audience-facing information. Root cause:
claim-type mismatch (DIAGNOSIS_IDENTIFICATION vs NEGATION) is a contributor,
not the sole cause -- the deeper gap is architectural: D-063's dominance
check compares a claim against a candidate's WHOLE TEXT (coverage), never
claim-against-claim (equivalence), so the paraphrase relationship was never
even asked about. Counterfactual: after correct claim-vs-claim dedup, A
would strictly dominate B (D-063's own rule proven still correct). Product
decision: NOT human-choice-eligible -- the same D-062.1 pattern, a
resolver/claim-canonicalization gap, not a genuine tie. General missing
contract identified: claim-vs-claim equivalence distinguishing rhetorical/
contrastive hindsight negation from standalone factual negation, preserving
number/entity/diagnosis/causal-direction/temporal-order safety invariants.
No code modified.

## D-065 -- negation semantic role design (design only, no code)

Full design contract for the D-064 gap. Two-class semantic-role model:
FACTUAL_NEGATION (negation changes the truth conditions of protected
content -- diagnosis/number/dose/currency/entity/causal-direction/quantified-
outcome -- always independent, never merged) vs CONTRASTIVE_HINDSIGHT_NEGATION
(negation is the before-half of an explicit before->after realization
structure over non-protected content only -- eligible for narrow,
arbiter-confirmed merge). Canonical claim unit: retain raw claims, add
additive semantic-role metadata (never a new claim_type, never a
replacement). Claim-vs-claim equivalence: hard-gated (numbers/entities/
diagnosis/causal-direction/temporal-direction/protected-content, all
categorical) before any arbiter call; reuses the EXISTING
`ClaimEquivalenceArbiter` protocol, no new provider/model. Owner subsystem:
`semantic_claims.py` (role classification) + `claim_coverage_best_take.py`'s
existing critical-claims dedup step (equivalence merging) -- no new
authority layer. Full adversarial matrix (13 cases) and sales/UGC
generalization matrix (beauty/wellness/consumer/storytelling, plus the
mandatory "It did not reduce bloating" vs "It reduced bloating" never-merge
counter-example) specified. D-064 confirmed to auto-resolve after this
contract; true negation safety confirmed preservable; no new authority
layer required. READY FOR IMPLEMENTATION DIRECTIVE: YES. No code modified.

## D-066 -- negation semantic role implementation (Phase 1, engine change)

Implements D-065's design conservatively across exactly two files:

**`semantic_claims.py`**: additive `Claim.negation_role` field
(`FACTUAL_NEGATION` / `CONTRASTIVE_HINDSIGHT_NEGATION` / `""` not-applicable
default) -- `claim_type` itself untouched. `_classify_negation_role` requires,
conjunctively, for a NEGATION-typed clause: (1) a contrast marker in the
original (pre-split) sentence; (2) the negation clause's own predicate uses a
belief/perception/seeming verb (`_BELIEF_PERCEPTION_MARKERS`, new) -- the
primary safety differentiator, since no factual negation ("did not reduce
bloating", "was not the cream", "did not cost $49", "did not cause the
breakout") ever uses one; (3) a LATER, non-degenerate, not-itself-negated
clause exists in the same sentence (an incomplete "I didn't think..." or a
later clause that is itself negated never qualifies); (4) neither clause
carries any protected-content marker or digit evidence
(`_negation_role_hard_exclusion`, reusing the exact marker vocabulary
`claim_coverage_best_take._is_low_information_incidental` already uses).
An explicit temporal/retrospective marker on the later clause is treated as
strong supporting evidence, not a hard requirement (D-040's own >=2-content-
token clause-split floor already screens out degenerate "but it helped"-style
completions in practice; requiring an explicit marker in addition would have
silently excluded real, plain-outcome sales/UGC phrasing).

**`claim_coverage_best_take.py`**: `_find_hindsight_alignment` searches,
for every CONTRASTIVE_HINDSIGHT_NEGATION-eligible CRITICAL claim in an
already-ambiguous (D-063 `>=2`-selected) family, for a claim-vs-claim (never
claim-vs-whole-candidate-text) equivalence among the family's own
ACTION_EVENT/STATE_RESULT claims -- deterministic hard gates
(`_hindsight_alignment_hard_gates_pass`: candidate claim_type excluded from
DIAGNOSIS_IDENTIFICATION/CORRECTION/MEASUREMENT_QUANTITY, no digit evidence
on either side, no protected marker on the candidate) run BEFORE a
negation-agnostic content-token overlap floor, itself before any arbiter
call -- reuses the EXISTING `ClaimEquivalenceArbiter` protocol verbatim, no
new provider/model, no new arbiter class. A confirmed alignment merges the
two claims for coverage-unit purposes only (`_covered_claim_ids`'s additive
`hindsight_alignments` parameter, default None, byte-identical for every
pre-D-066 call site) -- raw claims, text, and provenance are never rewritten.
**D-063's own dominance rule (`_critical_coverage_dominant_candidate`) is
completely unchanged**: only the coverage sets it consumes are corrected
upstream. Diagnostics (`claim_coverage_best_take.hindsight_alignments`,
never exposed to any UI) record, per eligible claim: role, aligned claim id/
text (or none), arbiter invocation/verdict, coverage-unit relation, reason.

**Verified live safety proofs** (direct adversarial execution, not just
shipped tests): an arbiter that would confirm ANY pairing is proven to
never even be asked about a protected (diagnosis) candidate
(`test_arbiter_never_reaches_protected_candidate_even_if_it_would_confirm`);
the mandatory "It did not reduce bloating" vs "It reduced bloating"
counter-example never merges even with an always-confirming arbiter (the
negation clause never becomes eligible in the first place -- no belief/
perception verb -- and `any_pair_contradicts` independently blocks it too);
the D-061 QA_ENGINE-flagged gastritis/ulcer diagnosis-substitution shape
stays unmerged; an incomplete/failed-delivery candidate is never preferred
even when alignment succeeds (D-063's own completeness gate, untested by
this directive, confirmed still intact); a genuine contradiction elsewhere
in the two candidates' text still blocks dominance even after a successful
alignment; an arbiter exception fails closed with no crash. `POST_RESOLVER_
SEMANTIC_MUTATORS` stays 0 -- both changed files are pre-Freeze, and no new
call site was added anywhere else.

**Tests**: 20 new in `test_cutsell_d066_negation_semantic_role.py`
(role-classification adversarial suite: true factual negation, diagnosis,
number, price, entity, causal-direction, incomplete sentence, double
negation, sarcasm-like phrasing, temporal reversal, both-sides-negated,
no-contrast-marker -- all FACTUAL_NEGATION; positive paraphrase suite +
sales/UGC positive shapes -- CONTRASTIVE_HINDSIGHT_NEGATION; one disclosed
Phase-1 conservative boundary documented, not hidden). 12 new in
`test_cutsell_d066_hindsight_alignment.py` (hard-gate unit tests, the
protected-candidate arbiter-never-asked proof, the D-064 generic auto-
resolve chain end to end under confirming/declining/absent arbiters, the
mandatory bloating and gastritis/ulcer never-merge counter-examples, beauty
and consumer-product positive sales/UGC shapes). No Video00-specific text or
ids in any test or production code -- the D-064 generic-chain fixture
reproduces the general pattern only.

**Offline qualification**: compileall clean. All D-050 through D-066 test
files (including `test_cutsell_claim_coverage_best_take.py` and
`test_cutsell_clean_cut_core_evaluation_suite.py`): 433 passed. CleanCutBench
LEGACY 54/54, AUTHORITATIVE 54/54 (both within that same 433). Full `tests/`
(minus `test_semantic_stitch.py`): 2325 passed, 13 subtests passed, only the
same 2 pre-existing/unrelated failures already on record since D-056.1.

**QA_ENGINE**: self-conducted independent adversarial review (same
disclosed posture as D-061/D-063's QA_ENGINE passes). Directly re-executed,
outside the shipped test suite, every mandatory safety question from the
directive's Section 14: number/entity/diagnosis/causal/temporal/true-
negation safety and ASR-uncertainty fail-closed behavior, all PASS,
including the specific dangerous-failure-mode probe ("can a genuine
contradiction be silently merged because it resembles rhetorical
negation?") -- answered NO, both via the eligibility gate itself (a factual
negation never acquires a belief/perception verb) and independently via
`any_pair_contradicts`'s own unchanged safety net. 0 new P0/P1/P2/P3.

### Final report (verbatim, as delivered)
D-066 COMPLETE

NEGATION SEMANTIC ROLE:
PASS

FACTUAL_NEGATION SAFETY:
PASS

CONTRASTIVE_HINDSIGHT:
PASS

CLAIM-VS-CLAIM EQUIVALENCE:
PASS

D-064 GENERIC CASE:
PASS

D-063 DOMINANCE AFTER DEDUP:
PASS

NUMBER SAFETY:
PASS

ENTITY SAFETY:
PASS

DIAGNOSIS SAFETY:
PASS

CAUSAL SAFETY:
PASS

TEMPORAL SAFETY:
PASS

ASR UNCERTAINTY:
FAIL_CLOSED

SALES/UGC GENERALIZATION:
PASS

POST_RESOLVER SEMANTIC MUTATORS:
0/0

LEGACY:
54/54

AUTHORITATIVE:
54/54

FULL TEST COUNT:
D-050-D-066 test files 433 passed; full tests/ (minus test_semantic_stitch.py)
2325 passed, 13 subtests passed, only the same 2 pre-existing/unrelated
failures already on record.

QA_ENGINE VERDICT:
PASS

P0:
0
P1:
0
P2:
0
P3:
0

READY FOR ONE VIDEO00 CANARY:
YES

Then STOP.

Do not launch Modal.

## D-067 -- authoritative orphan root-cause forensic (report only, no code, no RAW)

D-066 canary run 33882792100 reached Freeze BLOCKED via a separate, previously-
unexamined authority: `realization_resolver_authority` (Unified Resolver)
found 3 unconfirmed orphan realizations, all `discard_reason:
"high_confidence_semantic"` / `decision_reason:
"hybrid_editorial_semantic_delete_with_no_verified_replacement_never_silently_
confirmed"`. Traced and classified all 3 (category F, IDENTITY/PROVENANCE BUG
in substance, two with strong independent coverage evidence never wired into
`replacement_verified`); D-050D1's own three-way orphan contract itself found
PASS (executed exactly as documented -- the gap is one layer upstream). Shared
root cause: `semantic_ledger.py` reads only `hybrid_editorial`'s own
`later_retry_replacement_id` field, never `later_retry_semantic_overlap` alone
nor `hybrid_story_coverage_guard`'s independent coverage verdicts. Orphans
A/B/C: NOT_PROVEN safe (do not infer from overlap alone). Freeze counterfactual:
YES, this is the only structural blocker between the engine and Render if
correctly classified.

## D-068 -- replacement provenance wiring investigation (report only, no code, no RAW)

Explained WHY `later_retry_semantic_overlap > 0` can coexist with
`later_retry_replacement_id == null`. Proved, by direct source reading, that
`_later_semantic_retry_replacement`'s own body makes this pairing
unreachable (best/best_overlap always co-assigned) -- disproving control-flow
and diagnostics-construction bugs. Could not locate the actual mechanism from
static reading alone; root cause: E (artifact/logging mismatch), reached by
elimination, not direct proof -- flagged as the one open item requiring the
raw persisted artifact, not the console-log transcript, to close.

## D-069 -- raw persisted artifact provenance audit (no engine code, no GPU, no RAW)

Recovered the run's own raw `result.json` via the existing, already-authorized,
zero-GPU `cutsell-video00-d044-forensic-extract.yml` workflow (S3 read only, no
Modal/RunPod). The "impossible" pair is present in the RAW artifact itself,
byte-identical across raw file / workflow extraction / console log -- D-068's
category E (log/artifact mismatch) is disproven. Root class re-opened as
UNKNOWN pending offline code-level reproduction. Orphans A/B/C remain
NOT_PROVEN.

## D-070 -- replacement provenance offline reproduction (no GPU, no RAW)

Reproduced the live anomaly offline by calling the real, unmodified
`_later_semantic_retry_replacement` with real recovered field values:
`(None, 1.0)` and `(None, 0.75)`, exact matches. Root cause found and proven
via `inspect.getsource`: `complete_retry_identity_guard.py`'s
`install_complete_retry_identity_guard()` monkeypatches
`hybrid_session_cleanup._later_semantic_retry_replacement` at
`cutsell_worker` package-import time. Its own additional sequence-identity
veto (`_sequence_identity < 0.52`) discards a candidate the base function
found, while returning the base function's own `overlap` unchanged --
producing the "impossible" pair. Not a bug: a deliberate, documented,
previously-unexamined safety layer. RED regression test (scratchpad only,
not committed) proved the mechanism against the real code. Orphans remain
NOT_PROVEN (this strengthens, not weakens, that caution).

## D-071 -- complete retry identity guard modernization design (design/forensic only, no code, no RAW)

Confirmed the guard's founding failure class (same-topic narrative
continuation mistaken for a retake, proven by its own test suite) and
designed, without implementing, a second, PATH-B semantic-preservation
certification path as a composition of already-existing primitives
(claim coverage, D-063 critical-coverage dominance, D-066 negation-role
classification, contradiction detection) -- never new semantic machinery.
Found the required evidence does not exist at the guard's own pipeline
stage; recommended owner is the existing Unified Resolver
(`realization_resolver.py`), never a new authority. Orphans A/B/C: all
NOT_PROVEN under PATH B on existing evidence (A strongest, B lacks a valid
PATH-B-shaped candidate at all, C has an unresolved entity-level lexical
divergence). Observability fix recommended regardless of PATH B's fate.

## D-072 -- complete retry identity observability (engine change, additive only)

Implemented ONLY the D-071 observability recommendation -- no PATH B, no
threshold change, no behavior change. Added, additively:

- `cutsell_worker/hybrid_session_cleanup.py`: `_scan_replacement_candidates`
  (a pure, side-effect-free re-scan mirroring `_later_semantic_retry_
  replacement`'s own gate cascade verbatim, reporting best-overlap-seen and
  eligible-candidate-count regardless of whether any candidate cleared the
  floor -- consulted only by the new diagnostics, never by any decision);
  wired the per-decision loop to read-and-clear the guard's new diagnostic
  side channel and add 5 new additive keys to each `hybrid_editorial_chunks`
  decision dict (`replacement_candidate_clip_id_before_guard`,
  `sequence_identity`, `sequence_identity_threshold`,
  `lexical_identity_passed`, `replacement_rejection_reason`).
- `cutsell_worker/complete_retry_identity_guard.py`: bounded rejection-reason
  vocabulary (`NO_CANDIDATE`, `SEMANTIC_OVERLAP_BELOW_THRESHOLD`,
  `NUMBER_PRESERVATION_FAILED`, `SEQUENCE_IDENTITY_BELOW_THRESHOLD`,
  `LEXICAL_REPLACEMENT_VERIFIED`, `INCOMPLETE_RETRY_LOOSER_MATCH`,
  `NOT_APPLICABLE`); a frozen `ReplacementGuardDiagnostic` dataclass; a
  ContextVar side channel (same established pattern as
  `hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN`), set once per
  `protected()` call immediately before each existing `return` (no return
  VALUE changed anywhere), read-and-cleared once by the consumer;
  `_diagnose_no_replacement_reason` distinguishes NO_CANDIDATE /
  SEMANTIC_OVERLAP_BELOW_THRESHOLD / NUMBER_PRESERVATION_FAILED via the new
  scan helper, purely for reporting.

Verified: all 4 pre-existing guard tests pass byte-for-byte unchanged; 11 new
targeted tests (`test_cutsell_d072_complete_retry_observability.py`) cover
every reason, ContextVar clear-on-read, cross-clip non-leakage, and
cross-thread isolation; 2 new end-to-end integration tests in
`test_cutsell_hybrid_session_cleanup.py` prove the fields reach
`apply_hybrid_session_cleanup`'s own real diagnostics, reproducing the exact
D-070 shape and the NOT_APPLICABLE default for non-"failed" decisions.
Repo-wide grep confirms the 4 new field names appear in exactly the 2
producer files and nowhere else (no selection/resolver/Freeze/orphan
consumer). Only one call site of `_later_semantic_retry_replacement` exists
in the whole codebase (`hybrid_session_cleanup.py`'s own loop), so no other
caller can leave an unconsumed, later-misattributed diagnostic.

Final report (verbatim, as delivered):

D-072 COMPLETE

COMPLETE RETRY OBSERVABILITY:
PASS

SEQUENCE IDENTITY SURFACED:
YES

PRE-GUARD CANDIDATE SURFACED:
YES

REJECTION REASON SURFACED:
YES

DECISION BEHAVIOR CHANGED:
NO

ORPHAN POLICY CHANGED:
NO

FREEZE POLICY CHANGED:
NO

CONTEXT ISOLATION:
PASS

LEGACY:
54/54

AUTHORITATIVE:
54/54

FULL TEST COUNT:
tests/test_cutsell_*.py: 1714 passed, 0 failed. D-050-D-072 targeted sweep
(31 files): 462 passed. CleanCutBench: 54/54 LEGACY, 54/54 AUTHORITATIVE.

QA_ENGINE VERDICT:
PASS -- observability-only confirmed: decisions unchanged (byte-for-byte on
all founding safety tests plus new explicit assertions), orphan/Freeze code
paths untouched, no new semantic authority (the new scan helper is provably
side-effect-free and consulted only by diagnostics), no diagnostic state
leakage (ContextVar is per-context, read-once-cleared, single call site).

P0:
0
P1:
0
P2:
0
P3:
0

PATH B IMPLEMENTED:
NO

READY FOR PATH B DESIGN/IMPLEMENTATION:
YES (observability groundwork in place; PATH B itself remains a future,
separately-authorized directive per D-071)

Then STOP.

No Modal.
No RAW.

## D-073 -- semantic replacement certification / Unified Resolver PATH B implementation

Implemented PATH B: a second, additive, semantic replacement-certification
path inside the Unified Resolver's own existing orphan-resolution authority
(`resolve_orphan_realizations_shadow` in `realization_resolver.py`). PATH A
(the lexical guard in `complete_retry_identity_guard.py`/
`hybrid_session_cleanup.py`) is untouched -- tried first, unconditionally,
exactly as before; PATH B is attempted only when PATH A found nothing AND
the discard already reached `hybrid_editorial_chunks`'s own semantic delete
decision (never for `PRE_GROUP_REJECTED` discards).

**Candidate discovery.** A true orphan (discarded before grouping, so
`semantic_idea_id is None` by definition) has no formal idea/retry relation
to consult. PATH B reuses D-072's own `replacement_candidate_clip_id_before_
guard` diagnostic (already gated by real topical overlap) as its sole
legitimate candidate-discovery mechanism -- carried through the Ledger via a
new `DiscardRecord.pre_guard_candidate_clip_id` field (`semantic_ledger.py`,
additive, observation-only).

**Certification chain** (`_attempt_semantic_replacement_certification`,
fail-closed at every step, in order): pre-guard candidate exists -> maps to
an existing Ledger realization -> that realization is `state == "selected"`
(the module's only available proxy for "verified same relation" for a true
orphan) -> not proven incomplete (`complete_idea is not False`) ->
realization-level NUMBER hard gate (see below) -> D has >=1 extractable
claim (WHEN UNCERTAIN, KEEP) -> no contradiction signal
(`_detect_contradiction_signals`, reused verbatim) -> EVERY one of D's
claims (not just CRITICAL ones -- Section 6) has a verified preserving claim
on R's side (`_claim_preserved`).

**Claim-level preservation** (`_claim_preserved`) tries, in order:
`_claims_dedup_equivalent` (existing hard type/negation/digit gates, reused
verbatim -- gives NUMBER/NEGATION/entity-substitution safety by
construction); `_claim_content_subsumed` (new: a cross-TYPE deterministic
superset check -- D's content tokens subset of R's, D's digit values subset
of R's, negation polarity agrees -- needed because `classify_claim` can
promote a combined clause to a different `claim_type` the instant a number
appears, which would otherwise defeat the exact-type gate on a genuinely
safe superset); D-066's CONTRASTIVE_HINDSIGHT_NEGATION safe contract
(reimplemented for `CanonicalClaimRecord`, same imported constants/
functions, same protected-type exclusion, same bounded arbiter question).

**Direction safety (Sections 8/12).** CAUSE_EFFECT/TEMPORAL_RELATION claims
can NEVER be certified preserved, unconditionally, no arbiter escape hatch
(`_DIRECTION_SENSITIVE_CLAIM_TYPES`). Found during testing that
`classify_claim` only assigns CAUSE_EFFECT to a connector-split clause
("because", "due to", ...) -- a bare causal verb ("The medication caused
the rash.") classifies plain ACTION_EVENT, so type-alone detection missed
the directive's own required causal-reversal case. Added
`_claim_signals_direction_sensitive`: also blocks on a small, generic,
bilingual causal-verb marker set this module owns (`_CAUSAL_VERB_MARKERS`)
plus the existing `_CAUSE_EFFECT_MARKERS`/`_TEMPORAL_MARKERS` connector
vocabulary (reused, not redefined) -- a deterministic, PATH-B-local text
signal, not a change to `classify_claim` itself.

**Number safety beyond claim identity.** `mint_canonical_claim_id`
(canonical_identity.py, pre-existing, unrelated D-050C purpose) groups
claims by `(claim_type, content_tokens)` only -- not exact text -- so two
claims differing ONLY by digit ("...3 centimeters..." vs "...5
centimeters...") can mint to the SAME `canonical_claim_id`, making D's and
R's claim the literal same Ledger object (trivially "self-preserved").
Added an independent realization-TEXT-level NUMBER hard gate
(`_realization_digit_values`, reading each realization's own raw `.text`,
untouched by claim-id collapsing) as an additional safety net PATH B owns
itself -- not a change to shared claim-minting infra (out of scope, wide
blast radius). Both this gate and `_claim_content_subsumed`'s own digit
check are SUBSET checks (every number D asserts must survive in R), not
exact-set equality -- R may carry additional numbers beyond D's own
(Section 3's safe-superset direction applies to numbers like any other
fact); an initial exact-equality draft was caught and fixed before
qualification, with a dedicated regression test locking in the subset
behavior.

**Verdict model (additive):** `REPLACEMENT_VERIFIED_SEMANTIC` (new PATH B
verdict) alongside unchanged `REPLACEMENT_VERIFIED_SAFE` (PATH A),
`REVIEW_REQUIRED`, `PRE_GROUP_REJECTED`. New `OrphanRealizationReview`
fields: `verification_method` (`"lexical"` | `"semantic"` | `""`),
`semantic_replacement_evidence` (Section 15's diagnostic fields:
`semantic_replacement_evaluated`, `candidate_replacement_realization_id`,
`same_idea_verified`, `critical_claims_preserved`,
`unique_required_content_preserved`, `hard_gate_results`,
`arbiter_invoked`, `semantic_replacement_verified`,
`semantic_replacement_reason`, `preserved_claim_ids`) -- diagnostic-only,
never a second authority.

**Arbiter.** `claim_equivalence_arbiter` threaded as an additive optional
keyword through `resolve_orphan_realizations_shadow` ->
`resolve_realizations_shadow` (existing param) and
`apply_authoritative_realization_resolution`; production wiring added ONLY
at `universal_clean_cut.py`'s `apply_authoritative_realization_resolution`
call site, deliberately NOT at the earlier `resolve_realizations_shadow`
call site (preserves existing grouped-idea dedup arbiter behavior
unchanged -- out of this directive's scope). Deterministic evidence always
tried first; the bounded arbiter only ever narrows the D-066 hindsight
ambiguous band; provider exception/malformed result fails closed
(pre-existing `_claims_dedup_equivalent`/hindsight-loop behavior, reused
verbatim).

**POST_RESOLVER_SEMANTIC_MUTATORS reconfirmed 0.** Traced
`universal_clean_cut.py`'s AUTHORITATIVE-mode path after
`apply_authoritative_realization_resolution`: only StoryValidator
(re-validates, can freeze-block, never re-selects) and the repair loop
(Boundary-only physical repair) run afterward -- nothing downstream flips
`selected`/`discarded` membership again.

Verified: 23 new targeted tests
(`test_cutsell_d073_semantic_replacement_certification.py`) cover the full
D-073 Section 12 directional matrix, Section 13 orphan A/B/C-like
regression, Section 14 historical-guard regression (founding
`complete_retry_identity_guard` tests re-run unchanged), arbiter
failure/decline, provenance-not-found/not-selected fail-closed cases, and
the safe-superset-number regression. All real end-to-end fixtures through
`build_semantic_ledger_shadow` (real `extract_claims`, real Ledger
reconstruction) -- no Video00-specific text/ids.

Final report (verbatim, as delivered):

D-073 COMPLETE

SEMANTIC REPLACEMENT PATH:
IMPLEMENTED (REPLACEMENT_VERIFIED_SEMANTIC, inside the existing Unified
Resolver orphan authority)

LEXICAL PATH UNCHANGED:
YES (PATH A tried first, unconditionally; founding guard tests pass
byte-for-byte)

DIRECTIONAL PRESERVATION:
PASS (superset MAY VERIFY; reverse direction NEVER VERIFIES)

UNIQUE FACT SAFETY:
PASS (every D claim, not just CRITICAL ones, must find a preserving R
claim)

SAME-TOPIC CONTINUATION SAFETY:
PASS (founding guard case re-verified at this layer: REVIEW_REQUIRED)

NUMBER SAFETY:
PASS (claim-level + independent realization-text-level hard gate; subset-
safe for legitimate additional numbers in R)

NEGATION SAFETY:
PASS (factual negation reversal: REVIEW_REQUIRED)

ENTITY/DIAGNOSIS SAFETY:
PASS (diagnosis substitution: REVIEW_REQUIRED even with an always-confirm
arbiter)

CAUSAL SAFETY:
PASS (unconditional block, no arbiter escape hatch, marker-based detection
added to cover the realistic bare-causal-verb shape `classify_claim` alone
misses)

TEMPORAL SAFETY:
PASS (unconditional block; reversal case also independently fails on
content mismatch)

ORPHAN A-LIKE/B-LIKE/C-LIKE:
A-like: REPLACEMENT_VERIFIED_SEMANTIC. B-like (no pre-guard candidate):
REVIEW_REQUIRED. C-like (truncated/no resolvable equivalence):
REVIEW_REQUIRED.

POST_RESOLVER SEMANTIC MUTATORS:
0 (reconfirmed by tracing the AUTHORITATIVE-mode path)

LEGACY:
54/54

AUTHORITATIVE:
54/54

FULL TEST COUNT:
tests/test_cutsell_*.py: 1737 passed, 0 failed (1714 D-072 baseline + 23
new D-073 tests). D-050-D-073 targeted sweep (32 files, including founding
`complete_retry_identity_guard`): 393 passed. CleanCutBench: 54/54 LEGACY,
54/54 AUTHORITATIVE.

QA_ENGINE VERDICT:
PASS -- self-administered adversarial review attacking numbers, negation,
diagnosis/entity substitution, causal/temporal direction, unique facts,
partial retries, same-topic-different-event, ASR uncertainty, arbiter
failure/decline, and provenance validity found two real defects during
implementation (claim-type-mismatch defeating superset preservation;
causal-verb reversal undetected by type-alone gating; exact-set-equality
number check over-blocking safe supersets) -- all three fixed and locked in
by dedicated regression tests before this verdict. Known residual
limitation: causal/temporal direction detection is marker/connector-based
(deterministic, reusing existing vocabulary), not a full parse -- a causal
or temporal reversal using language outside the marker vocabulary would not
be flagged as direction-sensitive and would fall through to ordinary
claim-preservation comparison. This is a disclosed, not hidden, gap; no
false-certification was found for any tested causal/temporal shape, and
building a general causal/temporal parser is out of D-073's scope (Section
9: no new model/provider).

P0:
0
P1:
0
P2:
1 (disclosed causal/temporal marker-coverage limitation above -- monitor,
do not silently expand scope to fix without a separate directive)
P3:
0

READY FOR ONE VIDEO00 CANARY:
Offline-qualified; awaiting explicit user authorization before any Modal
canary (per D-073's own "Do not launch Modal" instruction).

Then STOP.
Do not launch Modal.

## D-073.1 -- same-idea proxy safety audit (offline only, one defect found and fixed)

Audited D-073's `candidate.state == "selected"` proxy for "verified same
semantic idea/retry relation" (the only signal available for a TRUE orphan,
which has no `semantic_idea_id` by definition).

**Candidate origin (Section 1).** `candidate_clip_id = discard.pre_guard_
candidate_clip_id` is the SOLE source (`realization_resolver.py` line
~1134) -- no arbitrary search over selected realizations exists anywhere in
PATH B. Traced upstream into `complete_retry_identity_guard.py`'s
`protected()` wrapper: for a COMPLETE discarded realization, the pre-guard
candidate already passed same-source-asset, later-start-within-24s,
candidate.complete_idea==True, label confidence>=0.68, a real NUMBER-
PRESERVATION filter (every number in D's raw text present in the
candidate's), and a real topical-overlap floor (`_semantic_overlap`
containment coefficient) >=0.64 -- it reaches PATH B only because it
ADDITIONALLY failed the sequence-identity check (SequenceMatcher
ratio<0.52) that would otherwise have made it PATH A's own certified
replacement. For an INCOMPLETE discarded realization, the discovery step is
looser (`INCOMPLETE_RETRY_LOOSER_MATCH`): no number-preservation filter, a
lower 0.50 overlap floor -- a real, disclosed discovery-time asymmetry, but
PATH B's own downstream claim-level checks (number/negation/content
subsumption) independently re-verify preservation regardless of which
discovery path found the candidate, so this asymmetry affects only which
candidate is offered for verification, never whether a false one is
certified.

**What "selected" proves (Section 2).** Only that the candidate realization
survives into the final kept timeline -- never semantic equivalence,
direction, entity identity, or that R's own net assertion matches D's. It
functions purely as a necessary precondition; all actual safety is carried
by the separate, independent directional claim-preservation chain (D-073
Sections 3-8), not by this proxy itself.

**Adversarial testing (Sections 3-6).** Ran the exact founding
`complete_retry_identity_guard` case, the full same-topic-different-event
matrix (4 cases), the valid-superset matrix (2 cases), and an always-YES
adversarial arbiter against both the founding case and the same-topic
matrix -- all rejected correctly, and the arbiter was never even invoked
(deterministic content/type mismatches reject before any arbiter-reachable
branch). One genuine defect was found and proven via a constructed
adversarial case NOT in the directive's own matrix: R's matching claim
carrying REPORTED/ATTRIBUTED speech ("Some customers said it did not work
for them...") could satisfy preservation for a D claim asserting the
identical words directly and unattributed ("It did not work for me.") --
even though R's own net assertion, in a contrastive-rebuttal frame
("...but it worked great for me"), is the OPPOSITE of D's. Every
deterministic content/negation/digit gate passed; the claim-preservation
layer had no concept of speech attribution.

**Fix** (`realization_resolver.py`): `_REPORTED_ATTRIBUTION_MARKERS` (a
small, generic, bilingual reporting-verb/attribution marker set this
module owns, same established pattern as `_CAUSAL_VERB_MARKERS`) and
`_preservation_blocked_by_attribution_asymmetry` -- blocks the ASYMMETRIC
case only (R's matching claim carries attribution language D's own claim
does not); a D claim that itself already carries the same attribution
framing may still match a same-attribution R claim normally (verified by a
dedicated symmetric-attribution sanity test). Applied once, up front, in
`_claim_preserved`, filtering the candidate pool before all three matching
strategies -- not a redesign, not a new authority, not a lowered gate.

**Provenance requirement (Section 7).** Based on code and 35 total
adversarial/positive tests exercised across D-073 and this audit: an
actual `semantic_idea_id`/`retry_family_id` relationship is NOT required
for safety, given the full directional all-claim preservation contract is
applied on top. Every attempted false certification was rejected by the
claim-preservation layer, independent of the selected-state proxy; the one
defect found (reported-attribution asymmetry) would NOT have been caught
by requiring a same-idea id either, since it lives entirely in the
claim-preservation layer, not in the idea-relation layer. Conclusion:
the selected-state proxy is a correctly-scoped necessary precondition, and
requiring an additional same-idea id would add complexity without closing
any gap the preservation chain doesn't already close.

Verified: 12 new targeted tests
(`test_cutsell_d073_1_same_idea_proxy_audit.py`) lock in the founding case,
the full same-topic-different-event matrix, the valid-superset matrix, the
always-YES arbiter attack, the discovered reported-attribution defect
(both broken-before and fixed-after), and the symmetric-attribution sanity
case. 0 regressions: tests/test_cutsell_*.py 1749 passed (1737 D-073
baseline + 12 new). CleanCutBench 54/54 LEGACY, 54/54 AUTHORITATIVE.
Founding `complete_retry_identity_guard` tests unaffected.

Final report (verbatim, as delivered):

D-073.1 COMPLETE

SELECTED-STATE PROXY:
SUFFICIENT (as a necessary precondition; all safety is carried by the
directional claim-preservation chain, not by this proxy)

PRE-GUARD CANDIDATE RELATION:
Sole origin is `discard.pre_guard_candidate_clip_id` -- no arbitrary
search. For a complete D: already passed same-source-asset, later-start-
within-24s, candidate-completeness, label-confidence, NUMBER-PRESERVATION,
and >=0.64 topical overlap, failing only sequence-identity. For an
incomplete D: looser (no number-preservation, 0.50 overlap floor) --
disclosed, but re-verified independently by PATH B's own claim-level gates
regardless.

FOUNDING CONTINUATION CASE:
REVIEW_REQUIRED (content-token subset check rejects: R's clause never
contains "she" -- D's own subject/agent is not literally present in R)

SAME-TOPIC DIFFERENT-EVENT SAFETY:
PASS (all 4 matrix cases: REVIEW_REQUIRED)

VALID SUPERSET REPLACEMENT:
PASS (both matrix cases: REPLACEMENT_VERIFIED_SEMANTIC)

ALWAYS-YES ARBITER SAFETY:
PASS (arbiter never invoked for any same-topic-different-event or founding
case -- deterministic gates reject first)

ACTUAL SAME-IDEA ID REQUIRED:
NO (based on tests/code -- the claim-preservation chain is sufficient and
would not have been strengthened by an id-based relation for the one
defect found)

CODE CHANGE REQUIRED:
YES -- one targeted fix: `_preservation_blocked_by_attribution_asymmetry`
in realization_resolver.py, closing a proven reported/attributed-speech
false-certification case discovered during this audit (not in the
directive's own required matrix).

QA_ENGINE VERDICT:
PASS post-fix -- the audit's own adversarial search found one real P1-class
defect (reported-attribution asymmetry could falsely certify a rebuttal
frame as preserving a direct claim); fixed and locked in by 3 dedicated
regression tests (broken-before shape, arbiter-attack variant, symmetric-
attribution sanity check) before this verdict. No other unsafe case
survived the founding case, the full same-topic-different-event matrix, or
the always-YES arbiter attack.

P0:
0
P1:
1 (reported-attribution asymmetry -- FIXED, verified by regression tests,
outcome: fixed)
P2:
0
P3:
0

READY FOR ONE VIDEO00 CANARY:
Offline-qualified; awaiting explicit user authorization before any Modal
canary.

Then STOP.

No Modal.
No RAW.

## D-073.1 canary -- Modal run 33907530147, head 7cef038

Authorized ONE Video00 Modal benchmark on the D-073.1-qualified head, full
canonical path (AUTHORITATIVE resolver -> Freeze -> Boundary/Render/QC/
delivery gate if Freeze passes). No code changes before or after the run.

**PATH B fired live for real, correctly.** `realization_resolver_shadow`
(also mirrored in `realization_resolver_authority`, 18/18 ideas):
`real_c5617c48790ccb70a1c1` (discard_reason `semantic_failed_plus_later_
overlapping_complete_retake`) -> `REPLACEMENT_VERIFIED_SEMANTIC`,
`verification_method: "semantic"`, replaced by `real_305dfd300185ebd6e0f0`,
evidence: `same_idea_verified: true`, `critical_claims_preserved: true`,
`unique_required_content_preserved: true`, `hard_gate_results: {contradiction:
false, realization_number_match: true}`, `arbiter_invoked: false` (fully
deterministic, zero incremental LLM cost), 1 preserved claim id. 2 other
orphans resolved via PATH A (`REPLACEMENT_VERIFIED_SAFE`/`lexical`,
unchanged). 7 `PRE_GROUP_REJECTED` (never reached hybrid semantic judgment).
Exactly 1 remained `REVIEW_REQUIRED` (`no_pre_guard_candidate` -- no
discovery candidate existed at all; correctly fails closed, WHEN UNCERTAIN,
KEEP). No unsafe certification, no crash, no arbiter spend beyond what was
already necessary.

**Freeze BLOCKED -- for a reason entirely unrelated to D-073/D-073.1.**
`selection_boundary_contract.status: "not_frozen_freeze_blocked_by_
coherence_review"`. Root cause: StoryValidator's own pre-existing, untouched
`_lost_semantic_atoms` broader content-loss signal on `clip_6ce67f1f00383863
ee5b` (`classification: REAL_CONTENT_LOSS`, `content_loss_suppressed_by:
null` -- same-idea paraphrase credit did not apply to this clip's retry
family) -- `blocking=True` here comes from that broader content-vocabulary
signal (`coverage_against_final_keep < 0.45`), NOT from the co-located
CONTEXTUAL "2023" atom also listed on the same row (confirmed by reading
`final_story_coherence_validation.py`'s own `blocking = content_loss or
any(blocks_freeze(...))` line). `repair_loop` correctly attempted and
declined (`UNIQUE_FACT_LOST`, `no_repair_strategy_exists_for_this_finding_
kind`, `repaired: false`) -- `status: NEEDS_HUMAN_REVIEW`. This exact
class of finding (a `papillary_cancer_preserved`-area content check
flipping) is an established, long-documented recurring pattern in this
fixed Video00 fixture across many prior canaries (D-058 through D-066),
attributed to run-to-run ASR/transcript nondeterminism -- not new, not
caused by this directive, not patched, per the directive's own instruction.

**Downstream (Boundary/Render/PostRender QC/delivery gate): correctly
never attempted.** Per D-050C3's own architecture: a blocked Freeze must
never reach Boundary/Render. `live_render_qc.status: "not_attempted"`,
`preview_skipped_reason: "freeze_blocked_no_render"`, `delivery_status:
"NOT_DELIVERABLE_not_attempted"`. `validate_video00_architecture.py`
(the validator that actually understands this branch) confirmed
`architecture_verified: true`, 0 failed checks, explicitly including
`semantic_failure_correctly_blocked_freeze_and_boundary: true` and
`no_render_attempted_on_a_blocked_semantic_plan: true`.

**Legacy-shaped validators failed as expected, not as a new regression.**
`validate_video00_selection_lock.py` and `validate_video00_regression_qa.py`
both compare against a LEGACY-mode baseline (`baseline_run_id 33126865755`,
23 selected clips) and assume a completed render exists; this run has 19
selected clips (AUTHORITATIVE mode's own freeze-blocked draft,
`output_duration_sec: null`, no render). Human Gold: 14/18 (the same
recurring 4 checks failing as multiple prior canaries:
`papillary_cancer_preserved`, `pimples_micro_2_present`, `pimples_micro_
order`, `sonography_good_before_diagnosis`); the 23-vs-19 count difference
is an explicit non-blocking warning per D-032 (`count_differs_not_treated_
as_failure`). Neither validator understands AUTHORITATIVE-mode
freeze-blocked results as a valid terminal state -- a known, disclosed
harness limitation, not investigated or patched per this directive's own
instruction.

**Modal execution:** `ok: true`, `elapsed_sec: 283.1`, no exception, no
crash-loop. Automatic scale-to-zero on function return -- no persistent
GPU resource created.

D-073.1 CANARY COMPLETE

RESOLVER MODE:
AUTHORITATIVE (confirmed active: 18/18 ideas populated in both shadow and
authority diagnostics)

PATH B LIVE FIRING:
YES -- 1 real orphan certified REPLACEMENT_VERIFIED_SEMANTIC, fully
deterministic, all hard gates recorded true, no unsafe certification

PATH A LIVE FIRING:
YES -- 2 orphans, unchanged REPLACEMENT_VERIFIED_SAFE/lexical

UNRESOLVED ORPHANS:
1 (no_pre_guard_candidate -- fails closed correctly)

FREEZE STATUS:
BLOCKED -- not_frozen_freeze_blocked_by_coherence_review, root cause
entirely unrelated to D-073/D-073.1 (pre-existing StoryValidator broader
content-loss signal, same recurring fixture-specific finding as multiple
prior canaries)

BOUNDARY/RENDER/POSTRENDER QC/DELIVERY GATE:
NOT ATTEMPTED (correct -- architecture validator confirms this is the
required behavior for a blocked Freeze)

ARCHITECTURE VALIDATOR:
PASS (architecture_verified: true, 0 failed checks)

HUMAN GOLD REGRESSION QA:
14/18 (known recurring pattern, unrelated to this directive, not patched)

SELECTION LOCK:
FAIL (legacy-shaped comparison against a different-mode baseline; expected,
not investigated per directive)

MODAL EXECUTION:
ok=true, 283.1s, clean teardown, no persistent GPU resource

PATCHED FROM THIS RESULT:
NO

SECOND RAW LAUNCHED:
NO

Then STOP.

## D-074 -- StoryValidator content-loss forensic (report only, no code, no RAW)

Forensically root-caused the D-073.1 canary's sole Freeze blocker
(`clip_6ce67f1f00383863ee5b`, "Tuve problemas de estómago en una temporada,
en 2023, hay que voltar."). Confirmed via the run's own printed
diagnostics: this discard is part of a real 3-4-take retry cluster for one
story beat (stomach/digestion problems -> endoscopy -> gastritis
diagnosis); the clean, complete winner of that SAME beat
(`clip_7c1583e73e46714ce837`, richer -- includes the actual diagnosis and
treatment duration) IS selected. The clip never entered any `take_judge_
groups` contest (confirmed: its own beat's group `tg_f9f1e23d81a778bb74`
contains only two OTHER clips), so StoryValidator's own `_same_idea_
paraphrase_credit` never had a chance to credit it -- not a bug in that
mechanism, a population it was never designed to see. Root cause: B
(SAME_IDEA_SUPPRESSION_GAP), with C (ASR/segmentation variance in which
takes reach grouping) as a contributing factor. Validator judged
PARTIALLY_CORRECT: accurate on literal token coverage, correctly
conservative given no equivalence evidence, but the resulting REAL_
CONTENT_LOSS classification mischaracterized the actual audience-facing
outcome (the story is told, more completely, elsewhere). Freeze
counterfactual: YES, would have passed -- this was the only blocking
finding in the entire coherence validation. Engine fix judged required and
ready to design; not implemented under this report-only directive.

## D-075 -- pre-group discard semantic preservation design (design only, no code, no RAW)

Accepted design for closing D-074's gap generally (no code). Mapped every
discard population StoryValidator/the Resolver can see (hybrid editorial
deletes, draft-review removals, the `clean_cut_or_composite_resolution`
catch-all, grouped retry losers) against which already carry
`realization_id`/`semantic_idea_id`/claims/replacement provenance/
selected-sibling relation/semantic-equivalence evidence -- confirming the
gap is architectural drift (an unintended seam between two correctly-
scoped authorities), not a deliberate policy.

Designed `SEMANTIC_PRESERVATION_PROOF`: one reusable, directional evidence
object (`GROUPED_SAME_IDEA` | `LEXICAL_REPLACEMENT` | `SEMANTIC_
REPLACEMENT` | `PRE_GROUP_SEMANTIC_PRESERVATION`) all answering the same
question -- is D's required meaning already accounted for by selected R --
minted ONLY by existing authorities (the one new method, PRE_GROUP_
SEMANTIC_PRESERVATION, by the Unified Resolver itself, reusing D-073's own
certification chain verbatim); StoryValidator CONSUMER ONLY, never a new
authority. Candidate discovery for the new method bounded to STRONG
provenance relation (attempt_id match OR source_span_id overlap, plus same
source_asset_id) -- explicitly never temporal proximity or topical
similarity alone. Verified conceptually safe against the founding same-
topic-continuation case, the D-074 stomach/digestion pair (NOT_PROVEN
without execution, correctly not claimed as YES), and the sales/UGC
matrix. No new authority required; no Freeze policy change; LOW
implementation risk; ready for implementation.

## D-076 -- PRE_GROUP_SEMANTIC_PRESERVATION implementation

Implements D-075's design, with one added safety refinement: temporal
proximity alone can never discover a candidate (enforced structurally --
`_pre_group_relationship_evidence` has no access to start/end timing at
all, only provenance identity).

**Real infrastructure gap found and fixed as a prerequisite.**
`CandidateTake.attempt_id`/`.source_span_id` are genuinely minted upstream
(`attempt_reconstruction.py`, `take_segmentation.py`) but were NEVER
threaded through to `DraftClip` -- the schema the Ledger/Resolver actually
consume. Added both as additive, optional `DraftClip` fields (contracts.py)
carried through unchanged at the one existing passthrough site
(`pipeline.py`'s `_draft_clip`, same convention as `realization_id`'s own
D-050A addition) -- and a matching additive `RealizationRecord.source_
asset_id` field (semantic_ledger.py). This activates already-existing,
already-computed upstream identity, not new grouping/ASR logic.

**SEMANTIC_PRESERVATION_PROOF** (realization_resolver.py): typed, frozen
`SemanticPreservationProof` dataclass with the exact fields specified
(discarded_id, preserving_id, proof_method, preserved_claim_ids,
nonrequired_omissions, hard_gate_results, arbiter_invoked, verified,
rejection_reason, relationship_evidence, candidate_discovery_method).
`build_semantic_preservation_proofs` unifies three sources into one
clip_id-keyed lookup: PATH A/PATH B's own existing verdicts (reframed,
never recomputed) for `hybrid_editorial_chunks` discards, plus the one new
`resolve_pre_group_semantic_preservation_shadow` pass over exactly the two
named populations (`clean_cut_or_composite_resolution`, `draft_review_
removed`) -- a SEPARATE, additive pass that never touches `resolve_orphan_
realizations_shadow`'s own existing discard walk or `PRE_GROUP_REJECTED`
verdict for these same records.

**Certification chain reused, not duplicated.** Extracted D-073's own
certification body (candidate-selected check, completeness check, NUMBER
gate, claims fetch, contradiction check, all-claims-preserved loop) into
`_certify_directional_semantic_preservation`, called identically by PATH B
(byte-identical behavior, verified by re-running its full existing suite
unchanged) and by the new `_attempt_pre_group_semantic_preservation`. One
additive parameter, `nonrequired_digit_omissions` (default empty,
byte-identical for PATH B): a digit value the caller has already
classified CONTEXTUAL via the EXISTING `semantic_atom_importance.
classify_number_atom`/`blocks_freeze` (no new importance tier) may be
recorded as a nonrequired omission instead of failing the realization-
level NUMBER gate. Found and fixed mid-implementation that this relaxation
also had to reach the PER-CLAIM content-token subset check (`_claim_
content_subsumed`/`_claims_dedup_equivalent` have no visibility into the
realization-level gate's own classification) -- added `_sanitize_claim_
for_nonrequired_omissions`, which strips ONLY the classified-safe digit
value from one claim's own `content_tokens`/`text` before the shared
per-claim loop runs; returns the SAME object unchanged (identity, not just
equivalence) when no omissions are supplied, guaranteeing PATH B is
unaffected.

**Candidate discovery** (`_find_pre_group_candidates`,
`_pre_group_relationship_evidence`): same `source_asset_id` AND (shared
`attempt_id` OR overlapping `source_span_ids`) -- provenance identity
only, no text, no timing. Tries every qualifying candidate in stable order
against the full certification chain; stops at the first that verifies.

**Two real defects found and fixed during implementation (self-
administered QA_ENGINE):** (1) the realization-level number relaxation
did not reach the per-claim content-token check (found via the exact
D-074 generic fixture failing `required_claim_not_preserved` despite a
correctly-classified CONTEXTUAL year) -- fixed via claim sanitization
above. (2) StoryValidator's own consumption originally only fired `if
blocking`, silently skipping a case where a verified proof existed but the
row was already non-blocking for an unrelated reason (a lone CONTEXTUAL
atom) -- Section 6's "must not be silently dropped from diagnostics"
requires surfacing the proof either way; relaxed to fire whenever
GROUPED_SAME_IDEA did not already claim the row, never overriding that
mechanism's own precedence.

**StoryValidator consumption** (`final_story_coherence_validation.py`):
`_lost_semantic_atoms` gained an additive `semantic_preservation_proofs`
parameter (default `None`, byte-identical to before D-076 when omitted);
one dict lookup per row, no candidate discovery, no claim extraction, no
arbiter invocation of its own. Wired only at `universal_clean_cut.py`'s
AUTHORITATIVE-mode second StoryValidator pass (the one point in the
pipeline both the Ledger and a resolved draft exist together) -- the first,
pre-Ledger StoryValidator pass and LEGACY/SHADOW mode are structurally
unaffected. Pre-group proofs computed once and shared between StoryValidator
consumption and the new `diagnostics["semantic_preservation_proofs"]`
observability key, avoiding a redundant second arbiter pass.

**POST_RESOLVER_SEMANTIC_MUTATORS reconfirmed 0.** StoryValidator's
consumption changes only its own `blocking`/`classification` computation
(a validation-gate decision it already owned); nothing new mutates
`selected`/`discarded` membership.

Verified: 21 new tests (`test_cutsell_d076_pre_group_semantic_
preservation.py`) covering the D-074 generic regression, the mandatory
no-relation negative control (different attempt, different source,
temporally-close-but-unrelated), historical same-topic-continuation safety
even with a relation supplied, the full sales/UGC matrix, the reported-
attribution regression re-run through the new path, the always-YES arbiter
attack (proven unable to create a relation or override any hard gate), and
StoryValidator consumption (suppression, fail-closed with no proof,
byte-identical when the parameter is omitted). 0 regressions:
tests/test_cutsell_*.py 1770 passed (1749 baseline + 21 new). CleanCutBench
54/54 LEGACY, 54/54 AUTHORITATIVE. PATH B's own founding tests unaffected.

Final report (verbatim, as delivered):

D-076 COMPLETE

SEMANTIC_PRESERVATION_PROOF:
PASS

PRE_GROUP PRESERVATION:
PASS

STRONG RELATION REQUIRED:
PASS (same source_asset_id AND attempt_id match OR source_span_id overlap
-- both real, upstream-minted identities now threaded through DraftClip)

TEMPORAL-ONLY DISCOVERY:
IMPOSSIBLE (structural -- the discovery function has no access to
start/end timing at all)

STORYVALIDATOR CONSUMER ONLY:
PASS

D-074 GENERIC CASE:
PASS

NO-RELATION NEGATIVE CONTROL:
PASS

SAME-TOPIC CONTINUATION SAFETY:
PASS

NUMBER SAFETY:
PASS

NEGATION SAFETY:
PASS

ATTRIBUTION SAFETY:
PASS

ENTITY/DIAGNOSIS SAFETY:
PASS

CAUSAL SAFETY:
PASS

TEMPORAL SAFETY:
PASS

SALES/UGC GENERALIZATION:
PASS

POST_RESOLVER SEMANTIC MUTATORS:
0

LEGACY:
54/54

AUTHORITATIVE:
54/54

FULL TEST COUNT:
tests/test_cutsell_*.py: 1770 passed, 0 failed (1749 baseline + 21 new).
D-050-D-076 targeted sweep: 426 passed. CleanCutBench: 54/54 LEGACY, 54/54
AUTHORITATIVE.

QA_ENGINE VERDICT:
PASS -- self-administered adversarial review (no-relation negative
control, same-topic continuation with an artificially-supplied relation,
reverse-direction loss, number, negation, attribution, entity/diagnosis,
causal, temporal, incomplete candidate, ASR uncertainty/no-extractable-
claims, always-YES arbiter) found two real defects during implementation
(realization-level number relaxation not reaching the per-claim check;
StoryValidator silently skipping an already-non-blocking row with a valid
proof) -- both fixed and locked in by dedicated regression tests before
this verdict. No unsafe suppression found for any tested shape.

P0:
0
P1:
0
P2:
0
P3:
0

READY FOR ONE VIDEO00 CANARY:
Offline-qualified; awaiting explicit user authorization before any Modal
canary.

Then STOP.

Do not launch Modal.

## D-077 -- PRE_GROUP claim-equivalence completion (implementation)

Closes D-076's own disclosed limitation: the literal-token-subset
certification chain (PATH A, PATH B, and D-076's PRE_GROUP_SEMANTIC_
PRESERVATION) could not bridge a genuine SYNONYM paraphrase with no
literal word overlap on the specific fact (e.g. "problemas de estomago"
vs "problemas de digestion") -- confirmed via a generic stand-in fixture,
"I had stomach problems for a while." vs "I had digestive problems, had
an endoscopy, and was diagnosed with gastritis.": the discarded claim
classifies `ACTION_EVENT` (the marker-less, generic-statement fallback),
the candidate's own richer clause classifies `DIAGNOSIS_IDENTIFICATION`
(triggered by "diagnosed with"), so the pre-existing exact-claim-type-
match gate on `_claims_dedup_equivalent` blocked the pair from ever
reaching the existing D-061 claim-equivalence arbiter, and the
deterministic cross-type superset check (`_claim_content_subsumed`)
requires a literal token subset a synonym pair can never satisfy.

IMPLEMENTATION: one new, additive, OPT-IN claim-matching strategy,
`cross_type_ambiguous_bridge`, added as the 4th (last-tried) strategy
inside `_claim_preserved` (realization_resolver.py) -- reached only after
`dedup_equivalent`/`content_subsumed`/`hindsight_semantic` have all
already failed deterministically (Section 2's "deterministic first").
Gated by a new `allow_cross_type_ambiguous_bridge` parameter threaded
through `_claim_preserved` and `_certify_directional_semantic_
preservation`, defaulting `False` everywhere -- PATH B's own call site
(`_attempt_semantic_replacement_certification`) never passes it, so PATH
B is byte-identical (proven by a dedicated regression test routing the
exact cross-type synonym pair through PATH B's own `hybrid_editorial_
chunks` population and confirming REVIEW_REQUIRED, zero arbiter calls).
Only D-076's own pre-group path (`_attempt_pre_group_semantic_
preservation`) opts in.

`_cross_type_ambiguous_bridge_eligible(d_claim, r_claim)` gates entry:
  1. ENTITY/DIAGNOSIS SAFETY (Section 3/9), drawn structurally rather than
     via a new entity extractor ("no new provider/model/protocol"):
     requires `d_claim.claim_type == ACTION_EVENT` -- a claim already
     classified ENTITY_RELATION/STATE_RESULT/DIAGNOSIS_IDENTIFICATION/
     MEASUREMENT_QUANTITY/UNIQUE_CONCLUSION/CORRECTION/NEGATION already
     asserts something SPECIFIC of its own (a diagnosis, a measured
     value, a correction, a negation); an ACTION_EVENT claim, by
     construction, never does, so a richer candidate can only ever be
     ADDING to it, never substituting a fact D itself already named. The
     existing D-073 diagnosis-substitution fixture ("...it was gastritis"
     vs "...it was an ulcer") is `ENTITY_RELATION`, not `ACTION_EVENT`, on
     D's own side -- this gate structurally never reaches it; the
     pre-existing exact-type-match dedup gate still protects it,
     unchanged (re-verified through the pre-group path too, Section 9).
  2. candidate side restricted to a fixed allowlist of "asserts something
     specific" types, excluding NEGATION/CORRECTION/CAUSE_EFFECT/
     TEMPORAL_RELATION;
  3. `_claim_signals_direction_sensitive` re-checked on BOTH sides
     (causal/temporal safety);
  4. negation-polarity agreement;
  5. digit-value subset agreement (D's own numbers, if any, must survive
     in R);
  6. overlap coefficient (non-digit content tokens) must fall in the SAME
     genuinely-ambiguous band PATH B's own dedup arbiter already uses
     (`_DEDUP_AMBIGUOUS_FLOOR` to `_CLAIM_DEDUP_THRESHOLD`) -- confidently
     similar or confidently dissimilar pairs never reach the arbiter.

Candidate discovery (D-076's own strong-relation requirement -- same
`source_asset_id` AND matching `attempt_id`/overlapping `source_span_
ids`) is completely untouched; this directive adds no new discovery path
and the arbiter cannot manufacture a relation (verified: an always-YES
arbiter with no relation supplied never gets called at all).

Reuses the EXACT existing D-061 `ClaimEquivalenceArbiter.claim_covered`
protocol -- no new provider, model, budget system, or retry loop. When
the configured arbiter is `None`, fails closed (no verification).

OBSERVABILITY (Section 12): every consultation this strategy attempts is
recorded (whether reachable or not) into a new additive `arbiter_
consultations` field on `SemanticPreservationProof` and on the shared
`evidence["arbiter_consultations"]` dict key -- method, deterministic
result, eligibility, invocation, verdict, confidence, reason, and
best-effort provider/model/budget introspection (`_arbiter_provenance`,
read-only `getattr`, never fabricated). Confirmed via dedicated tests:
zero arbiter calls for a deterministic pass (literal-subset D-074 case)
and a deterministic fail (number mismatch); exactly one call, fully
recorded, for the genuine ambiguous cross-type case.

StoryValidator's own consumption is completely unchanged -- no new
StoryValidator code path; the synonym-shaped proof flows through the
exact same `semantic_preservation_proofs` lookup D-076 already wired
(verified with a dedicated test).

TESTS: tests/test_cutsell_d077_pre_group_claim_equivalence.py, 20 new
tests -- D-074 synonym shape (verifies with arbiter, fails closed without
one, never overridden by a declining honest arbiter), mandatory no-
relation negative control (even with always-YES), same-topic different-
event, the full hard-mismatch matrix (number/negation/diagnosis/
attribution/causal/temporal, each with an always-YES arbiter and zero
calls), the existing D-073 diagnosis fixture re-verified through the new
path, sales/UGC positive (ambiguous, may verify) and negative (hard
failure, never reaches arbiter) generalization, cost/observability
(0 calls on deterministic pass/fail, full consultation record on the
ambiguous case), strong relation via `source_span_relation` alone,
StoryValidator consumer-only re-verification, and PATH B byte-identical
regression.

QUALIFICATION: compileall clean; D-050-D-077 targeted sweep 446 passed;
CleanCutBench 54/54 LEGACY and 54/54 AUTHORITATIVE; full suite 1790
passed (1770 baseline + 20 new), 0 regressions.

QA_ENGINE VERDICT: PASS. Adversarial matrix (always-YES arbiter attack
against every hard gate: number, negation, diagnosis/entity, attribution,
causal, temporal, same-topic continuation, no-relation-at-all, and the
pre-existing D-073 diagnosis-substitution fixture routed through this new
path) found no unsafe override in any case -- every one fails closed with
zero arbiter calls, confirming the hard gates run BEFORE the arbiter is
even reachable, not merely that a well-behaved arbiter happens to decline.

P0: 0
P1: 0
P2: 0
P3: 0

READY FOR ONE VIDEO00 CANARY: offline-qualified; awaiting explicit user
authorization before any Modal canary.

Then STOP.

No Modal. No RAW.

## D-078 -- final Freeze-blocker dual-truth forensic (report only, no code, no RAW)

Forensically root-caused both remaining D-077 canary Freeze blockers using the
canary's own live diagnostics (run 33917709135, head 86253f3).

**CRITICAL_CLAIM_LOST dual-truth bug**: `_lost_semantic_atoms` (bag-of-words,
GROUPED_SAME_IDEA-eligible) correctly classified the estomago/gastritis clip
non-blocking; `_lost_critical_claims` (D-038's own, completely separate
claim-coverage backstop -- its own `claim_coverage()`/`resolve_ambiguous_
coverage()` chain, no shared claim identity with the Ledger, never consuming
any `SemanticPreservationProof`) independently flagged the SAME source text
as CRITICAL_CLAIM_LOST. Root cause: `classify_claim`'s own unconditional
"any negation token -> the whole clause is NEGATION" rule fired on a bare
"no" inside a rhetorical/idiomatic aside ("no hay que preguntar" / "no need
to ask") that does not negate the clause's actual asserted content (having
stomach/digestive problems); `claim_coverage()`'s negation-flip guard then
capped coverage at 0.05, below its own ambiguous-arbiter floor (0.10), so
the claim never even reached an arbiter. Human-meaning check: YES, the final
video fully communicates the claim -- SAME_PROPOSITION as the already-
preserved story atom, not a distinct or contradictory one.

**Separate, unrelated blocker**: DUPLICATE_IDEA/UNRESOLVED_RETRY on
`tg_bb8eb9fceecf9c6d6f` -- classified AUTO_RESOLVABLE (Candidate B carries a
unique, Human-Gold-verified critical fact; Candidate A is a higher-delivery-
score restatement of only the non-critical hindsight half, already covered,
in briefer form, inside Candidate B). D-063 CRITICAL_COVERAGE_DOMINANCE +
D-066's hindsight-alignment mechanism were shown (offline, `tests/
test_cutsell_d066_hindsight_alignment.py`) to already resolve this EXACT
shape correctly given a confirming arbiter -- not a missing mechanism, an
observed-live absence/decline of that one arbiter consultation for this pair.

Freeze counterfactual: fixing both blockers, no other blocker remains --
Freeze would pass. Recommended design: `_lost_critical_claims` should
consume an existing, claim-SCOPED (never idea- or clip-scoped)
`SemanticPreservationProof` for the exact canonical claim id, never a
coarse `GROUPED_SAME_IDEA`/clip-level credit. Not implemented in D-078
(report only, per directive).

## D-079 -- final Freeze-blocker integration: claim single-truth + dominance reachability

Implements D-078's own recommended design for the CRITICAL_CLAIM_LOST dual-
truth bug; confirms (does not re-implement) the D-063/D-066 dominance
mechanism already resolves the retry-family blocker correctly.

PHASE 1 -- CANONICAL CLAIM IDENTITY: `semantic_claims.extract_claims`
already mints `Claim.canonical_claim_id` via `mint_canonical_claim_id
(claim_type, content_tokens)` -- the SAME function the Semantic Ledger's own
`CanonicalClaimRecord.canonical_claim_id` uses. This is a pure function of a
claim's own type+content, never of idea id or clip id -- the SAME source
clip's SAME clause, extracted independently by `_lost_critical_claims` and
by the Ledger, produces the IDENTICAL id. No new minting scheme; this
existing, already-shared identity space is simply now READ from both sides.

PHASE 2 -- CLAIM-SCOPED PROOF CONSUMPTION: `_lost_critical_claims` gains an
additive `critical_claim_preservation_index: Mapping[str, object] | None =
None` parameter (default None -- byte-identical everywhere else). Before
emitting a CRITICAL_CLAIM_LOST finding, it checks `index.get(claim.
canonical_claim_id)`; suppresses ONLY when `proof.verified` AND the exact
canonical claim id is a member of `proof.preserved_claim_ids` (re-checked
explicitly, never trusted implicitly) -- recording `claim_preservation_
consumed`, `canonical_claim_id`, `proof_method`, `preserving_realization_id`
into the existing `claim_coverage_confirmations` list. `build_preserved_
claim_id_index` (realization_resolver.py) builds this index from ALL
verified proofs' `preserved_claim_ids` -- `GROUPED_SAME_IDEA` (StoryValidator's
own clip-level credit) carries no `preserved_claim_ids` and can structurally
never appear here, satisfying "a generic same-idea proof is NOT enough."

The index needed a genuinely new proof source: PATH A/B only evaluate TRUE
orphans (no `semantic_idea_id`), D-076's pre-group pass only evaluates
discards that never reached `hybrid_editorial_chunks`'s own semantic
judgment. A discard that DID reach grouping and lost, WITHIN its own idea,
to that idea's own resolved winner/composite was never evaluated by ANY
claim-level chain at all -- the third, remaining population. New, additive
`INTRA_IDEA_SEMANTIC_PRESERVATION` certification path (realization_
resolver.py): `resolve_intra_idea_semantic_preservation_shadow` walks every
`resolve_realizations_shadow` idea resolved RESOLVED_WINNER/RESOLVED_
COMPOSITE, certifies every non-winning member against that idea's OWN
resolved winner/composite (candidate discovery = the idea's own resolution,
never a search) via the EXACT SAME shared `_certify_directional_semantic_
preservation` chain PATH B/D-076/D-077 use, unmodified.

PHASE 1/4 root fix: a new, narrow, opt-in `rhetorical_aside_negation_bridge`
matching strategy inside `_claim_preserved` (`allow_rhetorical_aside_
negation_bridge`, default False -- PATH B's and D-076's own calls pass
nothing, byte-identical). Eligible ONLY when a claim is NEGATION-typed, not
already CONTRASTIVE_HINDSIGHT_NEGATION (D-066 owns that shape), and its raw
text contains an EXACT match from a small, fixed, general, bilingual marker
list (`_RHETORICAL_ASIDE_NEGATION_MARKERS`: "no need to ask" / "no hay que
preguntar" / "needless to say" / "no doubt" / etc. -- never a bare "no"/"not"
token, so a genuine factual negation like "did not have gastritis" is
structurally unaffected). Strips ONLY that exact phrase, re-classifies the
residual via the UNMODIFIED `classify_claim`, and requires the residual
carry NO remaining negation marker of its own (`residual_still_negated`
fails closed) and NOT re-classify NEGATION itself. The residual (now
typically ACTION_EVENT) is then matched via the EXISTING, unmodified dedup/
subsumed/cross-type-bridge strategies -- never a new matching primitive.
`classify_claim`/`_negations`/`_claim_has_negation` and every existing
NEGATION hard gate are completely untouched globally.

PHASE 3 -- HARD SAFETY: re-verified via adversarial tests with an always-
YES arbiter that number mismatch, negation reversal, diagnosis substitution,
attribution asymmetry, causal reversal, and same-topic-different-event can
never be suppressed by the new path -- each fails via the SAME pre-existing
hard gates the shared chain already enforces (fully reused, not
reimplemented), or via the idea's own Resolver-level contradiction check
firing even earlier (no certification is even attempted).

PHASE 5/6/7 -- D-063 DOMINANCE REACHABILITY: traced and confirmed (no code
change -- `claim_coverage_best_take.py` is not touched by D-079 at all)
that `_critical_coverage_dominant_candidate` + D-066's `_find_hindsight_
alignment` ALREADY run, unconditionally, before every DUPLICATE_IDEA/
UNRESOLVED_RETRY emission for a 2+-selected family, and ALREADY correctly
resolve the exact D-078 retry shape (Candidate A: critical diagnosis +
hindsight, delivery 0.9; Candidate B: hindsight only, delivery 0.95) to A,
via existing, unmodified `tests/test_cutsell_d066_hindsight_alignment.py`
tests (`test_d064_generic_chain_auto_resolves_with_confirming_arbiter`);
without a confirming arbiter the family correctly stays REVIEW_REQUIRED,
never forced (`test_d064_generic_chain_stays_ambiguous_without_arbiter_
confirmation`/`_with_no_arbiter_at_all`) -- Phase 6's own "true tie
preservation" and Phase 7's own retry-regression requirement, both already
green, re-run here as confirmation.

PHASE 8 -- SINGLE-TRUTH INVARIANT: dedicated tests confirm that for the
exact same canonical required proposition, a verified claim-scoped
preservation proof and a CRITICAL_CLAIM_LOST finding never coexist, across
both a genuine positive case (verified proof, no finding) and a genuine
negative case (no proof reaches verified, finding still emitted).

PHASE 9 -- NO NEW AUTHORITY: reconfirmed. The Unified Resolver / existing
claim-preservation chain mints the one new proof type; `_lost_critical_
claims` only consumes it (a dict lookup gating whether it emits its own,
pre-existing finding kind) -- never mutates `selected`/`discarded`
membership. POST_RESOLVER_SEMANTIC_MUTATORS stays 0.

TESTS: tests/test_cutsell_d079_claim_single_truth.py, 18 new tests --
canonical claim identity sharing, fail-closed with no canonical identity/no
arbiter, the D-078 negation shape resolving via the new intra-idea proof
(with and without arbiter), the mandatory no-relation negative control, the
full hard-safety matrix (number/negation/diagnosis/attribution/causal/
same-topic-different-event) under an always-YES arbiter, rhetorical-aside
marker exact-phrase-only safety (a bare negation without the marker never
bridges; a genuine second negation alongside the marker never bridges
either), sales/UGC positive/negative generalization, the single-truth
invariant (positive and negative cases), and StoryValidator/`_lost_
critical_claims` consumer-only re-verification.

QUALIFICATION: compileall clean; D-050-D-079 targeted sweep 464 passed;
CleanCutBench 54/54 LEGACY and 54/54 AUTHORITATIVE; full suite 1808 passed
(1790 baseline + 18 new), 0 regressions.

QA_ENGINE VERDICT: PASS. Adversarial matrix (always-YES arbiter attack
against every hard gate for the NEW rhetorical-aside-bridge specifically,
mandatory no-relation negative control, canonical-id-to-wrong-proposition
attack via Phase 1's own deterministic minting) found no unsafe suppression
in any case.

P0: 0
P1: 0
P2: 0
P3: 0

READY FOR ONE VIDEO00 CANARY: offline-qualified; awaiting explicit user
authorization before any Modal canary.

Then STOP.

No Modal. No RAW.

## D-079 canary (run 33925842853) -- Human Gold 13/18, root cause deferred to D-080

Freeze BLOCKED via `_lost_semantic_atoms` (pre-existing, unmodified by D-079)
on an incidental "2023" atom in a discarded gastritis clip -- unrelated to
either D-079 validation objective. `lost_critical_claims: []` and
`deterministic_best_take_authority: {"status": "absent"}` for the whole run:
D-079's claim-scoped suppression and D-063/D-066 dominance were both
NOT_EXERCISED, not falsified -- the papillary-cancer retry pair never
reached `take_judge_groups` as a contested family at all, because the
biopsy-confirmation clip was deleted upstream by `hybrid_editorial`'s own
P1_RETRY_EQUIVALENCE pass before grouping ever ran. Full forensic: D-080.

## D-080 -- upstream semantic decision stability forensic (report only)

Compared the D-077 canary (18/18) against the D-079 canary (13/18) on the
same source. Root cause: two separate live-LLM semantic-judgment stages each
returned a different verdict on materially identical deterministic evidence
across the two runs.

PAPILLARY: `hybrid_editorial`'s P1_RETRY_EQUIVALENCE pass relabeled the
biopsy-confirmation clip from `"winner"` (0.94, kept fail-open) to `"failed"`
(0.88, `semantic_failed_plus_local_performance`) on the identical
`local_failure_corroborated=True`/`dense_physical_reset:8` signal in both
runs, and `applied_delete=True` removed it before grouping/take_judge/
dominance/StoryValidator ever saw it.

SONOGRAPHY: `take_judge_groups`' own semantic-candidate classifier
(`pipeline.py::_semantic_best_take`) relabeled the same short/long retry
pair from a decisive `{"failed","winner"}` (which correctly overrode the raw
DeliveryScorer ranking via `semantic_override_applied`) to a non-actionable
`{"keep","keep"}` on byte-identical DeliveryScorer scores (`0.6817`/
`0.6211`), causing `_semantic_best_take` to fall through to the raw,
completeness-blind local score and select the incomplete clip.

Neither divergence traces to ASR (text/boundaries stable both runs) or to
semantic-compute budget/ordering (identical `requested_chunk_count`,
identical budget-exhaustion cutoff, the affected chunk fully available and
executed in both runs). Both are UNSTABLE_MODEL_DECISION.

Architectural finding: paid LLM judgments currently hold irreversible
destructive-delete authority (`hybrid_editorial`) before any authoritative
semantic layer inspects the candidate. Recommendation: evidence-only until
Resolver (option B), never destructive authority.

## D-081 -- pre-Resolver destructive semantic authority cutover

Implements D-080's architectural recommendation for the papillary-cancer
class of regression. Canonical rule: MECHANICAL CERTAINTY MAY DELETE EARLY;
SEMANTIC JUDGMENT MAY NOT IRREVERSIBLY DELETE BEFORE THE AUTHORITATIVE
RESOLUTION BOUNDARY.

CUTOVER (`cutsell_worker/hybrid_session_cleanup.py`): every `delete_basis`
driven by a probabilistic LLM label (`high_confidence_semantic`,
`semantic_failed_plus_later_overlapping_complete_retake`,
`semantic_failed_plus_local_performance`, `semantic_bts_plus_local_
performance`, `semantic_bts_inside_corroborated_failure_cluster` --
collected in the new `_SEMANTIC_JUDGMENT_DELETE_BASES`) no longer sets
`applied_delete=True`. Each instead sets a new additive `semantic_delete_
recommended: bool` evidence field (label/confidence/local_failure_
corroborated/local_failure_reasons/delete_basis/cost diagnostics all
preserved unchanged for observability). The candidate stays in `kept`,
reachable by grouping, claim extraction, critical coverage, BestTake/
dominance, the Unified Resolver, and Semantic Ledger preservation exactly as
before -- no new Resolver was created; every existing downstream authority
decides survival unchanged.

MECHANICAL EXCEPTION PRESERVED: `micro_failed_plus_local_performance`
(near-empty fragment, duration<=1.25s AND token_count<=2, matching the
"deterministically unusable fragment" mechanical-certainty class) is the
one basis still an actual, early, real delete -- not every reject became
KEEP.

DOWNSTREAM CONSUMERS AUDITED: `hybrid_story_guard.py`'s and
`hybrid_composite_best_take.py`'s post-hoc "restore a performance-only
deletion" guards both key off `applied_delete`/`delete_basis` from this same
decision structure; since the affected bases no longer land in `deleted`,
both guards become safe no-ops for that basis rather than needing their own
changes. `hybrid_group_cleanup.py` contains a structurally similar
destructive delete but is dead code (imported/called nowhere in the live
pipeline) -- left untouched per scope discipline, not "spend engineering
time" on an unreachable path.

TESTS: 6 existing `tests/test_cutsell_hybrid_session_cleanup.py` cases and 1
`tests/test_cutsell_hybrid_pipeline.py` case updated from asserting the old
destructive behavior to the new evidence-only behavior (each still verifies
the SAME evidence -- label, confidence, corroboration -- is still recorded,
only that it no longer deletes). New `tests/test_cutsell_d081_pre_resolver_
semantic_authority.py` (10 tests): the mechanical exception still deletes;
every semantic-judgment basis enumerated as evidence-only; a generic
papillary-shape reproduction (critical content + local friction + LLM
"failed" survives); a model-variance matrix across all four `winner/failed`,
`keep/keep`, `failed/keep`, `winner/winner` label shapes proving neither
candidate is ever removed pre-Resolver in any shape; a true-failed-take case
proving a real downstream authority (`take_judge.rank_takes`, unmodified,
"never deletes content") still ranks the complete delivery above an
incomplete, corroborated-failed candidate once both are visible to it;
unique-required-fact safety; and three sales/UGC generalizations (a rough
take that is the only carrier of a dosage instruction, the same dosage take
safely losing downstream once a clean retry restates it, and a product-demo
step clip that must remain available).

QUALIFICATION: compileall clean; targeted D-050 through D-081 sweep (473
tests, including the new D-081 file) all pass; CleanCutBench 54/54, run
twice for determinism; full `tests/` suite (excluding `test_semantic_
stitch.py`, a pre-existing, unrelated collection error reproduced identically
on the pre-D-081 head) -- 2442 passed, 2 pre-existing unrelated failures
confirmed present on the pre-D-081 head too (`test_hybrid_story_guard_
incomplete_retry.py`'s `_covered_by_kept_delivery` case;
`test_video00_modal_hybrid_semantic_parity.py`'s CI-log-masking case) and
therefore deselected, not caused by this change. 0 new regressions.

CANDIDATE-COUNT IMPACT (Section 12): a synthetic 40-candidate stress fixture
(20% failed+corroborated, 10% bts+corroborated, 5% bts-high-confidence-alone)
showed 33 of 40 candidates that would have been destructively deleted under
the old rule now correctly carried downstream, 0 actually removed (no
near-empty fragments in the fixture). No new LLM call type or call volume
was introduced at this stage -- cost impact falls entirely on existing,
already-bounded downstream stages (`take_judge_groups`' own chunk_size/
window_stride windowing, `claim_coverage_best_take`'s per-retry-family
scope) which already process whatever `kept` population arrives; no new
deterministic bounding was added because none of the existing bounds are
sized off `hybrid_session_cleanup`'s own delete count.

QA_ENGINE VERDICT: PARTIAL PASS, with one carried-forward P1 finding.

The PAPILLARY-class regression (destructive delete before the authoritative
boundary) is fully closed and tested. The SONOGRAPHY-class regression is
NOT closed by this directive: its root cause is a materially different
mechanism -- `pipeline.py::_semantic_best_take` unconditionally falls back
to the raw, completeness-blind local (DeliveryScorer) rank whenever
`take_judge_groups`' semantic-candidate labels are non-decisive (zero or 2+
"winner" labels), which is exactly what happened when the labels degraded
from `{"failed","winner"}` to `{"keep","keep"}` across the two D-080 runs.
This violates the same D-081 canonical rule ("Raw DeliveryScore must not
become final authority merely because LLM labels are non-actionable") at a
second site, but fixing it safely requires changing take_judge_groups' core
one-winner-per-retry-family selection semantics -- a load-bearing mechanism
used by every retry family in the pipeline, not only these two shapes -- and
deciding how a non-decisive label set should hand off to (without
duplicating) the existing claim_coverage_best_take dominance mechanism
without creating a new Resolver (Section 5). That scoping and testing was
judged too large to responsibly complete inside this directive; it is
recorded here as open, not silently left unaddressed, and should be
implemented as its own scoped directive (D-082) before being relied upon to
close the sonography-class failure mode. A narrower, pre-existing residual
was also confirmed and accepted: the preserved mechanical
`micro_failed_plus_local_performance` exception (<=1.25s, <=2 tokens) could
in principle still destructively delete a genuinely critical single-token
utterance; this is unchanged, pre-existing, explicitly authorized behavior
(D-081 Section 4's own exception list), judged low-likelihood in practice
(ASR/attempt-reconstruction already merges speech into multi-word attempts
before this stage ever sees a candidate) and tracked as a monitored
residual, not a new defect.

P0: 0
P1: 1 (take_judge_groups' semantic-label-non-decisive fallback to raw
DeliveryScorer -- the sonography-class regression -- remains open; requires
D-082)
P2: 0
P3: 1 (micro_failed_plus_local_performance mechanical exception could in
principle delete a genuinely critical 1-2 token utterance; pre-existing,
bounded, monitored)

READY FOR CONTROLLED STABILITY BATTERY: PARTIAL -- the papillary-class fix
is ready; the sonography-class root cause requires D-082 before the full
D-080 stability matrix can be considered closed.

Then STOP.

No Modal. No RAW.

## D-082 -- non-decisive semantic label fallback: authoritative Best-Take stability

Closes D-081's own carried-forward P1: `pipeline.py::_semantic_best_take`
fell straight through to the raw, completeness-blind local (DeliveryScorer)
rank whenever `take_judge_groups`'s semantic-candidate labels were
non-decisive (zero or 2+ "winner" labels) -- the exact D-080 sonography
mechanism, where labels degraded from a decisive `{"failed","winner"}` to a
non-actionable `{"keep","keep"}` across two live runs on byte-identical
DeliveryScorer scores.

CANONICAL FALLBACK ORDER: when semantic labels are decisive (exactly one
"winner"), behavior is byte-identical to before D-082. When they are not,
`_semantic_best_take` now consults, in order, deterministic evidence this
codebase already computes elsewhere -- no new authority, no weighted
scoring:
1. D-081 `semantic_delete_recommended` evidence (soft exclusion, unless it
   would eliminate every candidate);
2. attempt completeness (`CandidateTake.complete_idea`, the exact signal
   `take_judge.score_take` already weights -- soft, same fail-open rule);
3/4. D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, reused verbatim via the
   new `claim_coverage_best_take.resolve_critical_coverage_dominance`
   (factored out of `apply_claim_coverage_best_take`'s own already-
   multi-selected-family branch so both callers share one dominance
   decision -- never reimplemented, never two copies);
5. unique-required-fact safety: delivery may only settle a candidate set
   whose CRITICAL-claim coverage (`claim_coverage_best_take.
   critical_coverage_sets`, the same per-candidate coverage sets dominance
   itself computes, exposed read-only) is genuinely IDENTICAL across
   survivors -- a disjoint/asymmetric split is left exactly as it was
   (`local_selected_clip_id`, the pre-D-082 safe default), never forced.
   Hardened with a direct `any_pair_contradicts` re-check (the SAME safety
   gate dominance's own internal implementation already uses) so a genuine
   factual contradiction can never reach delivery even if two coverage sets
   ever looked identical by claim-identity coincidence;
6-9. only once nothing above resolved it does the local DeliveryScorer
   ranking decide among the surviving, safe candidates -- delivery's proper
   role once content is effectively tied.

Any step finding nothing decisive falls open to the next; the final
fallback is always `local_selected_clip_id`, never worse than pre-D-082
behavior for a genuinely unresolved family.

MONKEYPATCH COMPANION FIX (required, not optional): `semantic_best_take_
integrity.py`'s `install_semantic_best_take_integrity()` replaces
`pipeline._semantic_best_take` at import time with a wrapper carrying three
independent, already-tested legacy tie-break heuristics (`_prefer_clear_
nonfailed_peer`, `_prefer_information_rich_tied_winner`, `_prefer_complete_
peer_with_preserved_critical_facts`) layered on top of whatever the base
function returns. Its signature/return arity hard-coded the pre-D-082
3-positional-argument, 2-tuple contract -- updated (mechanically, its own
internal logic and precedence order untouched) to accept the new `ranked`/
`semantic_delete_recommended` parameters, pass them through to `original`,
and return the new 3-tuple `(selected, preferred, reason)`. Without this
companion fix every real invocation through the package's normal import
path would raise `TypeError` on the missing `ranked` argument.

TESTS: 3 pre-existing `tests/test_cutsell_semantic_best_take.py` cases
updated for the new 3-tuple return contract (one also gained an explicit
`ranked=()` expectation for the genuine "nothing to consult" case). New
`tests/test_cutsell_d082_non_decisive_semantic_fallback.py` (15 tests): the
core sonography stability regression (decisive AND non-decisive label
variants, both correctly selecting the complete candidate); winner/winner
variance preferring the objectively more complete candidate; a 5-shape
model-label-variance matrix (`failed/winner`, `keep/keep`, `winner/winner`,
`keep/winner`, `failed/keep`) proving invariant final selection; critical
coverage precedence over delivery; two genuinely distinct critical facts
staying unresolved; a richer-but-explicitly-incomplete candidate never
dominating solely on word count; delivery correctly deciding a genuine tie;
D-081 evidence integration (soft exclusion, and confirmation it never
eliminates a whole group); three sales/UGC generalizations (a required
claim winning despite a lower delivery score, delivery favoring a cleaner
take when nothing required is at stake, and a price/offer claim winning
over a higher delivery score); decisive-label byte-identical regression;
and a QA_ENGINE-hardening test proving a genuine contradiction never reaches
delivery even with a large score gap in the contradicting candidate's
favor.

QUALIFICATION: compileall clean; targeted D-050 through D-082 sweep (558
tests) all pass; CleanCutBench 54/54, run twice; full `tests/` suite
(same 2 pre-existing, unrelated exclusions as D-081, confirmed unrelated) --
2457 passed (2442 D-081 baseline + 15 new D-082 tests), 0 new regressions.

QA_ENGINE VERDICT: PASS, after one hardening fix applied during self-review
(the direct `any_pair_contradicts` re-check in Step 5, described above --
a genuine contradiction could in principle have reached delivery if two
coverage sets ever collided to look identical; closed before this report,
not left as an open finding). Primary attacks run: higher DeliveryScore
beating a complete lower-score take (blocked by completeness/dominance
running first); a richer-but-unusable take beating a clean usable one
(blocked by the completeness exclusion, tested); unique fact loss (the
per-group single-winner model is pre-existing and unchanged by D-082 --
StoryValidator's own independent `_lost_critical_claims`/`_lost_semantic_
atoms` backstops remain the safety net for whatever a group's winner does
not cover, exactly as before); two genuinely distinct critical facts forced
into one winner (blocked, tested); model-label variance changing the final
result despite decisive deterministic evidence (tested invariant across 5
label shapes); delivery no longer functioning when content is genuinely
tied (tested, still works).

SCOPE NOTE: `hybrid_group_cleanup.py`'s own, differently-shaped destructive
delete (flagged as dead code, unreachable in the live pipeline, during
D-081's own QA_ENGINE pass) remains untouched -- still correctly out of
scope, not rediscovered as newly relevant here.

P0: 0
P1: 0
P2: 0
P3: 0

READY FOR CONTROLLED STABILITY BATTERY: YES.

Then STOP.

No Modal. No RAW.

## D-083 -- distinct-idea retry grouping safety (grouping only)

Follow-up to the D-082 CONTROLLED VIDEO00 STABILITY BATTERY's own P2 finding:
across all 3 runs, `take_judge_groups` conflated two genuinely distinct
symptom beats -- a back-acne mention treated with resorcinol, and a
hormonal-pimples-behind-the-ear/neck mention that itself appears across
three takes of increasing specificity -- into one mutually-exclusive retry
family, causing 5 identical Human Gold "pimples" check failures
(`pimples_micro_1/2/3_present`, `pimples_later_winner_present`,
`pimples_micro_order`) in every run while `acne_back_preserved` and
`pimples_bad_monolith_absent` both passed. GROUPING ONLY -- no BestTake,
Resolver, StoryValidator, Freeze, Boundary, Render/QC, or ASR change.

EXACT GROUPING-RULE TRACE: `session_boundaries.py` only computes coarse
mini-session boundaries and was ruled out directly. The deterministic
lexical mechanisms (`take_grouping.retry_similarity`/`group_takes`,
`_provider_members_compatible`, `_is_prefix_fragment`) were ruled out by
direct computation against the real (translated) clip texts -- every
acne-vs-pimples and pimples-vs-pimples pair scored 0.0 or strictly below its
respective merge threshold (the closest, the short generic pimples mention
vs. the discarded elaboration, scored 0.7032 against a 0.72 deterministic
threshold -- close, but not crossed). With every deterministic path ruled
out, all within-group pairs in the conflated baseline group became "weak
pairs" resolved entirely by `split_incohesive_retry_groups`'s (D-058 Phase
1) own arbiter call -- which, unlike its sibling
`reconcile_semantic_idea_equivalence`, applies NO content-divergence
override to any arbiter "same_idea=True" confirmation, marked or not. The
polished pimples restatement carries the exact "Otro sintoma"/"another
symptom" `_DISTINCT_ADDITION_MARKERS` marker this module already tracks
elsewhere; `split_incohesive_retry_groups` simply never consulted it.

FIX: `_within_group_arbiter_confirmation_diverges` in
`take_grouping_provider.py` -- reuses `_marked_side_diverges_in_content`
(D-048 FIX 1, already calibrated and already proven via the founding D-039
distinct-body-part case and the D-047 Case 1 genuine-restatement case)
inside `split_incohesive_retry_groups`'s own weak-pair confirmation loop,
exactly mirroring how `reconcile_semantic_idea_equivalence` already gates
its cross-group merges. Wired in as an unconditional check on every
confirmed pair: exactly one side marked AND real content divergence from
the other blocks the confirmation (logged to a new
`content_divergence_blocked` diagnostics list/count on the returned dict,
alongside the existing `arbiter_confirmed_pairs`); everything else is
unaffected.

REJECTED ALTERNATIVE (evaluated and disproven, not merely untried): a
broader, unconditional content-overlap floor on EVERY arbiter confirmation
in `split_incohesive_retry_groups` (marked or not), using claim-level
(`semantic_claims.extract_claims`) Dice/coverage/subset measures instead of
whole-text tokens. Numerically verified against this directive's own
Section 4/5/6/7 fixtures AND the real acne/pimples texts, it broke the
pre-existing `test_arbiter_confirmed_retry_stays_grouped` true-retry
paraphrase regression (`test_cutsell_d058_phase1_grouping_safety.py`): that
pair ("I had seasonal back acne ... an ointment" vs "Every season I would
get back breakouts ... an ointment for it") scores LOWER lexical/claim
overlap by every measure tried (raw shared tokens: 2; min-side coverage:
0.333; best claim-pair Dice: 0.286) than the specific unmarked pimples pair
this fix must NOT merge (shared: 4; min-side coverage: 0.800; best claim-pair
Dice: 0.421) -- proof that no fixed lexical-overlap threshold can separate
"same proposition, very different words" (an LLM arbiter's whole reason for
existing) from "different proposition, overlapping topic vocabulary" in
general. The marker is a genuine, narrow, non-lexical signal; widening the
override beyond it reintroduces exactly the false-positive class this
module's own paraphrase-retry contract already forbids.

HONEST RESIDUAL SCOPE: the fix closes every pairing that routes through the
marked restatement (acne-vs-restatement, short-mention-vs-restatement --
both proven to split even under an adversarial always-confirm arbiter in
the new test suite) and leaves the pre-existing, already-correct
monolith-vs-restatement merge untouched. The one pairing it does NOT
independently guarantee is an UNMARKED pair between two topically-similar,
asymmetric mentions (here: the short generic pimples mention vs. the
discarded longer elaboration) -- no deterministic signal tried can
distinguish that shape from a genuine unmarked paraphrase retry without
also breaking the paraphrase case. In the real 3-run battery, the more
likely single connecting edge was the marked restatement pair (the
elaboration and restatement share the identical specific location detail
and framing; the short mention shares only generic topic vocabulary with
either), so this fix is expected, but not proven offline, to fully close
the observed regression. Closing the residual with certainty would require
either a live arbiter-confirmation trace (the `distinct_idea_grouping_
safety` diagnostics key already exists in `pipeline.py` but is never
printed by the Modal RAW workflow's own diagnostic-dump script -- a real,
separate, and still-open observability gap worth a future scoped fix) or a
structured-justification arbiter contract -- both out of "grouping only"
scope for this directive.

TESTS: new `tests/test_cutsell_d083_distinct_idea_grouping_safety.py` (14
tests) -- Section 4 distinct-idea negative control (realistic-arbiter-
declines path, and a marked-side path proven safe under an adversarial
always-confirm arbiter); Section 5 true-retry positive control; Section 6
Fact-A-vs-Fact-A+B directional-completeness safety (both sides still
compete, content never silently disposed); Section 7 sales/UGC
generalization (dosage vs. benefit never compete; a benefit restatement may
compete); the real acne/pimples regression shape reproduced end-to-end
(acne-vs-marked-restatement splits even adversarially; acne vs. the two
unmarked mentions splits under a realistic declining arbiter; the genuine
monolith/restatement retry still merges; the full four-member conflated
group resolves to exactly the three intended families); diagnostics
observability (empty-groups shape, and a populated-block shape); and a
direct unit check that the new gate function requires exactly one marked
side.

QUALIFICATION: compileall clean; targeted D-050 through D-083 sweep (516
tests: the existing 502 plus this file's 14) all pass; CleanCutBench 54/54,
run twice; full `tests/` suite -- 2471 passed (2457 D-082 baseline + 14 new
D-083 tests), 2 pre-existing failures unrelated to this change (the same
`test_hybrid_story_guard_incomplete_retry.py` and `test_video00_modal_
hybrid_semantic_parity.py` cases D-081's own qualification already
identified and excluded), plus `test_semantic_stitch.py`'s pre-existing,
unrelated collection error -- 0 new regressions.

QA_ENGINE: adversarial pass asked "can two nearby distinct ideas still be
forced into one family?" -- built `ConfirmEverythingArbiter`, an arbiter
stub that confirms same_idea=True for every pair regardless of content, and
re-ran the acne-vs-marked-restatement and short-mention-vs-marked-
restatement cases against it: both still split, proving the marker +
content-divergence gate (not arbiter restraint) is what protects those
pairings. The same adversarial arbiter against the acne-vs-unmarked-mention
pairings does still force a merge -- this is the honestly-disclosed residual
above, not a hidden failure: no proven unique-content loss exists offline
(both mentions have a legitimate, independent chance to survive Selection
either way -- only the WHO-COMPETES boundary is what remains open for that
one unmarked shape).

P0: 0
P1: 0
P2: 1 -- the unmarked-pair residual described above; recommend closing via
either the `distinct_idea_grouping_safety` diagnostics observability gap
(print it from the Modal RAW workflow so a live arbiter trace can confirm
or rule out the residual directly) or a structured-justification arbiter
contract, both out of this directive's grouping-only scope.
P3: 1 -- the Modal RAW workflow's diagnostic-dump script never surfaces
`distinct_idea_grouping_safety` at all (only `semantic_idea_equivalence`),
so this whole D-058/D-083 cohesion-safety stage has been operating without
direct live observability since D-058 shipped.

DISTINCT-IDEA GROUPING: PASS (marker-mediated cases; unmarked residual
honestly disclosed, not proven safe).
PIMPLES REGRESSION: PASS expected, not certified offline (see residual
scope note above -- requires a live canary to certify with certainty).
TRUE RETRY GROUPING: PASS.
TEMPORAL-ONLY GROUPING: UNSAFE (temporal proximity alone was never
sufficient before this fix and remains insufficient after it -- both the
pre-existing deterministic thresholds and the new marker gate require
textual evidence).
UNIQUE CONTENT SAFETY: PASS.
SALES/UGC GENERALIZATION: PASS.

READY FOR ONE VIDEO00 CANARY: YES -- the fix is safe (zero regressions
across 516 targeted + 54x2 CleanCutBench + full suite), closes a real,
diagnosed defect for every marker-mediated pairing, and the one remaining
question (does the unmarked residual matter in practice) can only be
answered by observing live arbiter behavior, which is exactly what a single
canary run is for.

Then STOP.

No Modal. No RAW.

## D-083 canary (run 33940008705) -- distinct-idea grouping still conflated

Canary result: **DISTINCT-IDEA GROUPING FAIL**, **PIMPLES REGRESSION FAIL** --
identical to the pre-D-083 shape (same 5 pimples/acne checks failing, `acne_
back_preserved`/`pimples_bad_monolith_absent` still passing). D-081/D-082
retention both PASS (0 destructive deletes; no non-decisive fallback to raw
DeliveryScore). Root blocker classified live as **B -- unmarked-pair
residual**: the marker-gated D-083 fix correctly protects every pairing
involving the marked pimples restatement, but the live conflation routes
through at least one unmarked pair the fix does not independently guarantee
-- exactly the disclosed P2 residual, now observed rather than theoretical.
Not patched per directive; handed to D-084 for exact edge-level forensics.

## D-084 -- retry-family bridge edge forensic (report only, no code/GPU/RAW)

Recovered the exact live edge-level evidence for the D-083 canary's
conflated 5-member family via the zero-GPU D-044 forensic-extract workflow's
`trace_clip_ids` (D-045), reading `distinct_idea_grouping_safety` directly
from the persisted S3 `result.json` for the canary AND all 3 D-082 battery
runs (the key existed all along -- it was simply never printed by the Modal
RAW workflow's own dump script, a real, separate observability gap).

EXACT BRIDGE: `acné2 <-> pimples-monolith`, arbiter confidence 0.85
("Connecting back acne to specific neck and ear pimples."), plus a second,
redundant, independent false bridge `acné2 <-> pimples-short` at 0.80. Both
unmarked -- D-083's marker gate correctly protected the one pairing
involving the marked restatement (blocked, confirmed in diagnostics), but
these two edges never touch it. ROOT CLASS: DIRECT_FALSE_EDGE, on top of a
pre-existing BASELINE_GROUP_ALREADY_CONFLATED precondition.

Cross-run comparison: the identical `acné2<->monolith` bridge, at the
identical 0.85 confidence, was independently confirmed in **all 4** runs
checked (3 pre-D-083 D-082-battery runs plus the D-083 canary) -- every true
within-beat edge scored 0.90-0.98 in every run, every false cross-beat edge
scored 0.80-0.85 in every run. Classified **DETERMINISTIC_GROUPING_DEFECT**,
not LLM edge variance.

Proven from source: `_cohesive_components` is plain union-find with no
group-wide cohesion requirement -- pairwise/transitive connectivity alone is
sufficient today; the arbiter's own pairwise judgments are not even
internally coherent (it never confirmed acné2<->restatement directly, the
pair a coherent single-proposition read would also reject, yet transitivity
through the monolith puts them in one family anyway).

Recommended anti-bridge contract: bridge-edge validation (a materially
higher evidentiary bar only when an edge would union two already-
established, multi-member subclusters), informed by a canonical-proposition
check -- leaving singleton/small-cluster merges (the common true-retry case)
untouched. This became D-085's implementation.

READY FOR IMPLEMENTATION: YES. Then STOP (no code changed by D-084 itself).

## D-085 -- bridge-aware retry family cohesion (grouping only)

Implements D-084's recommended anti-bridge contract. Root defect (D-084's
own live, cross-run-verified forensic): `_cohesive_components` (D-058) is
plain union-find over an unordered edge set -- any accepted pairwise edge
can transitively connect two full components, with no requirement that the
RESULTING merged component still represent one shared audience-facing
proposition. Proven reproducible: the same 0.85-confidence cross-beat bridge
recurred in 4/4 independent live runs.

SESSION CLUSTER vs RETRY FAMILY (formalized in `take_grouping_provider.py`'s
own module comment): a session cluster -- everything upstream of this
module's final cohesion pass -- is only ever a bounded neighborhood of
clips worth comparing; temporal/topical proximity is never by itself
evidence of one retry family (a mutually-exclusive set of realizations
competing to express ONE proposition).

IMPLEMENTATION (`cutsell_worker/take_grouping_provider.py`):
- `_RetryEdge`/`_edge_sort_key`: every candidate edge (deterministic lexical
  match, or an arbiter same_idea confirmation already surviving D-083's
  unchanged marker gate) is processed in a fixed order independent of input
  order -- deterministic edges first, then semantic edges by descending
  confidence, then a clip-id tie-break. Proven input-order-independent by
  dedicated permutation tests (edge list order, group member order).
- `_bridge_aware_components`: incremental union-find tracking live
  component membership. An edge is a BRIDGE the instant either endpoint's
  CURRENT component already has >=2 members (two established components
  merging, or a singleton attaching to an existing multi-member one) -- a
  first-time singleton<->singleton merge is unaffected, byte-identical to
  pre-D-085 behavior.
- `_evaluate_bridge_cohesion`: a bridge is NEVER accepted on its own
  triggering pair's same_idea/confidence alone. It must additionally clear:
  (1) a deterministic `any_pair_contradicts` cross-component safety net
  (the same primitive D-082 already trusts for exactly this role -- no new
  authority), then (2) a bounded, component-level question posed to the
  SAME already-configured `SemanticEquivalenceArbiter` (no new provider/
  model), built from both components' own member texts (`_component_probe_
  text`, sorted by clip id for full determinism) rather than just the two
  touching clips, requiring `same_idea=True` at >= 0.90 confidence
  (`_BRIDGE_MIN_COHESION_CONFIDENCE`, calibrated directly against D-084's
  own numbers: true edges 0.90-0.98, false bridges 0.80-0.85). Absent/
  malformed/declined/low-confidence responses fail closed -- rejected, not
  merged. `shared_proposition`/`member_support` are populated only from that
  same arbiter call's own output; `distinct_required_facts` only from the
  existing contradiction primitive -- no fabricated semantic authority.
- D-083's marker gate is completely unchanged (still the only thing that
  can block an edge before it ever becomes a candidate for D-085's own
  bridge machinery) -- D-085 complements it, confirmed by dedicated tests.
- Modal RAW workflow (`cutsell-video00-modal-raw.yml`): now prints
  `diagnostics.distinct_idea_grouping_safety` directly (previously never
  printed at all -- the exact gap D-084's own forensic had to work around
  via the S3 side-channel). Closed, not repeated.

TESTS: new `tests/test_cutsell_d085_bridge_aware_retry_family_cohesion.py`
(25 tests) -- the exact D-084 5-member regression (two independent false
bridges, generic fixture, must not reconnect); edge-order and group-member-
order permutation invariance; bridge classification unit tests; component-
cohesion fail-closed cases (no arbiter, arbiter exception, declined,
low-confidence, high-triggering-confidence-still-rejected); D-083 marker
retention; true-retry safety (low-lexical-overlap paraphrase, directional
superset, singleton joining a coherent component, two true subclusters with
a genuine shared proposition merging); narrative-continuation safety
(adjacent chronology, different events); sales/UGC generalization (dosage/
outcome/product-use/experience/hook/CTA never bridge; a benefit restatement
still may); bounded-compute diagnostics; no winner/selection-authority keys
ever surfaced by grouping; and the QA_ENGINE-mandated always-YES-pairwise-
but-skeptical-component-check adversarial proof that the two-tier design,
not pairwise confidence, is what prevents the false bridges.

QUALIFICATION: compileall clean; D-050 through D-085 targeted sweep 526/526;
D-058/D-083 grouping suites + historical complete-retry-identity-guard
suites (`test_cutsell_complete_retry_identity_guard.py`, `test_cutsell_
d072_complete_retry_observability.py`) all pass; CleanCutBench 54/54, run
twice; full `tests/` suite -- 2495 passed (2471 D-083 baseline + 24 new
D-085 tests), same 2 pre-existing unrelated failures D-081 already
identified and excluded, plus `test_semantic_stitch.py`'s pre-existing
collection error -- 0 new regressions.

QA_ENGINE: the always-YES pairwise arbiter attack (every two-clip query
confirmed at uniform high confidence, with only the component-level probe
questions answered honestly) still correctly separates the two families --
proving pairwise confidence/vote count was never load-bearing in the fix.
Edge-order and group-order permutation attacks produce byte-identical
families. Candidate/group explosion: bridge evaluations are bounded well
below all-pairs (observed 3-4 bridge evaluations against 10 possible pairs
in the 5-member D-084 fixture); no unbounded re-evaluation.

GROUPING SELECTS WINNER: NO (unchanged -- BestTake/Resolver remain sole
winner authorities; grouping only decides who competes).
POST-RESOLVER SEMANTIC MUTATORS: 0 (unchanged from D-081's own count).

P0: 0
P1: 0
P2: 0
P3: 0

READY FOR ONE VIDEO00 CANARY: YES.

Then STOP.

No Modal. No RAW.

## D-085 canary (run 33952982672) -- bridge-aware cohesion PROVEN live

Head `2ad6203`. `diagnostics.distinct_idea_grouping_safety` now prints in the
main RAW log: `bridge_evaluated_count: 7`, `bridge_accepted_count: 1`,
`bridge_rejected_count: 6`, `groups_checked: 5`, `groups_split: 3`. Both
exact D-084 false bridges (back-acne <-> pimples-monolith at cohesion 0.1,
back-acne <-> short-pimples with the arbiter 0.95-confident they are
distinct) were REJECTED; the one accepted bridge reunited two true retries of
the pimples idea. Back-acne and neck/ear-pimples resolved as two separate
families (CONFLATED: NO). D-085 LIVE OBJECTIVE: PROVEN. Human Gold 16/18
(`pimples_micro_2_present` + `pimples_micro_order` fail: the discarded first
pimples mention's unique "rush" detail is not restated by the winning take --
a pre-existing BestTakeResolver completeness gap, not D-085's). Freeze
BLOCKED on ONE unrelated family (`tg_393d421c605ad934cd`,
`unresolved_unique_fact_asymmetry`) -- classified C, forensic'd as D-086.

## D-086 -- papillary diagnosis / hindsight family forensic (report only)

Recovered from the persisted S3 result via two zero-GPU forensic-extract runs
(`33954934990`, `33955123472`). A = diagnosis sentence + fumbled hindsight
clause (135.44-149.52, `real_0c9165624e0030d03ae3`); B = clean hindsight
lead-in (150.68-158.14, `real_98ec88ff1c0a3488cc35`); 1.16 s apart;
`retry_similarity` 0.0 -- grouped only by the SemanticArbiter on the shared
hindsight theme. CRITICAL claim sets disjoint (A: DIAGNOSIS_IDENTIFICATION;
B: CONTRASTIVE_HINDSIGHT_NEGATION) -> D-063 dominance correctly None ->
D-082 `unresolved_unique_fact_asymmetry`. Human Gold keeps BOTH consecutively
(gold spans 11 -> 12). SEMANTIC RELATION: COMPLEMENTARY_DISTINCT_IDEAS;
CUTSELL SHOULD: KEEP BOTH SEPARATELY; HUMAN CHOICE: NO; D-085 bridge check:
NOT_APPLICABLE (singleton-to-singleton edge).

Decisive finding: the authoritative RealizationResolver (mode AUTHORITATIVE,
status SEMANTICALLY_RESOLVED) had ALREADY resolved the family as
`RESOLVED_COMPOSITE [A, B]` ("minimal_composite_covers_all_critical_claims",
all 5 canonical claims covered, zero missing) -- but `CanonicalEditPlan`
derived `is_composite` only from legacy composite evidence, relabeled the
pair `unresolved_ambiguous`, and FinalEditReviewer blocked Freeze with
DUPLICATE_IDEA + UNRESOLVED_RETRY. ROOT CLASS: GROUPING (compound
realization attached through a non-critical shared clause) with the plan-
handoff seam as the mechanism that turned a correct resolution into a block.
IF CORRECTED, FREEZE WOULD PASS: YES. Note for future work: forwarding the
D-061 claim-equivalence arbiter into pipeline.py's D-082 ladder is NOT the
fix -- the legacy path that did consult it discarded B (the wrong answer).

## D-087 -- authoritative composite -> CanonicalEditPlan handoff (single truth)

SINGLE-TRUTH CONTRACT: in AUTHORITATIVE resolver mode the Unified Realization
Resolver decides winner / composite / review-required; `CanonicalEditPlan`
REPRESENTS that decision (plus structural validation); FinalEditReviewer
validates; Freeze gates. No new semantic authority, no post-resolver
semantic mutator.

IMPLEMENTATION (`cutsell_worker/canonical_edit_plan.py`):
- `AuthoritativeIdeaDecision` / `AuthoritativePlanSource` -- the resolver's
  own per-idea verdict (status, winner/composite realization ids, the
  Ledger's candidate realizations, covered/missing canonical claims,
  decision reason), built ONLY in universal_clean_cut.py's AUTHORITATIVE
  branch by `build_authoritative_plan_source(authoritative_result, ledger)`,
  stored as `diagnostics["authoritative_plan_source"]`, and passed
  explicitly through `run_repair_loop(..., authoritative_source=)` to every
  `build_canonical_edit_plan` call (v1 and each repaired v2/v3).
  `build_canonical_edit_plan` also reads the stored key back when called
  without the object (live_render_qc's plan re-resolution), so the plan
  Freeze recorded and the plan Render QC cross-checks agree.
- A `RESOLVED_COMPOSITE` idea whose members pass
  `_validate_authoritative_composite` is emitted `is_composite=True`,
  `coverage_status="complete"`, `winning_clip_ids` = the authoritative
  members in the resolver's own order (never re-sorted by DeliveryScore or
  clip id), with `plan_semantic_source="authoritative_realization_resolver"`,
  `authoritative_resolution_status`, `authoritative_composite_realization_
  ids`, `authoritative_resolved_clip_ids`, `authoritative_claim_coverage`,
  `authoritative_decision_reason`, `structural_validation_passed/_failures`
  on the idea and `plan_semantic_source` on the plan.
- Structural validity checks, all fail-closed (idea stays
  `unresolved_ambiguous` so Freeze still blocks, with the failures carried
  on the idea -- BLOCK and explain, never an invented alternative): empty
  composite; duplicate member ids; status not RESOLVED_COMPOSITE; missing
  critical claims non-empty; unknown realization; realization not selected;
  member outside the idea's Ledger candidate set; member outside the
  take_judge group; member clip stamped with another idea; stale fragment
  provenance (`parent_realization_id` disagreeing, or a parent-only fragment
  whose parent vanished); a selected member the composite does not name;
  factual contradiction between members (the D-056.3 shared contract). A
  `RESOLVED_WINNER` idea whose survivors are not exactly the winner's clips
  is likewise blocked and explained (never a silently different winner).
- Fragment/provenance (D-046/D-050A): realization -> clip mapping uses
  `realization_id` (fragments carry the parent's), then
  `parent_realization_id`, then a parent-only `parent_semantic_clip_id`
  resolved through the parent still present in the draft; a member
  `take_judge_groups` names by its pre-split clip id is explained by its
  fragments' realizations.
- LEGACY/SHADOW: no source is ever built or stored; `build_canonical_edit_
  plan(draft)` is byte-identical to pre-D-087 (legacy composite evidence
  path untouched; new fields defaulted).
- `final_edit_reviewer.py`: a family the resolver resolved but the plan
  structurally rejected reports DUPLICATE_IDEA/UNRESOLVED_RETRY with reason
  `authoritative_resolution_structural_validation_failed` + the failure
  list, owning authority `CanonicalEditPlan(structural_validation)`; a
  genuinely unresolved family keeps the pre-D-087 findings verbatim.
- STORY ORDER (Section 12): `realization_resolver._place_restored_clips_at_
  story_position` -- a clip the authoritative application restores into
  `selected` lands at its idea's story position (adjacent to its kept
  composite sibling in recording order; or in the departing legacy winner's
  slot for a winner replacement; appended only with no sibling context),
  instead of after the CTA as on run 33952982672. Membership is untouched
  -- positional only; resolver decision logic unchanged.
- RAW workflow prints `plan_semantic_source` + per-idea authoritative/
  structural fields and `diagnostics.authoritative_plan_source`.

TESTS: `tests/test_cutsell_d087_authoritative_composite_plan_handoff.py`
(39): Section 8 generic diagnosis+hindsight regression (direct, via the
repair loop, unstamped-mint mapping, and END TO END through the real Ledger
-> real resolver -> real application -> plan source -> plan/reviewer/repair
loop); Section 9 controls (REVIEW_REQUIRED, no authoritative decision,
RESOLVED_WINNER with two survivors -> still blocked); Section 10 malformed
composites (missing id, not selected, wrong idea, outside candidate set,
missing critical claim, contradictory, empty, duplicate ids, wrong status,
stale provenance x2, extra unexplained member) all fail closed; Section 11
fragments (D-050A split fragments; legacy parent-only provenance); Section
12 story order (resolver order over DeliveryScore/clip id; restored member
placed consecutively after/before its sibling; winner-replacement slot;
no-sibling append; membership invariance); Section 13 (valid composite is
PASS, never NEEDS_HUMAN_REVIEW); Section 14 sales/UGC (benefit + dosage
composite; hook + CTA distinct winners); Sections 5/15/16 (legacy shape
unchanged, diagnostics round trip + draft-key fallback, explicit source
precedence, never changes membership, LEGACY/SHADOW carry no key end to
end, AUTHORITATIVE carries and consumes it end to end).

QUALIFICATION: compileall clean; D-046 through D-087 targeted sweep 636/636;
CleanCutBench 54/54 LEGACY and 54/54 AUTHORITATIVE; full `tests/test_
cutsell_*.py` glob 1910/1910; whole `tests/` directory 2534 passed (2495
D-085 baseline + 39 new D-087 tests), the same 2 pre-existing unrelated
failures D-081 already identified and excluded (`test_hybrid_story_guard_
incomplete_retry`, `test_video00_modal_hybrid_semantic_parity` overlay-
masking assertion) -- 0 new regressions.

POST-RESOLVER SEMANTIC MUTATORS: 0 (CanonicalEditPlan represents, the
repair loop reorders only, StoryValidator/FinalEditReviewer validate).

P0: 0 / P1: 0 / P2: 0 / P3: 1 (a realization with clips split across
selected AND discarded buckets is represented from its selected pieces
only -- the D-046 physical-fragment shape -- rather than failed closed;
StoryValidator's independent lost-claim check still covers content loss).

READY FOR ONE VIDEO00 CANARY: YES.

## D-087 canary (run 33957582102) -- authoritative composite consumption PROVEN live

Head `a181273`. The D-086 family (`tg_b3fd0910f3dfc18d4e` = `idea_838ff338…`)
resolved `RESOLVED_COMPOSITE [A, B]`; CanonicalEditPlan emitted
`is_composite: true`, `coverage_status: complete`, `plan_semantic_source:
authoritative_realization_resolver`, `structural_validation_passed: true`,
members in resolver order; ZERO DUPLICATE_IDEA / UNRESOLVED_RETRY findings
(the two that blocked run 33952982672). 19/19 authoritative decisions
represented verbatim. D-087 LIVE OBJECTIVE: PROVEN. Human Gold 15/18
(`pimples_micro_2_present`, `pimples_bad_monolith_absent` -- arbiter
variance picked the monolith this run --, `pimples_micro_order`). Freeze
BLOCKED on ONE unrelated finding: `CRITICAL_CLAIM_LOST` on the gastritis
family (`tg_cb6bda4604271b263f`, "…no hay que preguntar"), classified C.
STORY ORDER: composite relative order preserved but NOT contiguous -- the
acne winner replacement landed between A and B and split "…resolvía con" |
"resorcina." (D-087-owned placement bug). Both forensic'd as D-088.

## D-088 -- claim-proof + story-order integration forensic (report only)

PART A: claim `claim_fabaabad04cf` / `cclaim_0d1dbd02d419041608f2`
("Tuve problemas de estómago en una temporada, en 2023, no hay que
preguntar.", whole sentence, NEGATION/FACTUAL_NEGATION, raw CRITICAL) from
`real_6db31c8d3122ce4b53d8`. Canonical id identical across StoryValidator
extraction and the Ledger (no drift). Ledger requirement group downgraded it
to SUPPORTING (`_effective_importance`: incidental + source-exclusive); the
resolver resolved `RESOLVED_WINNER` with `missing_critical_claim_ids: []`
and kept the loser as a contextual alternate; ClaimCoverageBestTake
suppressed its override as incidental; `_lost_critical_claims` re-derived
raw CRITICAL and blocked Freeze. The D-079 intra-idea proof was minted but
unverified (`required_claim_not_preserved`): after the rhetorical-aside strip
the residual and the winner's claim are same-type, so only deterministic
dedup applies (2/6 tokens, no arbiter route reachable). FIRST MISSING LINK:
G. DUAL-TRUTH INVARIANT (verified proof + finding): not violated. CLAIM
BLOCKER ROOT: importance dual-truth between the Ledger/Resolver and
StoryValidator. Intra-idea proofs were also not surfaced in diagnostics.

PART B: legacy keep `A[9], old-acne[10] (166.56-182.36), "resorcina."[11]
(191.14)`; restored `[B, new-acne]` in discarded-bucket order. B was placed
after A (composite rule); the acne replacement was anchored "after the last
ORIGINAL kept predecessor" (= A) -- invisible to the already-placed B -- and
inserted at 10, pushing B to 11: `A, new-acne, B, resorcina`. Replayed
offline with the production function: reproduces the live order exactly.
STORY-ORDER ROOT: A (multiple restorations sharing one anchor; departed slot
reduced to "after predecessor"). Fixes required: 2. Both implemented as D-089.

## D-089 -- effective claim importance single truth + authoritative story placement

PART A (`realization_resolver.build_effective_claim_importance_index` /
`build_effective_claim_importance_diagnostics`, `final_story_coherence_
validation._lost_critical_claims`): a deterministic canonical_claim_id ->
`EffectiveClaimImportance` index (raw importance, effective importance,
reason, semantic idea, requirement group, source realizations, source-
exclusive flag) built per idea from the Ledger's OWN `build_requirement_
groups` -> `_effective_importance` (the identical rule the resolver and
ClaimCoverageBestTake already honor; no second classifier). A canonical id
receiving two different authoritative answers across ideas fails closed to
CRITICAL (`cross_idea_conflict_fail_closed`). `_lost_critical_claims`
consumes it AFTER the D-079 proof lookup: only when the EXACT canonical id
is present, its effective importance is non-critical, AND the entry belongs
to this group's own idea (stamped `semantic_idea_id` or the deterministic
mint from the group id) and, when realization ids are stamped, shares a
source realization with the group's members. It then records a
`claim_coverage_confirmations` row with `critical_loss_suppressed_by:
canonical_effective_importance`, raw/effective importance and the reason,
instead of a CRITICAL_CLAIM_LOST finding. Absent id, wrong id, foreign idea,
or effective CRITICAL -> unchanged fail-closed finding. Built and threaded
only in universal_clean_cut.py's AUTHORITATIVE branch (`diagnostics
["canonical_effective_importance"]`, downgraded/conflict entries only);
LEGACY/SHADOW byte-identical.

PART B (`realization_resolver._place_restored_clips_at_story_position`,
rewritten): restorations are deterministic units applied to the CURRENT
sequence in stable order -- `AUTHORITATIVE_COMPOSITE_BLOCK` (every selected
member of an idea with a resolver composite order or a kept sibling, placed
atomically, in the resolver's member order -- recording time only orders
members with no explicit order -- at the earliest original member's
position), then `SINGLE_WINNER_REPLACEMENT` (the departed winner's ACTUAL
slot: immediately before the first ORIGINAL successor still in the sequence,
never inside a placed block; only with no surviving successor, after the
last surviving original predecessor), then `NO_ANCHOR_APPEND`. Blocks
sorted by anchor position then idea id; replacements by departed original
index then idea id; appends by start then clip id -- output independent of
the restored bucket's iteration order. Membership untouched by
construction. Every unit is logged (unit type, member clip/realization ids,
authoritative member order, departed clip id + original index, successor
anchor, predecessor fallback, chosen insertion index, sequence before/after,
contiguity validated, placement reason) on `AuthoritativeApplicationResult.
story_placement` -> `diagnostics["authoritative_story_placement"]`, printed
by the RAW workflow. The D-088 shape now yields `A, B, new-acne,
"resorcinol."` for either restoration order. Resolver decision logic,
composite membership, winners: unchanged.

TESTS: `tests/test_cutsell_d089_effective_importance_and_story_placement.py`
(31): Part A -- exact id + effective SUPPORTING suppresses (with the pre-
D-089 baseline finding proven on the same draft); exact id + effective
CRITICAL keeps the finding; missing id / None index fail closed; wrong id
and mismatched entry never suppress; same text/different proposition never
suppresses; corroborated (non-source-exclusive) incidental negation stays
CRITICAL; genuine negation / diagnosis negation / measurement / entity
identification / attribution-negation losses keep effective CRITICAL and
block exactly as baseline; number / entity / causal mismatch propositions
are never downgraded below raw; attribution asymmetry stays blocking and
uncertifiable by the proof chain; D-079 proof consumption unchanged (and the
aside shape still yields an unverified proof -- the index, not a proof,
closes it); foreign-idea entry never downgrades; cross-idea conflict fails
closed; StoryValidator pass threads the index; AUTHORITATIVE carries both
new diagnostics keys and no CRITICAL_CLAIM_LOST on the aside shape while
LEGACY keeps its finding and no keys. Part B -- D-088 exact generic order
for both restoration orders; contiguity + explicit order over recording
time; successor-anchored replacement; predecessor fallback; two
replacements sharing one predecessor; multiple blocks; replacements before/
after a block never split it; continuation adjacency; full permutation
invariance; membership invariance; END TO END through the real Ledger ->
resolver -> application with placement diagnostics.

QUALIFICATION: compileall clean; D-046 through D-089 targeted sweep 680/680;
CleanCutBench 54/54 LEGACY and 54/54 AUTHORITATIVE; full `tests/test_
cutsell_*.py` glob 1941/1941 (1910 D-087 baseline + 31 new); whole `tests/`
directory 2565 passed, the same 2 pre-existing unrelated failures D-081
already identified and excluded -- 0 new regressions.

POST-RESOLVER SEMANTIC MUTATORS: 0 (the index reports the resolver's own
importance; placement reorders already-selected clips only).

READY FOR ONE VIDEO00 CANARY: YES.

## D-090 -- post-resolver StoryValidator immutability (authority boundary)

**Trigger:** D-089 canary run 33960713625 (engine 40dde20). Resolver emitted
RESOLVED_COMPOSITE [`real_40a2720d8631ff22ab43`, `real_bd71970e8f6d76052093`]
for the hereditary family; StoryValidator's residual-family resolution (arbiter
merge 0.9) then discarded one member; CanonicalEditPlan correctly failed closed
(`realization_not_selected`); Freeze BLOCKED by a post-resolver semantic
membership mutation -- two authorities disagreeing on one family's membership.

**Root cause:** AUTHORITATIVE mode's second StoryValidator pass was the SAME
legacy resolving code as the pre-resolver pass (`apply_final_story_coherence_
validation`), so it re-asked the semantic-equivalence arbiter "same idea?" for a
family the resolver had already ruled a complementary composite, and acted on
the answer. Mutation points in that pass, enumerated: (1) alternates fold into
discarded; (2) residual multi-select family collapse (`_resolve_residual_family`
-> `discard_ids` -> `replace(draft, selected=..., discarded=...)`). Nothing else
in the module edits membership.

**Decision (implemented, offline only -- no Modal, no RAW):**

1. `post_authority_validation.py` (new): `PostAuthorityValidationContext` --
   the explicit typed contract built ONLY from the applied
   `AuthoritativeApplicationResult` + the D-087 `AuthoritativePlanSource`
   (status/decision consistency checked; `PostAuthorityIntegrityError` fails
   closed). `semantic_selection_signature` (ordered clip id, realization id,
   fragment parent, semantic idea, source span, canonical speech digest,
   authority identity) + `compare_selection_signatures` = the executable
   invariant: order-sensitive after StoryValidator, order-insensitive
   (membership/speech/provenance/authority) after the bounded repair loop.
2. `final_story_coherence_validation.apply_post_authority_story_validation`
   (the one AUTHORITATIVE second-pass entry point; `universal_clean_cut.py`
   imports and calls it). Mode is decided by the typed context alone -- never
   by a diagnostics key, two selected clips, or a composite label. With the
   context: no residual collapse, no arbiter call for membership, alternates
   folded on a working copy for the coverage checks' VIEW only (buckets
   returned untouched), contradiction / idea-coverage / lost-atom / lost-claim
   checks unchanged (D-076/D-079 proofs and D-089 index still consumed).
   Family bookkeeping: `canonical_edit_plan.assess_authoritative_membership`
   (extracted from `build_canonical_edit_plan`, which now consumes it too) --
   a structurally valid RESOLVED_COMPOSITE/RESOLVED_WINNER is
   `authoritative_families_accepted`; REVIEW_REQUIRED, missing/unselected/
   wrong-idea/extra member, missing critical claims, contradiction, or absent
   decision -> `authority_membership_findings` (blocking) + `unresolved_
   families`. Missing context -> `status: integrity_failure`,
   `freeze_blocked: true`, draft untouched, NEVER the legacy resolving pass.
   Without the context the legacy function is byte-for-byte unchanged
   (LEGACY/SHADOW and the first AUTHORITATIVE pass); its diagnostics gain one
   additive key `validation_mode: legacy_resolving`.
3. `universal_clean_cut.py` AUTHORITATIVE branch: context built, signature
   captured after authoritative application, validation invariant checked
   after StoryValidator, repair-projection invariant checked after
   `run_repair_loop`; `diagnostics["post_authority_validation"]` records
   mode, context status, source identity, all three signatures, both
   invariants, membership added/removed, speech/order/provenance changes,
   accepted families, blocking findings. Any drift -> `stage_status.post_
   authority_integrity_failure: true`, Freeze BLOCKED, `selection_boundary_
   contract.status: not_frozen_post_authority_integrity_failure`; the mutated
   draft is reported as-is (never silently restored) and the authoritative
   source is never rebuilt from it. Workflow prints both new blocks.

**Retired claim:** "StoryValidator is the last semantic authority allowed to
touch membership" -- true only of the legacy resolving pass. In AUTHORITATIVE
mode the ONE semantic Selection authority is the Unified Realization Resolver,
applied once by `apply_authoritative_realization_resolution`; everything after
it validates, represents, or reorders physically, and the D-090 invariant
proves it.

**RED reproduction (real import path, generic fixture -- no live texts/ids):**
two same-claim-type CRITICAL diagnosis claims in one retry family (legacy
ClaimCoverageBestTake declines the 2-piece composite on `types_a & types_b`
exactly as live; resolver composites [A, B]) + an always-merge arbiter: before
the fix StoryValidator discarded A, plan `realization_not_selected:real_A`,
reviewer DUPLICATE_IDEA+UNRESOLVED_RETRY, NEEDS_HUMAN_REVIEW, Freeze BLOCKED.
After: composite intact, plan complete, reviewer PASS, Freeze not blocked,
arbiter never asked. Tests: `tests/test_cutsell_d090_post_authority_
validation_immutability.py` (29: RED/GREEN reproduction, full-path composite,
valid winner, fragment identity, REVIEW_REQUIRED / contradictory / missing /
extra-member / no-decision blocking without rewrite, claim+atom loss active,
alternates untouched, explicit mode, missing context fails closed (unit + full
path), injected mutation in validation and in repair trips the invariant and
is not restored, pure reorder passes the repair projection, signature
speech/provenance/order/authority detection, D-089 placement + effective-
importance retention, LEGACY/SHADOW parity, real wrapper path, shared
assessment).

**Out of scope, unchanged:** resolver decision logic, grouping/D-085,
BestTake scoring, D-089 placement, effective-importance classification, Human
Gold, Freeze/delivery policy, Boundary/Render/QC, Human Choice/SWAP.

QUALIFICATION (offline): compileall clean (cutsell_worker, tests, benchmarks);
StoryValidator / CanonicalEditPlan / repair-loop / D-046 / D-050C2-C3 / D-050D1
/ D-056.3 / D-061 / D-076 / D-079 / D-087 / D-089 suites 258/258 + plan sweep
195/195; new D-090 file 29/29; CleanCutBench 54/54 LEGACY and 54/54
AUTHORITATIVE; full `tests/test_cutsell_*.py` glob 1970/1970 (1941 D-089
baseline + 29 new); whole `tests/` directory 2594 passed (= 2565 D-089
baseline + 29), the same 2 pre-existing unrelated failures D-081 identified
(`test_hybrid_story_guard_incomplete_retry`, `test_video00_modal_hybrid_
semantic_parity`) plus the pre-existing `tests/test_semantic_stitch.py`
collection error (module-level `score_take()` call, untouched since 8077aa4)
-- 0 new regressions.

QA_ENGINE (self-review by the implementing session; NOT independently
staffed): primary attack "can any post-authoritative validator still remove /
restore / replace / reinterpret a selected realization?" -- StoryValidator
post-authority pass returns `replace(draft)` (buckets untouched) and takes no
semantic-equivalence arbiter at all; the only other post-authority stage
before Freeze is the bounded repair loop (pure reorder), covered by the order-
insensitive invariant; both invariants are exercised by injected mutations.
Secondary attack "does read-only suppress real contradictions or losses?" --
`_contradiction_findings` checks EVERY still-selected pair per family
deterministically (a superset of the arbiter-gated rows the legacy collapse
produced); atom/claim/idea-coverage checks run on the same folded view as
before; unresolved families now BLOCK inside StoryValidator too (they were
already blocked by the plan/reviewer, so no new false block). Findings
introduced by D-090: P3 -- the authority-identity component of the signature
is tautological in-pipeline (one source object, one digest string), it only
guards a caller that rebuilds the source; P3 -- LEGACY diagnostics gain one
additive key (`validation_mode`). Historical open: D-087 P3 (split-bucket
realization represented from selected pieces only). POST_RESOLVER_SEMANTIC_
MUTATORS in the AUTHORITATIVE path: 0 (was 1: StoryValidator residual collapse).

## Change rule

When a new decision changes product behavior, update this file in the same development cycle. Do not silently redefine CutSell through code alone.