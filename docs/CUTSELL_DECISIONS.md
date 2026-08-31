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
one winner. Full `tests/test_cutsell_*.py` CI glob green, `compileall` clean. Not
pushed, per the same "propose and hold" instruction as D-038's own negation-guard
follow-up above -- awaiting explicit authorization before the next RAW.

## Change rule

When a new decision changes product behavior, update this file in the same development cycle. Do not silently redefine CutSell through code alone.